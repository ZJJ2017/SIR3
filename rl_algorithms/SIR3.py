import os
import time
import numpy as np

import pickle

import torch
import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim


from rl_algorithms.TD3fD import TD3fD
from rl_algorithms.gail import Discriminator

from utils.helper_functions import numpy2floattensor, numpy2floattensorList
from utils.buffer import ReplayBuffer, DynReplayBuffer, PrioritizedBufferWrapper
from utils.expUtils import demo_re_labeling, demo_number_control, read_traj_data, split_ratio
from utils.expUtils import buildSchedule
from utils.expUtils import Log


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SIR3(TD3fD):
    def __init__(self, td3_kwargs, cfg):
        super(__class__, self).__init__(td3_kwargs, cfg)

    def _initialize(self):
        """Initialize non-common things."""
        self.per_beta = self.hyper_params.per_beta
        self.use_n_step = self.hyper_params.nStep > 1

        self.pretrainDone = False
        self.episode_r = 0.
        self.episode_buffer = []
        self.success_np = np.zeros((100), dtype=int)
        self.ep_r_np = np.zeros((100), dtype=float)
        self.episode_cnt = 0
        self.expert_lowest_rewards = self.hyper_params.rewardThreshold

        self.loss_episode = list()

        self.log = Log(os.path.join(self.exp_path, "e.log"))

        if self.hyper_params.useDiscriminator:
            self.discr = Discriminator(
                self.env.observation_space.shape[0] + self.env.action_space.shape[0], 100,
                device)

        if not self.is_test:
            demos = []
            # load demo replay memory
            demo_path = self.hyper_params.demo_path
            if 'npz' in demo_path:
                demo_obs, demo_actions, demo_rewards, demo_dones, demo_next_obs, demo_episode_scores = read_traj_data(self.hyper_params.demo_path)
                for i in range(len(demo_obs)):
                    if i+1 == self.env.max_episode_steps: demo_dones[i] = True
                    demos.append((demo_obs[i], demo_actions[i], demo_rewards[i], demo_next_obs[i], demo_dones[i]))
            else:
                with open(demo_path, "rb") as f:
                    demos = pickle.load(f)
            self.log.record(f"Total demos: {len(demos)}")
            self.log.record("---------------------------------------")
            # Demo processing and re-labeling
            demos, ep_r_his = demo_re_labeling(self.env, demos, self.hyper_params.reLabeling, self.hyper_params.overTimeDone, self.hyper_params.reLabelingDone, self.hyper_params.reLabelingSuccess, self.hyper_params.reLabelingSqil)
            # Control the number of tracks and copy the number of tracks to the smallest batchsize
            if self.hyper_params.expertTrajN is not None:
                demos, ep_r_his, ep_index, step_cnt, copy_cnt, temp_traj = demo_number_control(demos, ep_r_his, self.hyper_params.expertTrajN, self.hyper_params.batch_size*self.hyper_params.sampleP)
                if copy_cnt > 0:
                    self.log.record(f"Used demo: {ep_index}, {step_cnt}")
                    self.log.record(f"-> Replicate {copy_cnt/len(temp_traj):0.2f} copies in batch, {step_cnt+copy_cnt}")
            if len(demos) > 0: self.log.record(f"Average rewards: {sum(ep_r_his)/len(ep_r_his):0.3f}")
            self.log.record("---------------------------------------")
            self.log.record(f"After processing: {len(demos)}")
            if self.use_n_step:
                # replay memory for multi-steps
                self.memory_n = ReplayBuffer(
                    max_len=self.hyper_params.buffer_size,
                    batch_size=self.hyper_params.batch_size,
                    n_step=self.hyper_params.nStep,
                    gamma=self.hyper_params.gamma
                )
            # replay memory for a single step
            self.memory = DynReplayBuffer(
                self.hyper_params.buffer_size,
                self.hyper_params.batch_size
            )
            # export buffer
            self.expert_memory = DynReplayBuffer(
                self.hyper_params.expert_buffer_size,
                self.hyper_params.batch_size,
            )
            if not self.hyper_params.trainWithoutPER:
                self.memory = PrioritizedBufferWrapper(
                    self.memory,
                    alpha=self.hyper_params.per_alpha,
                    epsilon_d=self.hyper_params.per_eps_demo,
                )
                self.expert_memory = PrioritizedBufferWrapper(
                    self.expert_memory,
                    alpha=self.hyper_params.per_alpha,
                    epsilon_d=self.hyper_params.per_eps_demo,
                )
            ep_index = 0
            for i, item in enumerate(demos):
                if self.hyper_params.useDiscriminator:
                    state, action, reward, nextState, done, t_done = item
                    temp_r = self.discr.predict_reward(
                        torch.FloatTensor(np.array([state])).to(device),
                        torch.FloatTensor(np.array([action])).to(device), self.hyper_params.gamma, 1)
                    temp_r = temp_r[0][0].cpu().numpy().tolist()
                    item = (state, action, temp_r, nextState, done, t_done)
                if self.hyper_params.expertTrajN is not None:
                    self.expert_memory.addWithEpR(item[:5], ep_r_his[i])
                else:
                    self.expert_memory.addWithEpR(item[:5], ep_r_his[ep_index])
                    if item[5]: ep_index += 1

        timesteps = (self.hyper_params.total_timesteps-self.hyper_params.start_timesteps)
        timesteps *= float(self.hyper_params.sample_decay_zoom)
        sample_p = self.hyper_params.sampleP
        self.sample_schedule = buildSchedule(self.hyper_params.samplePlanner, sample_p, 0, timesteps)

        if self.hyper_params.bcLossRatio != 0:
            timesteps = (self.hyper_params.total_timesteps-self.hyper_params.start_timesteps)/self.policy_freq
            timesteps *= float(self.hyper_params.bc_decay_zoom)
            self.bc_ratio_schedule = buildSchedule(self.hyper_params.bclossPlanner, self.hyper_params.bcLossRatio, self.hyper_params.bc_loss_FinalV, timesteps)

        if self.hyper_params.rlLossRatio != 0:
            timesteps = (self.hyper_params.total_timesteps-self.hyper_params.start_timesteps)/self.policy_freq
            timesteps *= float(self.hyper_params.bc_decay_zoom)
            self.rl_ratio_schedule = buildSchedule(self.hyper_params.rllossPlanner, self.hyper_params.rlLossRatio, 1, timesteps)
        
        # weight decay
        if self.learner_cfg.weight_decay != 0:
            self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), lr=self.learner_cfg.lr_actor, weight_decay=self.learner_cfg.weight_decay)
            self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(), lr=self.learner_cfg.lr_critic, weight_decay=self.learner_cfg.weight_decay)

    def pretrain(self):
        """Pretraining steps."""
        pretrain_loss = list()
        pretrain_step = self.hyper_params.pretrainStep
        if pretrain_step > 0: self.log.record("[INFO] Pre-Train %d step." % pretrain_step)
        for i_step in range(pretrain_step):
            experience = self.sample_experience(0)
            info = self.update_model(experience, pretrian=True)
            loss = info[0:3]
            pretrain_loss.append(loss)  # for logging
            # logging
            if i_step % 100 == 0:
                avg_loss = np.vstack(pretrain_loss).mean(axis=0)
                pretrain_loss.clear()
                if i_step % 500 == 0: self.write_log((avg_loss))
                    
        self.pretrainDone = True
        if self.hyper_params.trainWithoutBC: self.hyper_params.bcLossRatio = 0
        if pretrain_step > 0: self.log.record("[INFO] Pre-Train Complete!")

    def train(self):
        self.total_it += 1
        if len(self.memory) >= self.hyper_params.batch_size:
            experience = self.sample_experience(self.total_it)
            info = self.update_model(experience)
            loss = info[0:3]
            indices, new_priorities = info[3:5]
            self.loss_episode.append(loss)  # for logging
            if (new_priorities is not None) and (not self.hyper_params.trainWithoutPER):
                self.memory.update_priorities(indices[:info[5]], new_priorities[:info[5]])
                self.expert_memory.update_priorities(indices[info[5]:], new_priorities[info[5]:])
                # increase priority beta
                fraction = min(
                    float(self.total_it) / (self.hyper_params.total_timesteps - self.hyper_params.start_timesteps), 1.0)
                self.per_beta = self.per_beta + fraction * (1.0 - self.per_beta)

            if self.hyper_params.useDiscriminator:
                if self.total_it % self.hyper_params.train_discriminator_freq == 0:
                    for _ in range(self.hyper_params.d_gradient_steps):
                        onlineData, expertData = self.sample_experience(self.total_it, ratio=0.5, forDis=True)
                        self.discr.update(onlineData, expertData)

    def on_done(self, state, action, next_state, reward, done_bool, done, info):
        # logging
        if self.loss_episode:
            avg_loss = np.vstack(self.loss_episode).mean(axis=0)
            log_value = (avg_loss)
            self.write_log(log_value)
            self.loss_episode = []
            return log_value

    def addSampleToOnlineBuffer(self, tran):
        curr_state, action, reward, next_state, done_bool, done, info = tran
        re_done = False if self.hyper_params.reLabelingDone else done_bool
        if self.hyper_params.reLabeling:
            temp_r = reward if re_done else self.episode_r / len(self.episode_buffer)
        else:
            temp_r = reward
        if self.hyper_params.reLabelingSqil: temp_r = 0
        tran2 = (curr_state, action, temp_r, next_state, done_bool)
        self.memory.addWithEpR(tran2, np.array(self.episode_r))

    def update_buffer(self, curr_state, action, next_state, reward, done_bool, done, info):
        if info.get('success') is not None and self.env.absorption:  # absorption state env
            done_bool = info['success']
        if self.hyper_params.overTimeDone: done_bool = float(done)
        transition = (curr_state, action, reward, next_state, done_bool, done, info)
        self.episode_buffer.append(transition)
        self.episode_r += reward
        if done:
            # Calculated success rate
            self.success_np[self.episode_cnt % 100] = 1 if info['success'] else 0
            self.episode_cnt += 1
            success_rate = self.success_np.sum() / 100 if self.episode_cnt > 100 else self.success_np.sum() / self.episode_cnt
            # Record reward
            self.ep_r_np[self.episode_cnt % 100] = self.episode_r
            if self.pretrainDone: 
                self.log.record('success rate:{}, average reward:{}'.format(success_rate, np.mean(self.ep_r_np)))
            if self.hyper_params.expertDynamic and not self.hyper_params.trainWithoutBC:
                self.expert_lowest_rewards = self.expert_memory.get_lowest_rewards()
                if not self.hyper_params.trainWithoutPER:
                    buffer_max_len = self.expert_memory.buffer.max_len
                else:
                    buffer_max_len = self.expert_memory.max_len
                if len(self.expert_memory) >= (buffer_max_len):
                    threshold = self.expert_lowest_rewards
                else:
                    threshold = self.hyper_params.rewardThreshold
                if self.pretrainDone:
                    self.log.record('lowest reward in expert is {}'.format(self.expert_lowest_rewards))
                    self.log.record(f"online_memory: {len(self.memory)}, expert_memory: {len(self.expert_memory)}")
                for i, tran in enumerate(self.episode_buffer):
                    curr_state, action, reward, next_state, done_bool, done, info = tran
                    re_done = False if self.hyper_params.reLabelingDone else done_bool
                    if self.hyper_params.reLabeling:
                        temp_r = reward if re_done else self.episode_r / len(self.episode_buffer)
                    else:
                        temp_r = reward
                        if self.hyper_params.reLabelingSuccess:
                            success = self.episode_buffer[-1][-3]
                            if success and i > len(self.episode_buffer)-self.env.opt_n:
                                temp_r = (self.episode_r / self.env.opt_n) * (1 - success_rate)

                    if self.episode_r > threshold and self.pretrainDone and self.sample_schedule.current_value > 0:
                        if self.hyper_params.reLabelingSqil: temp_r = 1
                        if self.hyper_params.useDiscriminator:
                            temp_r = self.discr.predict_reward(
                                torch.FloatTensor(np.array([curr_state])).to(device),
                                torch.FloatTensor(np.array([action])).to(device), self.hyper_params.gamma, 1)
                            temp_r = temp_r[0][0].cpu().numpy().tolist()
                        tran2 = (curr_state, action, temp_r, next_state, done_bool)
                        self.expert_memory.addWithEpR(tran2, np.array(self.episode_r))
                    else:
                        if self.hyper_params.reLabelingSqil: temp_r = 0
                        if self.hyper_params.useDiscriminator:
                            temp_r = self.discr.predict_reward(
                                torch.FloatTensor(np.array([curr_state])).to(device),
                                torch.FloatTensor(np.array([action])).to(device), self.hyper_params.gamma, 1)
                            temp_r = temp_r[0][0].cpu().numpy().tolist()
                        tran2 = (curr_state, action, temp_r, next_state, done_bool)
                        self.memory.addWithEpR(tran2, np.array(self.episode_r))
            else:
                # online buffer
                for tran in self.episode_buffer:
                    self.addSampleToOnlineBuffer(tran)

            self.episode_buffer = []
            self.episode_r = 0

    def sample_experience(self, step, ratio=None, forDis=False):
        if ratio is None:
            demo_ratio = self.sample_schedule.value(step)
        else:
            demo_ratio = ratio
        if len(self.expert_memory) < self.hyper_params.batch_size*self.hyper_params.sampleP: demo_ratio = 0
        expert_batch_size = round(self.hyper_params.batch_size * demo_ratio)
        self_batch_size = self.hyper_params.batch_size - expert_batch_size

        if not self.hyper_params.trainWithoutPER:
            experiences_1 = self.memory.sample_dynamic(self_batch_size, self.per_beta)
        else:
            experiences_1 = self.memory.sample_dynamic(self_batch_size)
        e_batch = numpy2floattensorList(experiences_1[:6], device)
        experiences_1 = (tuple(e_batch) + experiences_1[6:])

        if expert_batch_size > 0:
            if not self.hyper_params.trainWithoutPER:
                expert_trans = self.expert_memory.sample_dynamic(expert_batch_size, self.per_beta)
            else:
                expert_trans = self.expert_memory.sample_dynamic(expert_batch_size)
            expert_trans = (numpy2floattensor(expert_trans[:6], device) + expert_trans[6:])

            if forDis:
                # 针对判别器的采样
                return experiences_1, expert_trans

            # online_buffer and expert_buffe
            temp = []
            for i, (o_sample, d_sample) in enumerate(zip(experiences_1, expert_trans)):
                if i < 6:
                    temp.append(torch.cat((o_sample, d_sample), dim=0))
                elif i == 6:
                    temp.append(o_sample+d_sample)
                else:
                    temp.append(np.append(o_sample, d_sample))
            experiences_1 = tuple(temp)

        experiences_1 += tuple([self_batch_size])  # Adds the length of the online buffer

        if self.use_n_step:
            indices = experiences_1[-3]
            experiences_n = self.memory_n.sample(indices)
            return (
                experiences_1,
                numpy2floattensor(experiences_n, device)
            )
        return experiences_1

    def _get_critic_loss(self, experiences, gamma):
        state, action, next_state, reward, not_done = experiences[:5]

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action = (
                self.actor_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * gamma * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        return (current_Q1 - target_Q).pow(2), (current_Q2 - target_Q).pow(2)

    def update_model(self, experience, pretrian=False):

        if self.hyper_params.trainBC:
            return self.update_model_bc(experience, pretrian)

        """Train the model after each episode."""
        use_n_step = self.hyper_params.nStep > 1
        if use_n_step:
            experience_1, experience_n = experience
        else:
            experience_1 = experience

        states, actions = experience_1[:2]
        if not self.hyper_params.trainWithoutPER:
            weights, indices, eps_d = experience_1[5:8]
        else:
            indices = None
        gamma = self.hyper_params.gamma

        # train critic
        critic1_loss, critic2_loss = self._get_critic_loss(experience_1, gamma)
        critic_loss_element_wise = critic1_loss + critic2_loss

        if use_n_step:
            gamma = gamma ** self.hyper_params.nStep

            _l1, _l2 = self._get_critic_loss(experience_n, gamma)
            critic_loss_n_element_wise = _l1 + _l2
            # to update loss and priorities
            critic_loss_element_wise += (
                critic_loss_n_element_wise * self.hyper_params.nStepLossRation
            )
        if not self.hyper_params.trainWithoutPER:
            critic_loss = torch.mean(critic_loss_element_wise * weights)
        else:
            critic_loss = torch.mean(critic_loss_element_wise)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # Compute actor losse
            actor_loss_element_wise = -self.critic.Q1(states, self.actor(states))
            if not self.hyper_params.trainWithoutPER:
                actor_loss = torch.mean(actor_loss_element_wise * weights)
            else:
                actor_loss = torch.mean(actor_loss_element_wise)

            if self.hyper_params.bcLossRatio != 0:
                # RL + BC
                if len(states[experience_1[-1]:]) > 0:
                    bc_loss = torch.mean(
                        torch.sum((self.actor(states[experience_1[-1]:]) - actions[experience_1[-1]:]) ** 2, dim=1))
                    b_rate = self.bc_ratio_schedule.value(self.total_it/self.policy_freq)
                    a_rate = self.rl_ratio_schedule.value(self.total_it/self.policy_freq)
                    actor_loss = actor_loss * a_rate + bc_loss * b_rate

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        else:
            actor_loss = torch.zeros(1)

        if pretrian:
            return (
                actor_loss.item(),
                torch.mean(critic1_loss).item(),
                torch.mean(critic2_loss).item()
            )
        else:
            new_priorities = None
            if self.total_it % self.policy_freq == 0 and not self.hyper_params.trainWithoutPER:
                # update priorities
                new_priorities = critic_loss_element_wise
                new_priorities += self.hyper_params.lambda3 * actor_loss_element_wise.pow(2)
                new_priorities += self.hyper_params.per_eps
                new_priorities = new_priorities.data.cpu().numpy().squeeze()
                new_priorities += eps_d

            return (
                actor_loss.item(),
                torch.mean(critic1_loss).item(),
                torch.mean(critic2_loss).item(),
                indices,
                new_priorities,
                experience_1[-1],
            )

    def update_model_bc(self, experience, pretrian=False):
        """Train the model after each episode."""
        use_n_step = self.hyper_params.nStep > 1
        if use_n_step:
            experience_1, experience_n = experience
        else:
            experience_1 = experience

        states, actions = experience_1[:2]
        if not self.hyper_params.trainWithoutPER:
            weights, indices, eps_d = experience_1[5:8]
        else:
            indices = None
        gamma = self.hyper_params.gamma

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            if self.hyper_params.bcLossRatio != 0:
                # BC loss
                if len(states[experience_1[-1]:]) > 0:
                    bc_loss = torch.mean(
                        torch.sum((self.actor(states[experience_1[-1]:]) - actions[experience_1[-1]:]) ** 2, dim=1))
                    b_rate = self.bc_ratio_schedule.value(self.total_it/self.policy_freq)
                    actor_loss = bc_loss * b_rate

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

        else:
            actor_loss = torch.zeros(1)

        if pretrian:
            return (
                actor_loss.item(),
                0,
                0
            )
        else:
            return (
                actor_loss.item(),
                0,
                0,
                indices,
                None,
                experience_1[-1],
            )
