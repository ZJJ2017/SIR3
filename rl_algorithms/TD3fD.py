import os
import copy
import time
import numpy as np

import pickle
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F

import gym

from rl_algorithms.TD3 import TD3
from utils.helper_functions import get_n_step_info_from_demo, numpy2floattensor

from utils.buffer import ReplayBuffer, PrioritizedBufferWrapper
from utils.expUtils import demo_re_labeling, addTimeFeature, rmTimeFeature


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


class TD3fD(TD3):
    def __init__(self, td3_kwargs, cfg):
        super(__class__, self).__init__(**td3_kwargs)

        self.is_test = cfg.test
        self.hyper_params = cfg.hyper_params
        self.learner_cfg = cfg.learner_cfg
        self.exp_path = cfg.exp_path
        self._initialize()

    def _initialize(self):
        """Initialize non-common things."""
        self.per_beta = self.hyper_params.per_beta
        self.use_n_step = self.hyper_params.nStep > 1

        self.loss_episode = list()

        if not self.is_test:
            if isinstance(self.hyper_params.demo_path, list):
                demo_path = self.hyper_params.demo_path[0]
            else:
                demo_path = self.hyper_params.demo_path
            # load demo replay memory
            with open(demo_path, "rb") as f:
                demos = pickle.load(f)			
            # demo处理和重标记
            self.env.prePro = True
            demos, ep_r_his = demo_re_labeling(self.env, demos)
            self.env.prePro = False
            print("demos %d" % len(demos))
            # 存储周期奖励
            self.ep_r_his = ep_r_his
            # 时间特征处理
            if self.env.time_reward is not None and len(demos[0][0]) < self.env.state_dim:
                demos = addTimeFeature(demos, self.env.max_episode_steps)
            elif self.env.time_reward is None and len(demos[0][0]) > self.env.state_dim:
                demos = rmTimeFeature(demos)
            # 删除demo中的其他信息，如info
            if len(demos[0]) > 5:
                demos_ls = []
                for _d in demos:
                    demos_ls.append((_d[0], _d[1], _d[2], _d[3], _d[4]))
                demos = demos_ls

            if self.use_n_step:
                demos, demos_n_step = get_n_step_info_from_demo(
                    demos, self.hyper_params.n_step, self.hyper_params.gamma
                )

                # replay memory for multi-steps
                self.memory_n = ReplayBuffer(
                    max_len=self.hyper_params.buffer_size,
                    batch_size=self.hyper_params.batch_size,
                    n_step=self.hyper_params.n_step,
                    gamma=self.hyper_params.gamma,
                    demo=demos_n_step,
                )

            # replay memory for a single step
            self.memory = ReplayBuffer(
                self.hyper_params.buffer_size,
                self.hyper_params.batch_size,
                demo=demos,
            )
            self.memory = PrioritizedBufferWrapper(
                self.memory,
                alpha=self.hyper_params.per_alpha,
                epsilon_d=self.hyper_params.per_eps_demo,
            )

    def update_buffer(self, state, action, next_state, reward, done_bool, done, info):
        transition = (state, action, reward, next_state, done_bool)
        self._add_transition_to_memory(transition)

    def pretrain(self):
        """Pretraining steps."""
        pretrain_loss = list()
        pretrain_step = self.hyper_params.pretrainStep
        self.log.record("[INFO] Pre-Train %d step." % pretrain_step)
        for i_step in range(1, pretrain_step + 1):
            experience = self.sample_experience()
            info = self.update_model(experience, pretrian=True)
            loss = info[0:3]
            pretrain_loss.append(loss)  # for logging

            # logging
            if i_step == 1 or i_step % 100 == 0:
                avg_loss = np.vstack(pretrain_loss).mean(axis=0)
                pretrain_loss.clear()
                log_value = (avg_loss)
                self.write_log(log_value)
        self.log.record("[INFO] Pre-Train Complete!\n")

    def train(self):
        self.total_it += 1

        if len(self.memory) >= self.hyper_params.batch_size:
            experience = self.sample_experience()
            info = self.update_model(experience)
            loss = info[0:3]
            indices, new_priorities = info[3:5]
            self.loss_episode.append(loss)  # for logging
            if new_priorities is not None:
                self.memory.update_priorities(indices, new_priorities)

        # increase priority beta
        fraction = min(float(self.total_it) / (self.hyper_params.total_timesteps-self.hyper_params.start_timesteps), 1.0)
        self.per_beta = self.per_beta + fraction * (1.0 - self.per_beta)

    def on_done(self, state, action, next_state, reward, done_bool, done, info):
        # logging
        if self.loss_episode:
            avg_loss = np.vstack(self.loss_episode).mean(axis=0)
            log_value = (avg_loss)
            self.write_log(log_value)
            self.loss_episode = []

    def _add_transition_to_memory(self, transition):
        """Add 1 step and n step transitions to memory."""
        # add n-step transition
        if self.use_n_step:
            transition = self.memory_n.add(transition)

        # add a single step transition
        # if transition is not an empty tuple
        if transition:
            self.memory.add(transition)
    
    def sample_experience(self):
        experiences_1 = self.memory.sample(self.per_beta)
        experiences_1 = (
            numpy2floattensor(experiences_1[:6], device)
            + experiences_1[6:]
        )
        if self.use_n_step:
            indices = experiences_1[-2]
            experiences_n = self.memory_n.sample(indices)
            return (
                experiences_1,
                numpy2floattensor(experiences_n, device),
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
        # critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        # return F.mse_loss(current_Q1, target_Q), F.mse_loss(current_Q2, target_Q)
        return (current_Q1 - target_Q).pow(2), (current_Q2 - target_Q).pow(2)

    def update_model(self, experience, pretrian=False):
        """Train the model after each episode."""
        use_n_step = self.hyper_params.nStep > 1
        if use_n_step:
            experience_1, experience_n = experience
        else:
            experience_1 = experience

        states, actions = experience_1[:2]
        weights, indices, eps_d = experience_1[5:8]
        gamma = self.hyper_params.gamma

        # train critic
        critic1_loss, critic2_loss = self._get_critic_loss(experience_1, gamma)
        critic_loss_element_wise = critic1_loss + critic2_loss

        if use_n_step:
            gamma = gamma ** self.hyper_params.n_step

            _l1, _l2 = self._get_critic_loss(experience_n, gamma)
            critic_loss_n_element_wise = _l1 + _l2
            # to update loss and priorities
            critic_loss_element_wise += (
                critic_loss_n_element_wise * self.hyper_params.nStepLossRation
            )

        critic_loss = torch.mean(critic_loss_element_wise * weights)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # Compute actor losse
            actor_loss_element_wise = -self.critic.Q1(states, self.actor(states))
            actor_loss = torch.mean(actor_loss_element_wise * weights)

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
            if self.total_it % self.policy_freq == 0:
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
            )

    def write_log(self, log_value: tuple):
        """Write log about loss and score"""
        loss = log_value
        total_loss = loss.sum()

        self.log.record(
            "total loss: %f actor_loss: %.3f critic1_loss: %.3f critic2_loss: %.3f \n"
            % (
                total_loss,
                loss[0]*self.policy_freq,
                loss[1],
                loss[2],
            )  # actor loss  # critic loss
        )

