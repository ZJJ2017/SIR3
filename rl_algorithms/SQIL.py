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
from utils.expUtils import plotSmoothAndSaveFig, addTimeFeature, rmTimeFeature

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SQIL(TD3):
    def __init__(self, td3_kwargs, cfg):
        super(__class__, self).__init__(**td3_kwargs)

        self.is_test = cfg.test
        self.hyper_params = cfg.hyper_params
        self.learner_cfg = cfg.learner_cfg
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
            print("demos %d" % len(demos))
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

            # replay memory for a single step
            self.memory = ReplayBuffer(
                self.hyper_params.buffer_size,
                self.hyper_params.batch_size // 2,
            )
            # expert memory
            self.expert_memory = ReplayBuffer(
                self.hyper_params.buffer_size,
                self.hyper_params.batch_size // 2,
            )
            temp_traj = []
            for item in demos:
                state, action, reward, next_state, done = item
                self.expert_memory.add((state, action, 1., next_state, done))
                temp_traj.append(item)
            traj_cnt = 0
            while len(self.expert_memory) < self.hyper_params.batch_size // 2:
                self.expert_memory.add(temp_traj[traj_cnt])
                if traj_cnt < len(temp_traj): traj_cnt += 1
                else: traj_cnt = 0

    def update_buffer(self, state, action, next_state, reward, done_bool, done, info):
        transition = (state, action, 0., next_state, done)
        self._add_transition_to_memory(transition)

    def train(self):
        self.total_it += 1

        if len(self.memory) >= self.hyper_params.batch_size:
            experience = self.sample_experience()
            info = self.update_model(experience)
            loss = info[0:3]
            self.loss_episode.append(loss)  # for logging

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

        # add a single step transition
        # if transition is not an empty tuple
        if transition:
            self.memory.add(transition)
    
    def sample_experience(self):
        experiences_1 = self.memory.sample()
        experiences_1 = (
            numpy2floattensor(experiences_1[:6], device)
            + experiences_1[6:]
        )
        experiences_2 = self.expert_memory.sample()
        experiences_2 = (
            numpy2floattensor(experiences_2[:6], device)
            + experiences_2[6:]
        )
        # 
        temp = []
        for i, (o_sample, d_sample) in enumerate(zip(experiences_1, experiences_2)):
            if i < 6:
                temp.append(torch.cat((o_sample, d_sample), dim=0))
            elif i == 6:
                temp.append(o_sample + d_sample)
            else:
                temp.append(np.append(o_sample, d_sample))
        experiences_1 = tuple(temp)

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

        critic_loss = torch.mean(critic_loss_element_wise)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # Compute actor losse
            actor_loss_element_wise = -self.critic.Q1(states, self.actor(states))
            actor_loss = torch.mean(actor_loss_element_wise)

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
            return (
                actor_loss.item(),
                torch.mean(critic1_loss).item(),
                torch.mean(critic2_loss).item(),
            )

    def write_log(self, log_value: tuple):
        """Write log about loss and score"""
        loss = log_value
        total_loss = loss.sum()

        print(
            "total loss: %f actor_loss: %.3f critic1_loss: %.3f critic2_loss: %.3f \n"
            % (
                total_loss,
                loss[0]*self.policy_freq,
                loss[1],
                loss[2],
            )  # actor loss  # critic loss
        )
