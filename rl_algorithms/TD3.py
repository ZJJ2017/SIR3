import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.buffer import ReplayBufferOri


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)
        
        self.max_action = max_action

        # self.loss_episode = list()

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 400)
        self.l5 = nn.Linear(400, 300)
        self.l6 = nn.Linear(300, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class TD3(object):
    def __init__(
        self,
        env,
        state_dim,
        action_dim,
        max_action,
        gamma=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        batch_size=256,
        lr_actor=3e-4,
        lr_critic=3e-4,
        **kwargs
    ):

        self.env = env

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0
        self.batch_size = batch_size
        self.memory = ReplayBufferOri(state_dim, action_dim)
        
        self.loss_episode = []

        # self.simhash = SimHash(self.env.state_dim, 32, 0.2)
        # self.icm = ICMModel(self.env.state_dim, self.env.action_dim, self.env.max_action, eta=1)

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def update_buffer(self, state, action, next_state, reward, done_bool, done, info):
        self.memory.add(state, action, next_state, reward, done_bool)

    def train(self):
        self.total_it += 1

        # Sample replay buffer 
        state, action, next_state, reward, not_done = self.memory.sample(self.batch_size)
        # reward += self.simhash.count(state)
        # reward += self.icm.compute_intrinsic_reward(state, next_state, action)


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
            target_Q = reward + not_done * self.gamma * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor losse
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            
            # Optimize the actor 
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # self.icm.update_model(state, next_state, action)

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        else:
            actor_loss = torch.zeros(1)

        self.loss_episode.append([actor_loss.item(), critic_loss.item(), critic_loss.item()])

    def on_done(self, state, action, next_state, reward, done_bool, done, info):
        # logging
        if self.loss_episode:
            avg_loss = np.vstack(self.loss_episode).mean(axis=0)
            log_value = (avg_loss)
            self.write_log(log_value)
            self.loss_episode = []
            return log_value

    def write_log(self, log_value: tuple):
        """Write log about loss and score"""
        loss = log_value
        total_loss = loss.sum()

        self.log.record(
            "total loss: %f actor_loss: %.3f critic_loss: %.3f"
            % (total_loss, loss[0]*self.policy_freq, loss[1])  # actor loss  # critic loss
        )

    def save(self, filename, info=None):
        params = {
            "critic": self.critic.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
        }
        path = os.path.join(filename, f"params_st_{str(self.total_it)}.pt") if info is None else os.path.join(filename, f"{str(info)}.pt")
        torch.save(params, path)
        print(f"[INFO] Saved the model and optimizer to {path} \n")

    def load(self, filename):
        params = torch.load(filename)
        self.critic.load_state_dict(params["critic"])
        self.critic_optimizer.load_state_dict(params["critic_optimizer"])
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(params["actor"])
        self.actor_optimizer.load_state_dict(params["actor_optimizer"])
        self.actor_target = copy.deepcopy(self.actor)
        print("[INFO] loaded the model from", filename)
        