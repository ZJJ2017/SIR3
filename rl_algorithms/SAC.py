import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.buffer import ReplayBufferOri

from utils.expUtils import SimHash

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal

	
class Network(nn.Module):
	def __init__(self, layer_num, input_dim, output_dim, hidden_dim, activation_function = torch.relu,last_activation = None):
		super(Network, self).__init__()
		self.activation = activation_function
		self.last_activation = last_activation
		layers_unit = [input_dim]+ [hidden_dim]*(layer_num-1) 
		layers = ([nn.Linear(layers_unit[idx],layers_unit[idx+1]) for idx in range(len(layers_unit)-1)])
		self.layers = nn.ModuleList(layers)
		self.last_layer = nn.Linear(layers_unit[-1],output_dim)
		self.network_init()
	def forward(self, x):
		return self._forward(x)
	def _forward(self, x):
		for layer in self.layers:
			x = self.activation(layer(x))
		x = self.last_layer(x)
		if self.last_activation != None:
			x = self.last_activation(x)
		return x
	def network_init(self):
		for layer in self.modules():
			if isinstance(layer, nn.Linear):
				nn.init.orthogonal_(layer.weight)
				layer.bias.data.zero_() 


class Actor(Network):
	def __init__(self, layer_num, input_dim, output_dim, hidden_dim, activation_function = torch.tanh,last_activation = None, trainable_std = False):
		super(Actor, self).__init__(layer_num, input_dim, output_dim, hidden_dim, activation_function, last_activation)
		self.trainable_std = trainable_std
		if self.trainable_std == True:
			self.logstd = nn.Parameter(torch.zeros(1, output_dim))
	def forward(self, x):
		mu = self._forward(x)
		if self.trainable_std == True:
			std = torch.exp(self.logstd)
		else:
			logstd = torch.zeros_like(mu)
			std = torch.exp(logstd)
		return mu, std


class Critic(Network):
	def __init__(self, layer_num, input_dim, output_dim, hidden_dim, activation_function, last_activation = None):
		super(Critic, self).__init__(layer_num, input_dim, output_dim, hidden_dim, activation_function, last_activation)
		
	def forward(self, *x):
		x = torch.cat(x, -1)
		return self._forward(x)
	

class SAC(object):
	def __init__(
		self,
		env,
		state_dim,
		action_dim,
		max_action,
		gamma=0.98,
		tau=0.02,
		alpha_init=0.2,
		batch_size=256,
		lr_actor=3e-4,
		lr_critic=3e-4,
		lr_alpha=3e-4,
		**kwargs
	):
		
		self.env = env

		self.actor = Actor(3, state_dim, action_dim, 256, torch.relu, None, True).to(device)

		self.q_1 = Critic(3, state_dim+action_dim, 1, 256, torch.relu, None).to(device)
		self.q_2 = Critic(3, state_dim+action_dim, 1, 256, torch.relu, None).to(device)
		
		self.target_q_1 = Critic(3, state_dim+action_dim, 1, 256, torch.relu, None).to(device)
		self.target_q_2 = Critic(3, state_dim+action_dim, 1, 256, torch.relu, None).to(device)
		
		self.soft_update(self.q_1, self.target_q_1, 1.)
		self.soft_update(self.q_2, self.target_q_2, 1.)
		
		self.alpha = nn.Parameter(torch.tensor(alpha_init))
		self.target_entropy = - torch.tensor(action_dim)

		self.q_1_optimizer = optim.Adam(self.q_1.parameters(), lr=lr_critic)
		self.q_2_optimizer = optim.Adam(self.q_2.parameters(), lr=lr_critic)
		
		self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
		self.alpha_optimizer = optim.Adam([self.alpha], lr=lr_alpha)		

		self.max_action = max_action
		self.gamma = gamma
		self.tau = tau
		
		self.total_it = 0
		self.batch_size = batch_size
		self.memory = ReplayBufferOri(state_dim, action_dim)
		
	def soft_update(self, network, target_network, rate):
		for network_params, target_network_params in zip(network.parameters(), target_network.parameters()):
			target_network_params.data.copy_(target_network_params.data * (1.0 - rate) + network_params.data * rate)
	
	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		_a, _ = self.get_action(state)
		return _a.cpu().data.numpy().flatten()

	def get_action(self, state):
		mu, std = self.actor(state)
		dist = Normal(mu, std)
		u = dist.rsample()
		u_log_prob = dist.log_prob(u)
		a = torch.tanh(u)
		a_log_prob = u_log_prob - torch.log(1 - torch.square(a) + 1e-6)
		return self.max_action * a, a_log_prob.sum(-1, keepdim=True)
	
	def q_update(self, Q, q_optimizer, states, actions, rewards, next_states, not_dones):
		# target
		with torch.no_grad():
			next_actions, next_action_log_prob = self.get_action(next_states)
			q_1 = self.target_q_1(next_states, next_actions)
			q_2 = self.target_q_2(next_states, next_actions)
			q = torch.min(q_1, q_2)
			v = not_dones * (q - self.alpha * next_action_log_prob)
			targets = rewards + self.gamma * v
		
		q = Q(states, actions)
		loss = F.smooth_l1_loss(q, targets)
		q_optimizer.zero_grad()
		loss.backward()
		q_optimizer.step()
		return loss
	
	def actor_update(self, states):
		now_actions, now_action_log_prob = self.get_action(states)
		q_1 = self.q_1(states, now_actions)
		q_2 = self.q_2(states, now_actions)
		q = torch.min(q_1, q_2)
		
		loss = (self.alpha.detach() * now_action_log_prob - q).mean()
		self.actor_optimizer.zero_grad()
		loss.backward()
		self.actor_optimizer.step()
		return loss, now_action_log_prob
	
	def alpha_update(self, now_action_log_prob):
		loss = (- self.alpha * (now_action_log_prob + self.target_entropy).detach()).mean()
		self.alpha_optimizer.zero_grad()    
		loss.backward()
		self.alpha_optimizer.step()
		return loss

	def update_buffer(self, state, action, next_state, reward, done_bool, done, info):
		self.memory.add(state, action, next_state, reward, done_bool)

	def train(self):
		self.total_it += 1
		
		# Sample replay buffer 
		states, actions, next_states, rewards, not_dones = self.memory.sample(self.batch_size)

		# q update
		q_1_loss = self.q_update(self.q_1, self.q_1_optimizer, states, actions, rewards, next_states, not_dones)
		q_2_loss = self.q_update(self.q_2, self.q_2_optimizer, states, actions, rewards, next_states, not_dones)

		# actor update
		actor_loss, prob = self.actor_update(states)
		
		# alpha update
		alpha_loss = self.alpha_update(prob)
		
		self.soft_update(self.q_1, self.target_q_1, self.tau)
		self.soft_update(self.q_2, self.target_q_2, self.tau)

	def save(self, filename, info=None):
		params = {
			"q_1": self.q_1.state_dict(),
			"q_2": self.q_2.state_dict(),
			"actor": self.actor.state_dict(),
		}
		path = os.path.join(filename, f"params_st_{str(self.total_it)}.pt") if info is None else os.path.join(filename, f"{str(info)}.pt")
		torch.save(params, path)
		print(f"[INFO] Saved the model and optimizer to {path} \n")

	def load(self, filename):
		params = torch.load(filename)
		self.q_1.load_state_dict(params["q_1"])
		self.q_2.load_state_dict(params["q_2"])
		self.target_q_1 = copy.deepcopy(self.q_1)
		self.target_q_2 = copy.deepcopy(self.q_2)

		self.actor.load_state_dict(params["actor"])
		print("[INFO] loaded the model from", filename)
