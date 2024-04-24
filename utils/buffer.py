# -*- coding: utf-8 -*-
"""Replay buffer for baselines."""

from collections import deque
from typing import Any, Deque, List, Tuple

import numpy as np
import random
import math
import torch

from utils.helper_functions import get_n_step_info
from utils.segment_tree import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ReplayBufferOri(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )


class ReplayBuffer(object):
    """Fixed-size buffer to store experience tuples.

    Attributes:
        obs_buf (np.ndarray): observations
        acts_buf (np.ndarray): actions
        rews_buf (np.ndarray): rewards
        next_obs_buf (np.ndarray): next observations
        done_buf (np.ndarray): dones
        n_step_buffer (deque): recent n transitions
        n_step (int): step size for n-step transition
        gamma (float): discount factor
        max_len (int): size of buffers
        batch_size (int): batch size for training
        demo_size (int): size of demo transitions
        length (int): amount of memory filled
        idx (int): memory index to add the next incoming transition
    """

    def __init__(
        self,
        max_len: int,
        batch_size: int,
        gamma: float = 0.99,
        n_step: int = 1,
        demo: List[Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]] = None,
    ):
        """Initialize a ReplayBuffer object.

        Args:
            max_len (int): size of replay buffer for experience
            batch_size (int): size of a batched sampled from replay buffer for training
            gamma (float): discount factor
            n_step (int): step size for n-step transition
            demo (list): transitions of human play
        """
        assert 0 < batch_size <= max_len
        assert 0.0 <= gamma <= 1.0
        assert 1 <= n_step <= max_len

        self.obs_buf: np.ndarray = None
        self.acts_buf: np.ndarray = None
        self.rews_buf: np.ndarray = None
        self.next_obs_buf: np.ndarray = None
        self.done_buf: np.ndarray = None

        self.n_step_buffer: Deque = deque(maxlen=n_step)
        self.n_step = n_step
        self.gamma = gamma

        self.max_len = max_len
        self.batch_size = batch_size
        self.demo_size = len(demo) if demo else 0
        self.demo = demo
        self.length = 0
        self.idx = self.demo_size

        # demo may have empty tuple list [()]
        if self.demo and self.demo[0]:
            self.max_len += self.demo_size
            self.length += self.demo_size
            for idx, d in enumerate(self.demo):
                state, action, reward, next_state, done = d
                if idx == 0:
                    action = (
                        np.array(action).astype(np.int64)
                        if isinstance(action, int)
                        else action
                    )
                    self._initialize_buffers(state, action)
                self.obs_buf[idx] = state
                self.acts_buf[idx] = np.array(action)
                self.rews_buf[idx] = reward
                self.next_obs_buf[idx] = next_state
                self.done_buf[idx] = done

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(
        self, transition: Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]
    ) -> Tuple[Any, ...]:
        """Add a new experience to memory.
        If the buffer is empty, it is respectively initialized by size of arguments.
        """
        assert len(transition) == 5, "Inappropriate transition size"
        assert isinstance(transition[0], np.ndarray)
        assert isinstance(transition[1], np.ndarray)

        self.n_step_buffer.append(transition)

        # single step transition is not ready
        if len(self.n_step_buffer) < self.n_step:
            return ()

        if self.length == 0:
            state, action = transition[:2]
            self._initialize_buffers(state, action)

        # add a multi step transition
        reward, next_state, done = get_n_step_info(self.n_step_buffer, self.gamma)
        curr_state, action = self.n_step_buffer[0][:2]

        self.obs_buf[self.idx] = curr_state
        self.acts_buf[self.idx] = action
        self.rews_buf[self.idx] = reward
        self.next_obs_buf[self.idx] = next_state
        self.done_buf[self.idx] = done

        self.idx += 1
        self.idx = self.demo_size if self.idx % self.max_len == 0 else self.idx
        self.length = min(self.length + 1, self.max_len)

        # return a single step transition to insert to replay buffer
        return self.n_step_buffer[0]

    def extend(
        self, transitions: List[Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]]
    ):
        """Add experiences to memory."""
        for transition in transitions:
            self.add(transition)

    def sample(self, indices: List[int] = None) -> Tuple[np.ndarray, ...]:
        """Randomly sample a batch of experiences from memory."""
        assert len(self) >= self.batch_size

        if indices is None:
            indices = np.random.choice(len(self), size=self.batch_size, replace=False)

        states = self.obs_buf[indices]
        actions = self.acts_buf[indices]
        rewards = self.rews_buf[indices].reshape(-1, 1)
        next_states = self.next_obs_buf[indices]
        dones = self.done_buf[indices].reshape(-1, 1)
        not_dones = np.ones_like(dones) - dones

        # state, action, next_state, reward, not_done
        # return (
        #     torch.FloatTensor(states).to(self.device),
        #     torch.FloatTensor(actions).to(self.device),
        #     torch.FloatTensor(next_states).to(self.device),
        #     torch.FloatTensor(rewards).to(self.device),
        #     torch.FloatTensor(not_dones).to(self.device)
        # )
        return (states, actions, next_states, rewards, not_dones)

    def _initialize_buffers(self, state: np.ndarray, action: np.ndarray) -> None:
        """Initialze buffers for state, action, resward, next_state, done."""
        # In case action of demo is not np.ndarray
        self.obs_buf = np.zeros([self.max_len] + list(state.shape), dtype=state.dtype)
        self.acts_buf = np.zeros(
            [self.max_len] + list(action.shape), dtype=action.dtype
        )
        self.rews_buf = np.zeros([self.max_len], dtype=float)
        self.next_obs_buf = np.zeros(
            [self.max_len] + list(state.shape), dtype=state.dtype
        )
        self.done_buf = np.zeros([self.max_len], dtype=float)

    def __len__(self) -> int:
        """Return the current size of internal memory."""
        return self.length


class DynReplayBuffer(ReplayBuffer):
    def __init__(self, max_len: int, batch_size: int, gamma: float = 0.99, n_step: int = 1,
                 demo: List[Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]] = None):
        """Initialize a ReplayBuffer object.

        Args:
            max_len (int): size of replay buffer for experience
            batch_size (int): size of a batched sampled from replay buffer for training
            gamma (float): discount factor
            n_step (int): step size for n-step transition
            demo (list): transitions of human play
        """
        super().__init__(max_len, batch_size, gamma, n_step, demo)
        assert 0 < batch_size <= max_len
        assert 0.0 <= gamma <= 1.0
        assert 1 <= n_step <= max_len

        self.obs_buf: np.ndarray = None
        self.acts_buf: np.ndarray = None
        self.rews_buf: np.ndarray = None
        self.next_obs_buf: np.ndarray = None
        self.done_buf: np.ndarray = None
        self.ep_r_buf: np.ndarray = None
        # self.ep_len_buf: np.ndarray = None

        self.n_step_buffer: Deque = deque(maxlen=n_step)
        self.n_step = n_step
        self.gamma = gamma

        self.max_len = max_len
        self.batch_size = batch_size
        self.demo_size = len(demo) if demo else 0
        self.demo = demo
        self.length = 0
        self.idx = self.demo_size

    def _initialize_buffers(self, state: np.ndarray, action: np.ndarray) -> None:
        """Initialze buffers for state, action, resward, next_state, done."""
        # In case action of demo is not np.ndarray
        self.obs_buf = np.zeros([self.max_len] + list(state.shape), dtype=state.dtype)
        self.acts_buf = np.zeros(
            [self.max_len] + list(action.shape), dtype=action.dtype
        )
        self.rews_buf = np.zeros([self.max_len], dtype=float)
        self.next_obs_buf = np.zeros(
            [self.max_len] + list(state.shape), dtype=state.dtype
        )
        self.done_buf = np.zeros([self.max_len], dtype=float)
        self.ep_r_buf = np.ones_like(self.done_buf)*math.inf
        # self.ep_len_buf = np.ones_like(self.done_buf)

    def add(
        self, transition: Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool], 
        ep_r: np.ndarray = None,
        ep_len: np.ndarray = None,
    ) -> Tuple[Any, ...]:
        """Add a new experience to memory.
        If the buffer is empty, it is respectively initialized by size of arguments.
        """
        assert len(transition) == 5, "Inappropriate transition size"
        assert isinstance(transition[0], np.ndarray)
        assert isinstance(transition[1], np.ndarray)

        self.n_step_buffer.append(transition)

        # single step transition is not ready
        if len(self.n_step_buffer) < self.n_step:
            return ()

        if self.length == 0:
            state, action = transition[:2]
            self._initialize_buffers(state, action)

        # add a multi step transition
        reward, next_state, done = get_n_step_info(self.n_step_buffer, self.gamma)
        curr_state, action = self.n_step_buffer[0][:2]

        if self.length == self.max_len:
            if ep_r is not None:
                if ep_r > np.min(self.ep_r_buf):
                    self.idx = self.ep_r_buf.argmin()
                else:
                    self.idx = None
            else:
                # self.idx = self.rews_buf.argmin()
                self.idx += 1
                self.idx = self.demo_size if self.idx % self.max_len == 0 else self.idx

        if self.idx != None:
            self.obs_buf[self.idx] = curr_state
            self.acts_buf[self.idx] = action
            self.rews_buf[self.idx] = reward
            self.next_obs_buf[self.idx] = next_state
            self.done_buf[self.idx] = done
            if ep_r is not None:
                self.ep_r_buf[self.idx] = ep_r
            # if ep_len is not None:
            #     self.ep_len_buf[self.idx] = ep_len

        if self.length < self.max_len:
            self.idx += 1
            self.length = min(self.length + 1, self.max_len)

        # return a single step transition to insert to replay buffer
        return self.n_step_buffer[0]

    def sample(self, indices: List[int] = None, threshold=None) -> Tuple[np.ndarray, ...]:
        """Randomly sample a batch of experiences from memory."""
        assert len(self) >= self.batch_size

        if indices is None:
            indices = np.random.choice(len(self), size=self.batch_size, replace=False)

        states = self.obs_buf[indices]
        actions = self.acts_buf[indices]
        rewards = self.rews_buf[indices].reshape(-1, 1)
        next_states = self.next_obs_buf[indices]
        dones = self.done_buf[indices].reshape(-1, 1)
        not_dones = np.ones_like(dones) - dones

        # if reLabelingDyn:
        #     ep_r = self.ep_r_buf[indices].reshape(-1, 1)
        #     ep_len = self.ep_len_buf[indices].reshape(-1, 1)
        #     if ep_r is not None and ep_len is not None:
        #         rewards = 
        if threshold is not None:
            ep_r = self.ep_r_buf[indices].reshape(-1, 1)
            rewards[ep_r >= threshold] *= 2
            rewards[ep_r < threshold] *= 0.5

        return (states, actions, next_states, rewards, not_dones)

    def sample_dynamic(self, batch_size,  threshold=None) -> Tuple[torch.Tensor, ...]:
        """Sample a batch of experiences."""
        self.batch_size = batch_size
        assert len(self) >= batch_size

        experiences = self.sample(threshold=threshold)

        return experiences

    def addWithEpR(
        self, transition: Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool],
        ep_r: np.ndarray = None,
        ep_len: np.ndarray = None,
    ) -> Tuple[Any, ...]:
        """Add experience and priority."""
        n_step_transition = self.add(transition, ep_r, ep_len)

        return n_step_transition

    def get_lowest_rewards(self):
        if len(self) > 0:
            return np.min(self.ep_r_buf)
        else:
            return -100000

class PrioritizedBufferWrapper(object):
    """Prioritized Experience Replay wrapper for Buffer.

    Refer to OpenAI baselines github repository:
    https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py

    Attributes:
        buffer (Buffer): Hold replay buffer as an attribute
        alpha (float): alpha parameter for prioritized replay buffer
        epsilon_d (float): small positive constants to add to the priorities
        tree_idx (int): next index of tree
        sum_tree (SumSegmentTree): sum tree for prior
        min_tree (MinSegmentTree): min tree for min prior to get max weight
        _max_priority (float): max priority
    """

    def __init__(
        self, base_buffer, alpha: float = 0.6, epsilon_d: float = 1.0
    ):
        """Initialize.

        Args:
            base_buffer (Buffer): ReplayBuffer which should be hold
            alpha (float): alpha parameter for prioritized replay buffer
            epsilon_d (float): small positive constants to add to the priorities

        """
        self.buffer = base_buffer
        assert alpha >= 0
        self.alpha = alpha
        self.epsilon_d = epsilon_d
        self.tree_idx = 0

        # capacity must be positive and a power of 2.
        tree_capacity = 1
        while tree_capacity < self.buffer.max_len:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)
        self._max_priority = 1.0

        # for init priority of demo
        self.tree_idx = self.buffer.demo_size
        for i in range(self.buffer.demo_size):
            self.sum_tree[i] = self._max_priority ** self.alpha
            self.min_tree[i] = self._max_priority ** self.alpha

    def add(
        self, transition: Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]
    ) -> Tuple[Any, ...]:
        """Add experience and priority."""
        n_step_transition = self.buffer.add(transition)
        if n_step_transition:
            self.sum_tree[self.tree_idx] = self._max_priority ** self.alpha
            self.min_tree[self.tree_idx] = self._max_priority ** self.alpha

            self.tree_idx += 1
            if self.tree_idx % self.buffer.max_len == 0:
                self.tree_idx = self.buffer.demo_size

        return n_step_transition

    def addWithEpR(
        self, transition: Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool], 
        ep_r: np.ndarray = None,
        ep_len: np.ndarray = None,
    ) -> Tuple[Any, ...]:
        """Add experience and priority."""
        n_step_transition = self.buffer.add(transition, ep_r, ep_len)
        if n_step_transition:
            self.sum_tree[self.tree_idx] = self._max_priority ** self.alpha
            self.min_tree[self.tree_idx] = self._max_priority ** self.alpha

            self.tree_idx += 1
            if self.tree_idx % self.buffer.max_len == 0:
                self.tree_idx = self.buffer.demo_size

        return n_step_transition

    def get_lowest_rewards(self):
        if len(self.buffer) > 0:
            return np.min(self.buffer.ep_r_buf)
        else:
            return -100000

    def _sample_proportional(self, batch_size: int) -> list:
        """Sample indices based on proportional."""
        indices = []
        p_total = self.sum_tree.sum(0, len(self.buffer))
        segment = p_total / batch_size

        i = 0
        while len(indices) < batch_size:
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            if idx > len(self.buffer):
                print(
                    f"[WARNING] Index for sampling is out of range: {len(self.buffer)} < {idx}"
                )
                continue
            indices.append(idx)
            i += 1
        return indices

    def sample(self, beta: float = 0.4) -> Tuple[torch.Tensor, ...]:
        """Sample a batch of experiences."""
        assert len(self.buffer) >= self.buffer.batch_size
        assert beta > 0

        indices = self._sample_proportional(self.buffer.batch_size)

        # get max weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self.buffer)) ** (-beta)

        # calculate weights
        weights_, eps_d = [], []
        for i in indices:
            eps_d.append(self.epsilon_d if i < self.buffer.demo_size else 0.0)
            p_sample = self.sum_tree[i] / self.sum_tree.sum()
            weight = (p_sample * len(self.buffer)) ** (-beta)
            weights_.append(weight / max_weight)

        weights = np.array(weights_)
        eps_d = np.array(eps_d)
        experiences = self.buffer.sample(indices)

        return experiences + (weights, indices, eps_d)

    def sample_dynamic(self, batch_size, beta: float = 0.4, threshold=None) -> Tuple[torch.Tensor, ...]:
        """Sample a batch of experiences."""
        self.buffer.batch_size = batch_size
        assert len(self.buffer) >= batch_size
        assert beta > 0

        indices = self._sample_proportional(batch_size)

        # get max weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self.buffer)) ** (-beta)

        # calculate weights
        weights_, eps_d = [], []
        for i in indices:
            eps_d.append(self.epsilon_d if i < self.buffer.demo_size else 0.0)
            p_sample = self.sum_tree[i] / self.sum_tree.sum()
            weight = (p_sample * len(self.buffer)) ** (-beta)
            weights_.append(weight / max_weight)

        weights = np.array(weights_)
        eps_d = np.array(eps_d)
        experiences = self.buffer.sample(indices, threshold)

        return experiences + (weights, indices, eps_d)

    def update_priorities(self, indices: list, priorities: np.ndarray):
        """Update priorities of sampled transitions."""
        assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < len(self.buffer)

            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha

            self._max_priority = max(self._max_priority, priority)

    def __len__(self) -> int:
        """Return the current size of internal memory."""
        return len(self.buffer)