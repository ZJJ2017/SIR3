import gym
from gym.wrappers import TimeLimit
import numpy as np
import random


def vector_vector(arr, brr):
    return np.sum(arr*brr) / (np.sqrt(np.sum(arr*arr)) * np.sqrt(np.sum(brr*brr)))


'''
Encapsulate the environment twice
'''

class AbsorbStateWrapper(gym.Wrapper):
    """
    :param env: (gym.Env)
    :param is_sparse:
    :param time_reward:
    """

    def __init__(self, env, is_sparse=False, time_reward=None):
        self.absorption = True  # Absorptive environment
        self.is_sparse = is_sparse
        self.time_reward = time_reward  # Timeout penalty
        self.ep_r = 0
        self.ep_step = 0
        self.r_reach = 10
        if self.time_reward is not None:
            env = TimeFeatureWrapper(env)
        super(AbsorbStateWrapper, self).__init__(env)
        self._init()

    def _init(self):
        pass

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.ep_step += 1
        if self.is_sparse:
            reward, terminated, truncated = self.sparse_reward(obs, reward, done, info)
            done = terminated or truncated
            info['success'] = terminated
        if done:
            self.ep_step = 0
        return obs, reward, done, info

    def sparse_reward(self, obs, reward, done, info):
        self.ep_r += reward
        sparse_reward = 0
        terminated, truncated = False, done
        # Reach the target point
        if self.judge(obs, info):
            terminated = True
            sparse_reward = self.ep_r + self.calSparseReward()
            self.ep_r = 0
        else:
            if truncated:
                sparse_reward = self.ep_r + self.overTime()
                self.ep_r = 0
        return sparse_reward, terminated, truncated

    def judge(self, obs, info):
        raise NotImplemented

    def overTime(self):
        if self.time_reward is not None:
            sparse_reward = self.time_reward
        else:
            sparse_reward = 0
        return sparse_reward

    def calSparseReward(self):
        # sparse_reward = self.alpha * self.ep_ctrl + self.beta * self.r_reach + self.theta * r_costStep
        sparse_reward = self.r_reach
        return sparse_reward


class ReacherWrapper(AbsorbStateWrapper):
    def _init(self):
        self.r_reach = 15
        self.opt_n = 27

    def judge(self, obs, info):
        if abs(obs[9]) < 1e-2 and abs(obs[8]) < 1e-2:
            return True
        else:
            return False


class PusherWrapper(AbsorbStateWrapper):

    def _init(self):
        self.r_reach = 15
        self.opt_n = 77

    def judge(self, obs, info):
        if abs(obs[17] - obs[20]) < 3e-2 and abs(obs[18] - obs[21]) < 3e-2 and abs(obs[19] - obs[22]) < 0.1:
            return True
        else:
            return False


class XPositionWrapper(AbsorbStateWrapper):
    """
    Wrapper the X coordinates of the absorption state in the mujoco environment
    """
    def __init__(self, env, is_sparse, time_reward, env_name):
        super(__class__, self).__init__(env, is_sparse, time_reward)

        self.env_name = env_name

    def _init(self):
        self.r_reach = 0
        self.absorption = False

    def judge(self, obs, info):
        return 0
        if 'HalfCheetah' in self.env_name:
            if obs[0] < -0.5: return 2
        else:
            return 0
        
    def sparse_reward(self, obs, reward, done, info):
        self.ep_r += reward
        sparse_reward = 0
        terminated, truncated = False, done
        # target
        judge = self.judge(obs, info)
        if judge != 0:
            if judge == 1:
                terminated = True
            elif judge == 2:
                truncated = True
            sparse_reward = self.ep_r
            self.ep_r = 0
        else:
            if truncated:
                sparse_reward = self.ep_r
                self.ep_r = 0
        return sparse_reward, terminated, truncated


class RobosuiteWrapper(AbsorbStateWrapper):

    def _init(self):
        self.r_reach = 0

    def judge(self, obs, info):
        if self.ep_r >= self.max_episode_steps:
            return 2
        else:
            return 0

    def sparse_reward(self, obs, reward, done, info):
        self.ep_r += reward
        sparse_reward = 0
        terminated, truncated = False, done
        # target
        judge = self.judge(obs, info)
        if judge != 0:
            if judge == 1:
                terminated = True
            elif judge == 2:
                truncated = True
            sparse_reward = self.ep_r
            self.ep_r = 0
        else:
            if truncated:
                sparse_reward = self.ep_r
                self.ep_r = 0
        return sparse_reward, terminated, truncated


class TimeFeatureWrapper(gym.Wrapper):
    """
    Add remaining time to observation space for fixed length episodes.
    See https://arxiv.org/abs/1712.00378 and https://github.com/aravindr93/mjrl/issues/13.

    :param env: (gym.Env)
    :param max_steps: (int) Max number of steps of an episode
        if it is not wrapped in a TimeLimit object.
    :param test_mode: (bool) In test mode, the time feature is constant,
        equal to zero. This allow to check that the agent did not overfit this feature,
        learning a deterministic pre-defined sequence of actions.
    """

    def __init__(self, env, max_steps=1000, test_mode=False):
        assert isinstance(env.observation_space, gym.spaces.Box)
        # Add a time feature to the observation
        low, high = env.observation_space.low, env.observation_space.high
        low, high = np.concatenate((low, [0])), np.concatenate((high, [1.]))
        env.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

        super(TimeFeatureWrapper, self).__init__(env)

        if isinstance(env, TimeLimit):
            self._max_steps = env._max_episode_steps
        else:
            self._max_steps = max_steps
        self._max_episode_steps = self._max_steps
        self._current_step = 0
        self._test_mode = test_mode

    def reset(self):
        self._current_step = 0
        return self._get_obs(self.env.reset())

    def step(self, action):
        self._current_step += 1
        obs, reward, done, info = self.env.step(action)
        return self._get_obs(obs), reward, done, info

    def _get_obs(self, obs):
        """
        Concatenate the time feature to the current observation.

        :param obs: (np.ndarray)
        :return: (np.ndarray)
        """
        # Remaining time is more general
        time_feature = 1 - (self._current_step / self._max_steps)
        if self._test_mode:
            time_feature = 1.0
        # Optionnaly: concatenate [time_feature, time_feature ** 2]
        return np.concatenate((obs, [time_feature]))


def gymnasium_wrapper(env, is_sparse=False, time_reward=None):

    import gymnasium as gym
    class FetchWrapper(gym.Wrapper):
        def __init__(self, env, is_sparse=False, time_reward=None):
            self.absorption = True  # Absorptive environment
            self.is_sparse = is_sparse
            self.time_reward = time_reward  # Timeout penalty
            self.ep_r = 0
            self.ep_step = 0
            self.r_reach = 10
            self.r_fail = 0
            self.opt_n = 16
            # if self.time_reward is not None:
            #     env = TimeFeatureWrapper(env)
            super(FetchWrapper, self).__init__(env)

        def seed(self, seed):
            obs, info = self.env.reset(seed=seed)

        def reset(self):
            obs, info = self.env.reset()
            return np.concatenate((obs['desired_goal'], obs['observation']))

        def step(self, action):
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.ep_step += 1
            if self.is_sparse:
                reward, terminated, truncated = self.sparse_reward(obs, reward, terminated, truncated, info)
                info['success'] = terminated
            done = terminated or truncated
            if done:
                self.ep_step = 0
            return np.concatenate((obs['desired_goal'], obs['observation'])), reward, done, info

        def sparse_reward(self, obs, reward, terminated, truncated, info):
            self.ep_r += reward
            sparse_reward = 0
            # Reach the target point
            # if self.judge(obs, info):
            #     terminated = True
            #     sparse_reward = self.ep_r + self.r_reach
            #     self.ep_r = 0
            # else:
            #     if truncated:
            #         sparse_reward = self.ep_r
            #         self.ep_r = 0
            if terminated:
                sparse_reward = self.ep_r + self.r_reach
            elif truncated:
                sparse_reward = self.ep_r
            return sparse_reward, terminated, truncated

        def judge(self, obs, info):
            pass

    return FetchWrapper(env, is_sparse, time_reward)


if __name__ == '__main__':
    env = gym.make('HalfCheetah-v2')
    obs = env.reset()
    action = [random.uniform(-1, 1) for _  in range(6)]
    obs, reward, done, info = env.step(action)
    print('make Success')
