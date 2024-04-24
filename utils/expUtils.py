import gym
import numpy as np
import torch
import math
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import animation 

from utils.envWrapper import ReacherWrapper, PusherWrapper
import random

'''
Some auxiliary functions to be used in the experiment, such as reward adjustment, reading trajectory and so on
'''

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        False


class Log:
    def __init__(self, filePath):
        self.filePath = filePath

    def close(self):
        pass

    def record(self, s):
        print(s)
        f = open(self.filePath, "a+")
        f.write(s+'\n')
        f.close()


# Smooth drawing image
def plotSmoothAndSaveFig(interval, data, filePath):
    temp_list = []
    if len(data) > interval:
        for index in range(interval, len(data)):
            temp_list.append(np.average(np.array(data[index - interval:index])))
    else:
        for index in range(min(interval, len(data))):
            temp_list.append(data[index])
    # plt.figure()
    plt.plot(temp_list)
    plt.savefig(filePath)
    plt.close()


def display_frames_as_gif(frames, id, path):
    try:
        patch = plt.imshow(frames[0])
        plt.axis('off')
        def animate(i):
            patch.set_data(frames[i])

        anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=5)
        anim.save(os.path.join(path, f'{id}.gif'), writer='pillow', fps=30)
        plt.close()
    except:
        pass


# Exploration: A Study of Count-Based Exploration for Deep Reinforcement Learning
# https://arxiv.org/abs/1611.04717
# Based on https://github.com/clementbernardd/Count-Based-Exploration/blob/main/python/simhash.py
class SimHash(object) :
    def __init__(self, state_emb, k, beta=0.01) :
        ''' Hashing between continuous state space and discrete state space '''
        self.hash = {}
        self.A = np.random.normal(0, 1, (k , state_emb))
        self.beta = beta

    def count(self, states):
        ''' Increase the count for the states and retourn the counts '''
        counts = []
        for state in states:
            # key = str(np.sign(self.A @ state).tolist())
            if torch.is_tensor(state):
                np_state = state.detach().cpu().numpy()
            else:
                np_state = state
            key = (np.asarray(np.sign(self.A @ np_state), dtype=int) + 1) // 2  # to binary code array
            key = int(''.join(key.astype(str).tolist()), base=2)  # to int (binary)
            if key in self.hash:
                self.hash[key] = self.hash[key] + 1
            else:
                self.hash[key] = 1
            counts.append(self.hash[key])

        # return np.array(counts)
        count_reward = self.beta / np.sqrt(np.array(counts).reshape(-1, 1))

        if torch.is_tensor(states):
            # return torch.from_numpy(count_reward).to(states.device)
            return torch.tensor(count_reward, dtype=states.dtype).to(states.device)
        else:
            return count_reward


def read_traj_data(expert_path):
    ob_flatten = True
    traj_data = np.load(expert_path, allow_pickle=True)
    traj_limit_idx = len(traj_data['obs'])
    observations = traj_data['obs'][:traj_limit_idx]
    actions = traj_data['actions'][:traj_limit_idx]
    rewards = traj_data['rewards'][:traj_limit_idx]
    episode_returns = traj_data['episode_returns'][:traj_limit_idx]
    episode_starts = traj_data['episode_starts'][:traj_limit_idx]
    if 'next_obs' in traj_data.keys():
        next_observations = traj_data['next_obs'][:traj_limit_idx]
    else:
        next_observations = np.concatenate(
            [traj_data['obs'][1:traj_limit_idx], traj_data['obs'][traj_limit_idx - 1:traj_limit_idx]])

    if len(observations.shape) > 2 and ob_flatten:
        observations = np.reshape(observations, [-1, np.prod(observations.shape[1:])])
    if len(actions.shape) > 2:
        actions = np.reshape(actions, [-1, np.prod(actions.shape[1:])])
    demo_dones = np.concatenate(
        (episode_starts[1:], np.array([1])))  # array of indiciator, where 1 means the end of episode -- Judy
    return observations, actions, rewards, demo_dones, next_observations, episode_returns


class LinearSchedule(object):
    """
    Linear interpolation between initial_p and final_p over
    schedule_timesteps. After this many timesteps pass final_p is
    returned.

    :param schedule_timesteps: (int) Number of timesteps for which to linearly anneal initial_p to final_p
    :param initial_p: (float) initial output value
    :param final_p: (float) final output value
    """

    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p
        self.current_value = initial_p
        self.current_step = 0

    def value(self, step):
        if step != 0:
            self.current_step += 1
        fraction = min(self.current_step / self.schedule_timesteps, 1.0)
        self.current_value = self.initial_p + fraction * (self.final_p - self.initial_p)
        return self.current_value


class ExponentialSchedule(object):
    def __init__(self, initial_value, decay_rate):
        self.current_value = initial_value
        self.decay_rate = decay_rate

    def value(self, step=666):
        if step != 0:
            self.current_value *= self.decay_rate
        return self.current_value


class CosineSchedule:
    def __init__(self, initial_value, decay_steps, final_p=0):
        self.initial_value = initial_value
        self.final_p = final_p
        self.decay_steps = decay_steps
        self.current_step = 0
        self.current_value = initial_value

    def value(self, step=666):
        if step != 0:
            self.current_step += 1
        cosine_decay = 0.5 * (1 + math.cos(math.pi * min(self.current_step / self.decay_steps, 1)))
        self.current_value = (self.initial_value - self.final_p) * cosine_decay + self.final_p
        return self.current_value


def split_ratio(n, start=0.0, end=1.0):
    if n == 1:
        return [end - start]
    common_difference = (end - start) / (n - 1)
    sequence = [start + i * common_difference for i in range(n-1)]
    sequence.append(end)

    return sequence


def sin_split_ratio(n, start=0.5, end=1.5):
    if n == 1:
        return [end - start]
    line = (start + end) / 2
    p = (line - start)
    common_difference = math.pi / (n-1)
    sequence = [math.sin(-math.pi/2 + i*common_difference)*(p) + line for i in range(n)]

    return sequence


# demo re-labeling
def demo_re_labeling(env, demos, reLabeling=False, overtimeDone=False, reLabelingLinear=None, reLabelingDone=False, reLabelingSuccess=False, reLabelingSqil=False):
    _demo_ls = []
    _demo_t = []
    _sum_rewards = 0
    _ep_r_his = []
    ep_step = 0
    ignore = False
    for d_tp in demos:
        if env.absorption:
            if ep_step % env.max_episode_steps == 0 and ep_step != 0:
                ep_step = 0
                ignore = False
            if ep_step < env.max_episode_steps and ignore:
                ep_step += 1
                continue
        demo_obs, demo_action, demo_reward, demo_next_obs, demo_done = d_tp[:5]
        info = None if len(d_tp) == 5 else d_tp[5]
        terminated, truncated = False, demo_done
        if env.is_sparse:
            demo_reward, terminated, truncated = env.sparse_reward(demo_next_obs, demo_reward, demo_done, info)
        if overtimeDone: terminated = terminated or truncated
        _sum_rewards += demo_reward
        _demo_t.append((demo_obs, demo_action, demo_reward, demo_next_obs, terminated, truncated))
        ep_step += 1
        if terminated or truncated:
            ignore = True
            if reLabelingLinear is not None:
                sp_r = reLabelingLinear if _sum_rewards > 0 else -reLabelingLinear
                sr = split_ratio(len(_demo_t), 1-sp_r, 1+sp_r)
            for i, _d in enumerate(_demo_t):
                if reLabeling:
                    if reLabelingLinear is not None:
                        _re_reward = _d[2] if _d[4] and not reLabelingDone else _sum_rewards / len(_demo_t) * sr[i]
                    else:
                        _re_reward = _d[2] if _d[4] and not reLabelingDone else _sum_rewards / len(_demo_t)
                    # if _d[5] and env.r_fail != 0 and (len(_demo_t) != env.max_episode_steps): _re_reward = env.r_fail
                else:
                    _re_reward = _d[2]
                    if reLabelingSuccess:
                        success = _demo_t[-1][4]
                        if success and i > len(_demo_t)-env.opt_n:
                            _re_reward = _sum_rewards / env.opt_n
                if reLabelingSqil:
                    _re_reward = 1
                _demo_ls.append((_d[0], _d[1], _re_reward, _d[3], _d[4], _d[4] or _d[5]))
            _demo_t = []
            _ep_r_his.append(_sum_rewards)
            _sum_rewards = 0

    # Time feature processing
    if env.time_reward is not None and len(_demo_ls[0][0]) < env.state_dim:
        _demo_ls = addTimeFeature(_demo_ls, env.max_episode_steps)
    elif env.time_reward is None and len(_demo_ls[0][0]) > env.state_dim:
        _demo_ls = rmTimeFeature(_demo_ls)

    return _demo_ls, _ep_r_his


def addTimeFeature(demos, max_step):
    demos_time = []
    step_cnt = 0
    for d_tp in demos:
        state, action, reward, next_state, done = d_tp[:5]
        info = None if len(d_tp) == 5 else d_tp[5]
        step_cnt += 1
        demos_time.append((np.concatenate((state, [1-(step_cnt-1)/max_step])), action, reward, np.concatenate((next_state, [1-(step_cnt)/max_step])), done, info))
        if done: step_cnt = 0
    return demos_time


def rmTimeFeature(demos):
    demos_t = []
    for d_tp in demos:
        state, action, reward, next_state, done = d_tp[:5]
        info = None if len(d_tp) == 5 else d_tp[5]
        demos_t.append((np.delete(state, -1, 0), action, reward, np.delete(next_state, -1, 0), done, info))
    return demos_t


def demo_quality_control(demos, ep_r_his, opt_demo, opt_ep_r_his, expertTrajQ):
    ep_index = 0
    insert_step_cnt = 0
    demos_ls = []
    ep_r_his_ls = []
    if len(ep_r_his) > 1:
        ep_n = (1 - min(max(expertTrajQ, 0), 1))*len(ep_r_his)
        for item in demos:
            if ep_index < ep_n: demos_ls.append(item)
            else: break
            if item[5]:
                ep_r_his_ls.append(ep_r_his[ep_index])
                ep_index += 1
        ep_index = 0
        for item in opt_demo:
            if ep_index < len(ep_r_his)-ep_n: 
                demos_ls.append(item)
                insert_step_cnt += 1
            else: break
            if item[5]:
                ep_r_his_ls.append(opt_ep_r_his[ep_index])
                ep_index += 1
    else:
        len_cnt = 0
        for item in demos:
            len_cnt += 1
            if item[5]: break
        sample_list = [i for i in range(len_cnt)]
        sample_list = random.sample(sample_list, int(len_cnt*expertTrajQ))
        for i, item in enumerate(demos):
            if i in sample_list: 
                insert_step_cnt += 1
                demos_ls.append(opt_demo[i])
            else: 
                demos_ls.append(item)
            if item[5]: 
                ep_r_his_ls.append(opt_ep_r_his[ep_index]*expertTrajQ+ep_r_his[ep_index]*(1-expertTrajQ))
                if len(ep_r_his) > 1: ep_index += 1
                break
    return demos_ls, ep_r_his_ls, ep_index, insert_step_cnt


def demo_number_control(demos, ep_r_his, expertTrajN, min_size):
    ep_index = 0
    step_cnt, copy_cnt = 0, 0
    temp_traj = []
    demos_ls = []
    demos_epr_ls = []
    if expertTrajN == 0:
        return demos_ls, demos_epr_ls, ep_index, step_cnt, copy_cnt, temp_traj
    if expertTrajN < 1:
        len_cnt = 0
        for item in demos:
            len_cnt += 1
            if item[5]: break
        sample_list = [i for i in range(len_cnt)]
        sample_list = random.sample(sample_list, int(len_cnt*expertTrajN))
    for i, item in enumerate(demos):
        tp_item = item[0:5]
        if expertTrajN < 1:
            if i not in sample_list: tp_item = None
        if tp_item is not None:
            step_cnt += 1
            demos_ls.append(tp_item)
            demos_epr_ls.append(ep_r_his[ep_index])
            temp_traj.append((tp_item, ep_r_his[ep_index]))
        if item[5]:
            if len(ep_r_his) > 1: ep_index += 1
            if ep_index >= expertTrajN:
                traj_cnt = 0
                while len(demos_ls) < min_size:
                    demos_ls.append(temp_traj[traj_cnt][0])
                    demos_epr_ls.append(temp_traj[traj_cnt][1])
                    copy_cnt += 1
                    if traj_cnt < len(temp_traj)-1: traj_cnt += 1
                    else: traj_cnt = 0
                break
    return demos_ls, demos_epr_ls, ep_index, step_cnt, copy_cnt, temp_traj

def buildSchedule(type, iniVal, finVal, timesteps):
    '''
    type: the type of schedule
    iniVal: the inital value of the schedule
    finVal: the final value of the schedule
    timesteps: the total steps of the schedule
    '''
    schedule = None
    if 'linear' in type:
        schedule = LinearSchedule(timesteps, initial_p=iniVal, final_p=finVal)
    elif 'exponential' in type:
        schedule = ExponentialSchedule(initial_value=iniVal, decay_rate=0.98)
    elif 'cosine' in type:
        schedule = CosineSchedule(iniVal, timesteps, finVal)
    else:
        schedule = LinearSchedule(timesteps, initial_p=iniVal, final_p=iniVal)
    return schedule
