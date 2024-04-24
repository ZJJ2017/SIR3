# -*- coding: utf-8 -*-


import argparse

import torch

import os
import time
import numpy as np
import datetime
import pprint

import yaml
import pickle
import copy

import gym
from utils.envWrapper import ReacherWrapper, PusherWrapper, XPositionWrapper, RobosuiteWrapper, gymnasium_wrapper
from utils.helper_functions import set_random_seed
from utils.config import YamlConfig, ConfigDict
from rl_algorithms import TD3, SQIL, TD3fD, R2

# robosuite类包影响gif，导入gym不能生成gif，不导入gymnasium不能生成
# import robosuite as suite
# from robosuite import load_controller_config
# from robosuite.wrappers import GymWrapper

# to gif
import matplotlib.pyplot as plt 
from matplotlib import animation 

def display_frames_as_gif(frames, id, path):
    patch = plt.imshow(frames[0])
    plt.axis('off')
    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=5)
    anim.save(os.path.join(path, f'{id}.gif'), writer='pillow', fps=30)
    plt.close()


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=100, return_mode='666', render=False, save_demo=False, save_gif=False, path=None, log=True, action_noise=0):
    if 'Fetch' in env_name:
        import gymnasium
        import robosuite  # 避免 GLFWError: (65544) b'X11: The DISPLAY environment variable is missing'
        eval_env = gymnasium.make(env_name, policy.env.max_episode_steps, render_mode='rgb_array')
        eval_env._max_episode_steps = eval_env.max_episode_steps = policy.env.max_episode_steps
        eval_env = gymnasium_wrapper(eval_env, policy.env.is_sparse, policy.env.time_reward)
    else:
        eval_env = gym.make(env_name)
        eval_env._max_episode_steps = eval_env.max_episode_steps = policy.env.max_episode_steps
        if env_name == 'Reacher-v2':
            eval_env = ReacherWrapper(eval_env, policy.env.is_sparse, policy.env.time_reward)
        elif env_name == 'Pusher-v2':
            eval_env = PusherWrapper(eval_env, policy.env.is_sparse, policy.env.time_reward)
        else:
            eval_env = XPositionWrapper(eval_env, policy.env.is_sparse, policy.env.time_reward, env_name)
    eval_env.seed(seed)

    score_list = []
    all_demo_ls = []
    demo_ep_r = []
    if save_demo:
        save_demo_path = path
        os.makedirs(save_demo_path, exist_ok=True)
    if save_gif:
        save_gif_path = os.path.join(path, "gif")
        os.makedirs(save_gif_path, exist_ok=True)

    tr_cnt = 0
    for i in range(eval_episodes):
        ep_r = 0.
        demo_ls = []
        frames = []
        state, done = eval_env.reset(), False
        success = False
        while not done:
            if render:
                eval_env.render()
                time.sleep(0.01)
            elif save_gif:
                if 'Fetch' in env_name:
                    frames.append(eval_env.render())
                else:
                    frames.append(eval_env.render(mode='rgb_array'))
            last_state = copy.deepcopy(state)
            action = policy.select_action(np.array(state))
            if action_noise > 0:
                action = (
                    action
                    + np.random.normal(0, action_noise, size=len(action))
                ).clip(-policy.env.max_action, policy.env.max_action)
            state, reward, done, info = eval_env.step(action)
            if eval_env.judge(state, info) and not success: success = True
            demo_ls.append((last_state, action, reward, state, done, info))
            ep_r += reward
        score_list.append(ep_r)
        if (render or save_gif or save_demo) and log:
            print(len(demo_ls), 'ep_r:', ep_r)    
            reward_mean, reward_var = np.mean(score_list), np.var(score_list)
            reward_median = np.median(score_list)
            reward_min, reward_max = np.min(score_list), np.max(score_list)
            print(f'mean/median reward {reward_mean:0.2f}/{reward_median:0.2f}, var {reward_var:0.2f}, max/min reward {reward_max:0.2f}/{reward_min:0.2f}')
        good = True # if not success else False
        # good = True if -50 < ep_r < -35 and len(demo_ls) < 1000 else False
        if good: tr_cnt += 1
        if save_gif and good:
            display_frames_as_gif(frames, i, save_gif_path)
        if save_demo and good:
            all_demo_ls += demo_ls
            demo_ep_r.append(ep_r)
            # if len(all_demo_ls) >= 1000 and tr_cnt >= 5:
            if len(all_demo_ls) >= 1000:
                reward_mean, reward_var = np.mean(demo_ep_r), np.var(demo_ep_r)
                reward_median = np.median(demo_ep_r)
                reward_min, reward_max = np.min(demo_ep_r), np.max(demo_ep_r)
                if log:
                    print(f'mean/median reward {reward_mean:0.2f}/{reward_median:0.2f}, var {reward_var:0.2f}, max/min reward {reward_max:0.2f}/{reward_min:0.2f}')
                _path = f"{str(env_name).lower().split('-')[0]}_demo_r{int(np.mean(reward_mean))}_n{len(all_demo_ls)}_t{tr_cnt}.pkl"
                with open(os.path.join(save_demo_path, _path), "wb") as f:
                    pickle.dump(all_demo_ls, f)
                break

    reward_mean, reward_var = np.mean(score_list), np.var(score_list)
    reward_median = np.median(score_list)
    reward_min, reward_max = np.min(score_list), np.max(score_list)
    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes")
    print(f'mean/median reward {reward_mean:0.2f}/{reward_median:0.2f}, var {reward_var:0.2f}, max/min reward {reward_max:0.2f}/{reward_min:0.2f}')
    print("---------------------------------------")
    if return_mode == 'mean':
        return reward_mean
    elif return_mode == 'all':
        return score_list, reward_mean, reward_var
    else:
        return score_list


def parse_args() -> argparse.Namespace:
    # configurations
    parser = argparse.ArgumentParser(description="Pytorch RL rl_algorithms")
    parser.add_argument("--env", default="Reacher-v2", help="OpenAI gym environment name")
    parser.add_argument("--policy", default="TD3", help="policy name")
    parser.add_argument("--seed", type=int, default=222222, help="random seed for reproducibility")
    parser.add_argument("--start_timesteps", default=5000, type=int, help="Time steps initial random policy is used")
    parser.add_argument("--total_timesteps", type=int, default=500000, help="total step num")
    parser.add_argument("--eval_freq", type=int, default=50000, help="eval model freq")
    parser.add_argument("--save_freq", type=int, default=10000, help="save model freq")
    parser.add_argument("--max_episode_steps", type=int, default=None, help="max episode step")
    parser.add_argument("--eval_episodes", type=int, default=10, help="episode of test during training")
    parser.add_argument("--load_model", default="", help="Model load file name")
    parser.add_argument("--test", dest="test", action="store_true", help="test mode (no training)")
    # --- Experimental hyperparameter ---
    parser.add_argument("--label", type=str, default='')
    parser.add_argument("--remark", type=str, default='')
    parser.add_argument("--isSparse", dest="isSparse", action="store_true")
    parser.add_argument("--overTimeReward", dest="overTimeReward", default=None)
    parser.add_argument("--render", action="store_false")
    parser.add_argument("--save_demo", action="store_true")
    parser.add_argument("--save_gif", action="store_true")
    parser.add_argument("--action_noise", type=float, default=0)

    return parser.parse_args()


def main():
    """Main."""
    args = parse_args()

    # test
    args.test = True
    if args.save_gif:
        args.render = False

    if args.load_model != "":
        if 'reacher_v2' in args.load_model:
            args.env = 'Reacher-v2'
        elif 'pusher_v2' in args.load_model:
            args.env = 'Pusher-v2'
        elif 'halfcheetah' in args.load_model:
            args.env = 'HalfCheetah-v2'
        elif 'hopper' in args.load_model:
            args.env = 'Hopper-v2'
        elif 'walker2d' in args.load_model:
            args.env = 'Walker2d-v2'
        elif 'swimmer' in args.load_model:
            args.env = 'Swimmer-v2'
        elif 'ant' in args.load_model:
            args.env = 'Ant-v2'
        elif 'humanoid' in args.load_model:
            args.env = 'Humanoid-v2'
        elif 'fetchpush' in args.load_model:
            args.env = 'FetchPushDense-v2'
        elif 'fetchreach' in args.load_model:
            args.env = 'FetchReachDense-v2'
        elif 'fetchpick' in args.load_model:
            args.env = 'FetchPickAndPlaceDense-v2'
        elif 'fetchslide' in args.load_model:
            args.env = 'FetchSlideDense-v2'
        if 'td3' in args.load_model or 'sir3' in args.load_model:
            args.policy = 'TD3'
        elif 'ddpg' in args.load_model:
            args.policy = 'DDPG'
        elif 'sqil' in args.load_model:
            args.policy = 'SQIL'
        elif 'sac' in args.load_model:
            args.policy = 'SAC'
    if 'pt' not in args.load_model:
        args.load_model = os.path.join(args.load_model, 'best_model.pt')
    if 'overtime' in args.load_model:
        args.overTimeReward = True

    # env initialization
    if 'Fetch' in args.env:
        import gymnasium
        if args.max_episode_steps is not None:
            env = gymnasium.make(args.env, args.max_episode_steps)
        else:
            env = gymnasium.make(args.env)
        args.max_episode_steps = env._max_episode_steps
        env = gymnasium_wrapper(env, args.isSparse, args.overTimeReward)
    else:
        env = gym.make(args.env)
        if args.max_episode_steps is None:
            args.max_episode_steps = env._max_episode_steps
        else:
            env._max_episode_steps = args.max_episode_steps
        if args.env == 'Reacher-v2':
            env = ReacherWrapper(env, args.isSparse, args.overTimeReward)
        elif args.env == 'Pusher-v2':
            env = PusherWrapper(env, args.isSparse, args.overTimeReward)
        else:
            env = XPositionWrapper(env, args.isSparse, args.overTimeReward, args.env)

    # set a random seed
    set_random_seed(args.seed, env)

    # === config ===
    NOWTIMES = datetime.datetime.now()
    curr_time = NOWTIMES.strftime("%y%m%d_%H%M%S")
    exp_path = os.path.join('checkpoint', str(args.label)) if len(str(args.label)) > 0 else 'checkpoint'
    exp_path = os.path.join(exp_path, str(args.env).replace('-', '_').lower(), str(args.policy).lower())
    if not args.test:
        os.makedirs(exp_path, exist_ok=True)
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print(f"Checkpoint: {exp_path}")
    print("---------------------------------------")
    if 'rob' in args.env:
        cfg_name = 'rob' 
    elif 'Fetch' in args.env:
        cfg_name = args.env.lower().replace('dense', '')
    else:
        cfg_name = args.env
    cfg_policy = 'sir3' if 'sir3' in str(args.policy).lower() else args.policy
    cfg_path = os.path.join('configs', str(cfg_name).replace('-', '_').lower(), str(cfg_policy).lower()+'.yaml')
    cfg = YamlConfig(cfg_path).get_config_dict()
    cfg.exp_path = exp_path
    cfg.policy = args.policy
    cfg.seed = args.seed
    cfg.test = args.test
    # --- Experimental parameter ---
    cfg.hyper_params.eval_freq = args.eval_freq
    cfg.hyper_params.save_freq = args.save_freq
    cfg.hyper_params.eval_episodes = args.eval_episodes
    cfg.hyper_params.start_timesteps = args.start_timesteps
    cfg.hyper_params.total_timesteps = args.total_timesteps
    # exp
    cfg.hyper_params.overTimeReward = args.overTimeReward
    cfg.hyper_params.isSparse = args.isSparse
    # --- Environmental parameter ---
    if 'Fetch' in args.env:
        state_dim = env.observation_space['observation'].shape[0]+env.observation_space['desired_goal'].shape[0]
    else:
        state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    cfg.env = ConfigDict()
    cfg.env.name = args.env
    cfg.env.state_dim = state_dim
    cfg.env.action_dim = action_dim
    cfg.env.max_action = max_action
    cfg.env.max_episode_steps = args.max_episode_steps
    pprint.pprint(cfg)
    # print(yaml.dump(cfg, sort_keys=False, default_flow_style=False))
    # exit(0)

    for key, value in cfg.env.items():
        setattr(env, key, value)
    kwargs = {
        "env": env,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "gamma": cfg.hyper_params.gamma,
        "tau": cfg.hyper_params.tau,
        "batch_size": cfg.hyper_params.batch_size,
        "lr_actor": cfg.learner_cfg.lr_actor,
        "lr_critic": cfg.learner_cfg.lr_critic,
    }

    # Initialize policy
    if "td3" in cfg.policy.lower():
        # Target policy smoothing is scaled wrt the action scale
        kwargs["policy_noise"] = cfg.noise_cfg.policy_noise * max_action
        kwargs["noise_clip"] = cfg.noise_cfg.noise_clip * max_action
        kwargs["policy_freq"] = cfg.hyper_params.policy_freq
        if "fd" in cfg.policy.lower():
            if "ex" in cfg.policy.lower():
                policy = SIR3.EXTD3fD(kwargs, cfg)
            else:
                policy = TD3fD.TD3fD(kwargs, cfg)
        else:
            policy = TD3.TD3(**kwargs)
    elif "sac" in cfg.policy.lower():
        kwargs["alpha_init"] = cfg.hyper_params.alpha_init
        kwargs["lr_alpha"] = cfg.learner_cfg.lr_alpha
        policy = SAC.SAC(**kwargs)
    elif "ddpg" in cfg.policy.lower():
        policy = DDPG.DDPG(**kwargs)
    elif "sqil" in cfg.policy.lower():
        kwargs["policy_noise"] = cfg.noise_cfg.policy_noise * max_action
        kwargs["noise_clip"] = cfg.noise_cfg.noise_clip * max_action
        kwargs["policy_freq"] = cfg.hyper_params.policy_freq
        policy = SQIL.SQIL(kwargs, cfg)
    else:
        print(f"Unsupported policy:{cfg.policy}")
        exit(0)

    if args.load_model != "":
        print("policy load ...")
        policy.load(args.load_model)
        # print("policy eval ->")
        # eval_policy(policy, cfg.env.name, cfg.seed)
    save_path = os.path.join('demo_data', str(args.label)) if len(str(args.label)) > 0 else 'demo_data'
    remark_str = str(curr_time)
    if args.remark != '': remark_str += ('_'+args.remark)
    save_path = os.path.join(save_path, str(args.env).replace('-', '_').lower(), str(args.policy).lower(), remark_str)
    # GO GO GO
    eval_policy(
        policy, cfg.env.name, cfg.seed+666, eval_episodes=10000,
        render=args.render, save_demo=args.save_demo, save_gif=args.save_gif, path=save_path, action_noise=args.action_noise
    )


if __name__ == "__main__":
    main()
