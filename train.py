import numpy as np
import torch
import gym
import argparse
import os
import time
import math
import pprint

import torch.optim as optim

from utils.envWrapper import ReacherWrapper, PusherWrapper, XPositionWrapper, RobosuiteWrapper, gymnasium_wrapper

from utils.buffer import ReplayBuffer
from utils.expUtils import plotSmoothAndSaveFig, Log, display_frames_as_gif
from enjoy import eval_policy as enjoy_eval

from tensorboardX import SummaryWriter


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=100, return_mode='666', render=False, save_gif_num=0, path=None):
    if 'Fetch' in env_name:
        import gymnasium
        eval_env = gymnasium.make(env_name, policy.env.max_episode_steps, render_mode='rgb_array')
        eval_env._max_episode_steps = eval_env.max_episode_steps = policy.env.max_episode_steps
        eval_env = gymnasium_wrapper(eval_env, policy.env.is_sparse, policy.env.time_reward)
        max_save_gif_num = save_gif_num = 0
    else:
        eval_env = gym.make(env_name)
        # eval_env.seed(seed + 666)
        eval_env._max_episode_steps = eval_env.max_episode_steps = policy.env.max_episode_steps
        if env_name == 'Reacher-v2':
            eval_env = ReacherWrapper(eval_env, policy.env.is_sparse, policy.env.time_reward)
            max_save_gif_num = save_gif_num*2
        elif env_name == 'Pusher-v2':
            eval_env = PusherWrapper(eval_env, policy.env.is_sparse, policy.env.time_reward)
            max_save_gif_num = save_gif_num*2
        else:
            # eval_env = XPositionWrapper(eval_env, policy.env.is_sparse, policy.env.time_reward, env_name)
            max_save_gif_num = save_gif_num
    # eval_env = policy.env.eval_env
    eval_env.seed(seed)

    score_list = []
    success_cnt = 0
    fail_cnt = 0
    gif_cnt = 0

    if save_gif_num > 0:
        save_gif_path = os.path.join(path, "gif")
        os.makedirs(save_gif_path, exist_ok=True)
    for _ in range(eval_episodes):
        ep_r = 0.
        frames = []
        state, done = eval_env.reset(), False
        state, done = eval_env.reset(), False
        while not done:
            if render:
                eval_env.render()
                time.sleep(0.01)
            elif save_gif_num > 0 and gif_cnt < max_save_gif_num:
                if 'Fetch' in env_name:
                    frames.append(eval_env.render())
                else:
                    frames.append(eval_env.render(mode='rgb_array'))
            action = policy.select_action(np.array(state))
            state, reward, done, info = eval_env.step(action)
            ep_r += reward
        score_list.append(ep_r)
        
        if render:
            policy.log.record('ep_r:', ep_r)
        if info.get('success') is not None:
            if info['success']: 
                success_cnt += 1
                if success_cnt <= save_gif_num:
                    display_frames_as_gif(frames, f'success_{success_cnt}', save_gif_path)
                    gif_cnt += 1
            else:
                fail_cnt += 1
                if fail_cnt <= save_gif_num:
                    display_frames_as_gif(frames, f'fail_{gif_cnt}', save_gif_path)
                    gif_cnt += 1
        else:
            fail_cnt += 1
            if fail_cnt <= save_gif_num:
                display_frames_as_gif(frames, f'random_{gif_cnt}', save_gif_path)
                gif_cnt += 1

    reward_mean, reward_var = np.mean(score_list), np.var(score_list)
    reward_median = np.median(score_list)
    reward_min, reward_max = np.min(score_list), np.max(score_list)
    success_ratio = success_cnt*100/eval_episodes
    if hasattr(policy, 'log'):
        policy.log.record("---------------------------------------")
        policy.log.record(f"Evaluation over {eval_episodes} episodes, success {success_cnt}, ({success_ratio:0.2f}%)")
        policy.log.record(f'mean/median reward {reward_mean:0.2f}/{reward_median:0.2f}, var {reward_var:0.2f}, max/min reward {reward_max:0.2f}/{reward_min:0.2f}')
        policy.log.record("---------------------------------------")
    if return_mode == 'mean':
        return reward_mean
    elif return_mode == 'all':
        return score_list, reward_mean, reward_var, success_ratio
    else:
        return score_list


def train(policy, cfg):
    # Experimental data related configuration
    writer = SummaryWriter(cfg.exp_path)
    if not hasattr(policy, 'log'):
        policy.log = Log(os.path.join(cfg.exp_path, "e.log"))
    train_record_fig = os.path.join(cfg.exp_path, "train_episode_rewards.jpg")
    test_record_fig = os.path.join(cfg.exp_path, "test_episode_rewards.jpg")

    policy.log.record(pprint.pformat(cfg)+'\n')

    # Evaluate untrained policy
    evaluations = []
    train_episode_rewards = []
    lr_list = []
    actor_loss = []
    critic_loss = []

    state, done = policy.env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    best_test_reward = -math.inf
    best_test_reward_var = math.inf
    best_test_success = 0

    if hasattr(policy, 'pretrain'):
        # A certain number of random samples are added to the online buffer for pre-training
        if cfg.hyper_params.randomSampleNumInPretrain is not None:
            state = policy.env.reset()
            for _ in range(cfg.hyper_params.randomSampleNumInPretrain):
                episode_timesteps += 1
                action = policy.env.action_space.sample()
                next_state, reward, done, info = policy.env.step(action)
                done_bool = float(done) if episode_timesteps < cfg.env.max_episode_steps else 0
                policy.update_buffer(state, action, next_state, reward, done_bool, done, info)
                state = next_state
                if done:
                    state = policy.env.reset()
                    episode_timesteps = 0
            state = policy.env.reset()
            episode_timesteps = 0

        policy.pretrain()
        if cfg.hyper_params.pretrainStep > 0:
            test_result = eval_policy(policy, cfg.env.name, cfg.seed)
            for _i, it in enumerate(test_result):
                evaluations.append(it)
                writer.add_scalar('pretrain_test/reward', it, _i)
            plotSmoothAndSaveFig(10, evaluations, os.path.join(cfg.exp_path, "pretrain_test_rewards.jpg"))
            evaluations = []
            # if np.mean(test_result) < cfg.hyper_params.rewardThreshold: cfg.hyper_params.expertSampleRatio = 0

    if cfg.hyper_params.lrDecay:
        policy.actor_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            policy.actor_optimizer, cfg.hyper_params.total_timesteps, eta_min=cfg.learner_cfg.lr_actor/10)
        policy.critic_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            policy.critic_optimizer, cfg.hyper_params.total_timesteps, eta_min=cfg.learner_cfg.lr_critic/10)

    train_time = 0
    start_time = time.time()
    episode_time = 0
    for t in range(int(cfg.hyper_params.total_timesteps)):
        
        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < cfg.hyper_params.start_timesteps:
            action = policy.env.action_space.sample()
        else:
            action = (
                policy.select_action(np.array(state))
                + np.random.normal(0, cfg.env.max_action * cfg.noise_cfg.expl_noise, size=cfg.env.action_dim)
            ).clip(-cfg.env.max_action, cfg.env.max_action)
        # Perform action
        next_state, reward, done, info = policy.env.step(action)
        done_bool = float(done) if episode_timesteps < cfg.env.max_episode_steps else 0

        if not cfg.hyper_params.skipBufferUpdate:
            # Store data in replay buffer
            policy.update_buffer(state, action, next_state, reward, done_bool, done, info)

        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        trian_state = 'pass'
        if t >= cfg.hyper_params.start_timesteps:
            trian_state = policy.train()

        if done:
            # Times
            episode_time = time.time() - start_time
            start_time = time.time()
            train_time += episode_time
            policy.log.record(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f} Done: {str(bool(done_bool))}")
            if hasattr(policy,'on_done'):
                done_log = policy.on_done(state, action, next_state, reward, done_bool, done, info)
                if done_log is not None:
                    actor_loss.append(done_log[0])
                    critic_loss.append((done_log[1]+done_log[2])/2)
            # Record
            train_episode_rewards.append(episode_reward)
            writer.add_scalar('train/reward', episode_reward, episode_num)
            writer.add_scalar('train/st_reward', episode_reward, t)
            # Reset environment
            state, done = policy.env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1
            # Times
            writer.add_scalar('train/st_time', episode_time, t)

        # Evaluate episode
        if ((t + 1) % cfg.hyper_params.eval_freq == 0 and t >= cfg.hyper_params.start_timesteps
                and trian_state != 'pass' and not cfg.hyper_params.onlyTrain):
            test_result = eval_policy(policy, cfg.env.name, cfg.seed, cfg.hyper_params.eval_episodes, return_mode='all')
            # if test_result[1] > best_test_reward and test_result[2]*0.5 < best_test_reward_var:
            best_flag = False
            if test_result[3] > best_test_success: best_flag = True
            elif test_result[1] > best_test_reward: best_flag = True
            if best_flag:
                best_test_reward = test_result[1]
                best_test_reward_var = test_result[2]
                best_test_success = test_result[3]
                policy.save(cfg.exp_path, info='best_model')
                policy.log.record(f'=> save best_model, best_success {best_test_success:0.2f}%, best_reward (mean/var): {test_result[1]:0.2f}/{test_result[2]:0.2f}')
                # save gif
                # need del os.environ['LD_PRELOAD']
                # if cfg.env.max_episode_steps < 1000:
                #     gif_num = 1 if cfg.env.max_episode_steps > 100 else 3
                #     enjoy_eval(policy, cfg.env.name, cfg.seed+666, gif_num, save_gif=True, path=cfg.exp_path, log=False)
            for _i, it in enumerate(test_result[0]):
                evaluations.append(it)
            plotSmoothAndSaveFig(50, evaluations, test_record_fig)
            plotSmoothAndSaveFig(50, train_episode_rewards, train_record_fig)
            plotSmoothAndSaveFig(1, actor_loss, os.path.join(cfg.exp_path, "train_actor_loss.jpg"))
            plotSmoothAndSaveFig(1, critic_loss, os.path.join(cfg.exp_path, "train_critic_loss.jpg"))

        # Save model
        if (((t + 1) % cfg.hyper_params.save_freq == 0) and ((t + 1) > 2*cfg.hyper_params.save_freq)
                and trian_state != 'pass' and not cfg.hyper_params.onlyTrain):
            # evaluations.append(eval_policy(policy, cfg.env.name, cfg.seed+666, return_mode='mean'))
            policy.save(cfg.exp_path)
            # pass

        if cfg.hyper_params.lrDecay:
            lr_list.append(policy.actor_scheduler.get_last_lr()[0])
            policy.actor_scheduler.step()
            policy.critic_scheduler.step()

    # termination
    policy.log.record(f'\nTraining takes {train_time/60:0.2f} minutes')
    policy.save(cfg.exp_path)
    policy.load(os.path.join(cfg.exp_path, 'best_model.pt'))
    # gif_num = 1 if cfg.env.max_episode_steps > 500 else 5
    gif_num = 0
    test_result = eval_policy(policy, cfg.env.name, cfg.seed+666, 1000, path=cfg.exp_path, save_gif_num=gif_num)  # Final evaluation 1000 rounds
    for _i, it in enumerate(evaluations):
        writer.add_scalar('test/reward', it, _i)
    plotSmoothAndSaveFig(50, evaluations, test_record_fig)
    plotSmoothAndSaveFig(50, train_episode_rewards, train_record_fig)
    # plotSmoothAndSaveFig(1, test_result, os.path.join(cfg.exp_path, "eval_episode_rewards.jpg"))
    plotSmoothAndSaveFig(1, actor_loss, os.path.join(cfg.exp_path, "train_actor_loss.jpg"))
    plotSmoothAndSaveFig(1, critic_loss, os.path.join(cfg.exp_path, "train_critic_loss.jpg"))
    # if cfg.hyper_params.lrDecay:
    # 	plotSmoothAndSaveFig(10, lr_list, os.path.join(cfg.exp_path, "lr_decay.jpg"))

    policy.log.close()
