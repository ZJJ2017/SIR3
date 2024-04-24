# -*- coding: utf-8 -*-


import argparse
from train import train, eval_policy
import jsbsim_gym.jsbsim_gym # This line makes sure the environment is registered

import os
import datetime

import gym
from utils.envWrapper import ReacherWrapper, PusherWrapper, XPositionWrapper
from utils.envWrapper import RobosuiteWrapper, gymnasium_wrapper
from utils.helper_functions import set_random_seed
from utils.config import YamlConfig, ConfigDict
from utils.expUtils import str2bool
from rl_algorithms import R2, TD3, SQIL, TD3fD, R2
# from rl_algorithms import DDPG, SIR3, TD3, SILCR, SAC, SQIL, TD3fD, SIR3

import time

'''
run

nohup xxx > out1.txt 2>&1 &
nohup xxx > /dev/null 2>&1 &

python run.py --env Reacher-v2 --policy SIR3 --isSparse --reLabeling
'''


def parse_args() -> argparse.Namespace:
    # configurations
    parser = argparse.ArgumentParser(description="Pytorch RL rl_algorithms")
    parser.add_argument("--env", default="Pusher-v2", help="OpenAI gym environment name")
    parser.add_argument("--policy", default="TD3", help="policy name (TD3, TD3fD, SQIL or R2)")
    parser.add_argument("--seed", type=int, default=222222, help="random seed for reproducibility")
    parser.add_argument("--total_timesteps", type=int, default=None, help="total step num")
    parser.add_argument("--eval_freq", type=int, default=None, help="eval model freq")
    parser.add_argument("--save_freq", type=int, default=None, help="save model freq")
    parser.add_argument("--max_episode_steps", type=int, default=None, help="max episode step")
    parser.add_argument("--eval_episodes", type=int, default=None, help="episode of test during training")
    parser.add_argument("--load_model", default="", help="Model load file name")
    parser.add_argument("--test", dest="test", action="store_true", help="test mode (no training)")
    # --- Experimental hyperparameter ---
    parser.add_argument("--label", type=str, default='')
    parser.add_argument("--remark", type=str, default='')
    parser.add_argument("--isSparse", dest="isSparse", action="store_true", help="Sparse reward or not")
    parser.add_argument("--overTimeDone", dest="overTimeDone", action="store_true", help="End of timeout")
    parser.add_argument("--overTimeReward", dest="overTimeReward", type=float, default=0, help="Whether to add timeout reward")
    parser.add_argument("--bcLossRatio", dest="bcLossRatio", type=float, default=None, help="BCLoss ratio")
    parser.add_argument("--rlLossRatio", dest="rlLossRatio", type=float, default=None, help="RLLoss ratio")
    parser.add_argument("--sampleP", dest="sampleP", type=float, default=0.7, help="The proportion of expert samples in the batch")
    parser.add_argument("--samplePlanner", type=str, default='fixed', help="Sample planner, linear, exponential, cosine, fixed")
    parser.add_argument("--bclossPlanner", type=str, default='linear', help="BC planner, linear, exponential, cosine, fixed")
    parser.add_argument("--bclossZoom", type=float, default=None, help="BC zoom")
    parser.add_argument("--bclossFinalV", type=float, default=None, help="the final value of BC loss ratio")
    parser.add_argument("--rllossPlanner", type=str, default='fixed', help="rl planner, linear, exponential, cosine, fixed")
    parser.add_argument("--expertDynamic", type=str2bool, default=True, help="Dynamic update of expert samples")
    parser.add_argument("--rewardThreshold", dest="rewardThreshold", type=float, default=None, help="Expert sample update threshold")
    parser.add_argument("--reLabeling", dest="reLabeling", action="store_true", help="Adaptive reward re-labeling")
    parser.add_argument("--reLabelingDynamic", dest="reLabelingDynamic", action="store_true", help="Adaptive dynamic reward re-labeling")
    parser.add_argument("--reLabelingOnline", dest="reLabelingOnline", action="store_true", help="Online samples re-labeled")
    parser.add_argument("--exploreReward", type=str, default='None', help="Explore reward, hash or icm")
    parser.add_argument("--lr", dest="lr",  type=float, default=None, help="Learning rate")
    parser.add_argument("--lrDecay", dest="lrDecay", action="store_true", help="Learning rate decay")
    parser.add_argument("--expertSampleRatio", dest="expertSampleRatio",  type=float, default=0, help="The proportion of expert buffer sampling after pre-training")
    parser.add_argument("--actorLossClamp", dest="actorLossClamp",  type=float, default=None, help="actor loss clamp")
    parser.add_argument("--actorGradClamp", dest="actorGradClamp",  type=float, default=None, help="actor grad clamp")
    parser.add_argument("--criticGradClamp", dest="criticGradClamp",  type=float, default=None, help="critic grad clamp")
    parser.add_argument("--pretrainStep", dest="pretrainStep", type=int, default=None, help="Pretrain step")
    parser.add_argument("--pretrainBC", dest="pretrainBC", type=str2bool, default=True, help="Pretrain with BC loss only")
    parser.add_argument("--trainBC", dest="trainBC", action="store_true", help="Train with BC loss only")
    parser.add_argument("--trainWithoutBC", dest="trainWithoutBC", action="store_true", help="Train without BC loss")
    parser.add_argument("--trainWithoutPER", dest="trainWithoutPER", type=str2bool, default=True, help="Train without priority experience replay")
    parser.add_argument("--weightDecay", dest="weightDecay",  type=float, default=None, help="weight decay")
    parser.add_argument("--reLabelingLinear", dest="reLabelingLinear",  type=float, default=None, help="Linear re-labeling")
    parser.add_argument("--reLabelingDone", dest="reLabelingDone", action="store_true", help="Finish also re-labeling")
    parser.add_argument("--reLabelingSuccess", dest="reLabelingSuccess", action="store_true", help="R2 re-labeling")
    parser.add_argument("--reLabelingSqil", dest="reLabelingSqil", action="store_true", help="SQIL re-labeling")
    parser.add_argument("--failPenalty", dest="failPenalty", action="store_true", help="Failure penalty")
    parser.add_argument("--expertTrajN", dest="expertTrajN",  type=float, default=None, help="Demo Number [1,2,3,4,5]")
    parser.add_argument("--expertTrajQ", dest="expertTrajQ",  type=float, default=None, help="Demo Quality [0.5]-[Insert range sample ratio]")
    parser.add_argument("--demoPath", type=str, default=None, help="demo path")
    parser.add_argument("--onlyTrain", dest="onlyTrain", action="store_true", help="Only train, without eval and save")
    parser.add_argument("--skipBufferUpdate", action="store_true", help="Do not update buffer")
    parser.add_argument("--useDiscriminator", dest="useDiscriminator", action="store_true", help="Using a Discriminator to Replace Rule-based Reward Shaping")

    return parser.parse_args()


def main():
    """Main."""
    args = parse_args()

    # env initialization
    env = gym.make(args.env)
    if args.max_episode_steps is None:
        args.max_episode_steps = env._max_episode_steps
    else:
        env._max_episode_steps = args.max_episode_steps
    if args.env == 'Reacher-v2':
        env = ReacherWrapper(env, args.isSparse, args.overTimeReward)
    elif args.env == 'Pusher-v2':
        env = PusherWrapper(env, args.isSparse, args.overTimeReward)
    # else:
    #     env = XPositionWrapper(env, args.isSparse, args.overTimeReward, args.env)

    # set a random seed
    set_random_seed(args.seed, env)

    # === config ===
    NOWTIMES = datetime.datetime.now()
    curr_time = NOWTIMES.strftime("%y%m%d_%H%M%S")
    exp_path = os.path.join('checkpoint', str(args.label)) if len(str(args.label)) > 0 else 'checkpoint'
    exp_path = os.path.join(exp_path, str(args.env).replace('-', '_').lower(), str(args.policy).lower())
    if args.isSparse: exp_path = os.path.join(exp_path, 'sparse')
    if args.overTimeReward is not None: exp_path = os.path.join(exp_path, 'overtime')
    remark_str = ''
    if args.expertTrajN is not None: remark_str += f'_expertTrajN-{args.expertTrajN}'
    if args.expertTrajQ is not None: remark_str += f'_expertTrajQ-{args.expertTrajQ}'
    if args.exploreReward != 'None': remark_str += '_explore'
    if args.reLabelingLinear is not None: remark_str += '_relabeling-linear'
    if args.reLabelingDynamic: remark_str += '_relabeling-dynamic'
    if args.reLabelingSuccess: remark_str += '_relabeling-success'
    if args.reLabelingSqil: remark_str += '_relabeling-sqil'
    if args.trainWithoutBC: remark_str += '_without-bc'
    if args.trainBC: remark_str += '_without-rl'
    if args.sampleP == 0.5: remark_str += '_sample1-1'
    # if args.bclossPlanner != 'fixed': remark_str += '_decay-bc'
    if args.rllossPlanner != 'fixed': remark_str += '_decay-rl'
    if args.samplePlanner != 'fixed': remark_str += '_decay-sample'
    if args.remark != '': remark_str += ('_'+args.remark)
    # exp_path = os.path.join(exp_path, str(curr_time)+remark_str+'_seed'+str(args.seed))
    exp_path = os.path.join(exp_path, 'seed'+str(args.seed)+remark_str+'_'+str(curr_time))
    os.makedirs(exp_path, exist_ok=True)
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print(f"Checkpoint: {exp_path}")
    print("---------------------------------------")
    if 'Fetch' in args.env:
        cfg_name = args.env.lower().replace('dense', '')
    else:
        cfg_name = args.env
    # cfg_policy = 'sir3' if 'sir3' in str(args.policy).lower() else args.policy
    cfg_policy = args.policy
    cfg_path = os.path.join('configs', str(cfg_name).replace('-', '_').lower(), str(cfg_policy).lower()+'.yaml')
    cfg = YamlConfig(cfg_path).get_config_dict()
    cfg.exp_path = exp_path
    cfg.policy = args.policy
    cfg.seed = args.seed
    cfg.test = args.test
    # --- Copy code ---
    # bck_path = os.path.join(exp_path, 'code')
    # os.makedirs(bck_path, exist_ok=True)
    # shutil.copytree('./configs', os.path.join(bck_path, 'configs'))
    # shutil.copytree('./rl_algorithms', os.path.join(bck_path, 'rl_algorithms'))
    # shutil.copytree('./utils', os.path.join(bck_path, 'utils'))
    # --- Experimental parameter ---
    if args.total_timesteps is not None:
        cfg.hyper_params.total_timesteps = args.total_timesteps
    if args.eval_freq is None:
        args.eval_freq = cfg.hyper_params.total_timesteps / 50
    if args.save_freq is None:
        args.save_freq = cfg.hyper_params.total_timesteps / 5
    if args.eval_episodes is None:
        args.eval_episodes = 50
        if args.max_episode_steps > 500: args.eval_episodes = 20
    cfg.hyper_params.eval_freq = args.eval_freq
    cfg.hyper_params.save_freq = args.save_freq
    cfg.hyper_params.eval_episodes = args.eval_episodes
    # exp
    if args.lr is not None:
        cfg.learner_cfg.lr_actor = args.lr
        cfg.learner_cfg.lr_critic = args.lr
    if args.weightDecay is not None:
        cfg.learner_cfg.weight_decay = args.weightDecay
    cfg.hyper_params.overTimeReward = args.overTimeReward
    cfg.hyper_params.isSparse = args.isSparse
    cfg.hyper_params.overTimeDone = args.overTimeDone
    cfg.hyper_params.sampleP = args.sampleP
    cfg.hyper_params.samplePlanner = args.samplePlanner
    cfg.hyper_params.bclossPlanner = args.bclossPlanner
    cfg.hyper_params.rllossPlanner = args.rllossPlanner
    cfg.hyper_params.expertDynamic = args.expertDynamic
    cfg.hyper_params.reLabeling = args.reLabeling
    cfg.hyper_params.reLabelingDynamic = args.reLabelingDynamic
    cfg.hyper_params.reLabelingOnline = args.reLabelingOnline
    cfg.hyper_params.exploreReward = args.exploreReward
    cfg.hyper_params.lrDecay = args.lrDecay
    cfg.hyper_params.expertSampleRatio = args.expertSampleRatio
    cfg.hyper_params.actorLossClamp = args.actorLossClamp
    cfg.hyper_params.actorGradClamp = args.actorGradClamp
    cfg.hyper_params.criticGradClamp = args.criticGradClamp
    cfg.hyper_params.pretrainBC = args.pretrainBC
    cfg.hyper_params.trainBC = args.trainBC
    cfg.hyper_params.trainWithoutBC = args.trainWithoutBC
    cfg.hyper_params.trainWithoutPER = args.trainWithoutPER
    cfg.hyper_params.reLabelingLinear = args.reLabelingLinear
    cfg.hyper_params.reLabelingDone = args.reLabelingDone
    cfg.hyper_params.reLabelingSuccess = args.reLabelingSuccess
    cfg.hyper_params.reLabelingSqil = args.reLabelingSqil
    cfg.hyper_params.failPenalty = args.failPenalty
    cfg.hyper_params.expertTrajN = args.expertTrajN
    cfg.hyper_params.expertTrajQ = args.expertTrajQ
    cfg.hyper_params.onlyTrain = args.onlyTrain
    cfg.hyper_params.skipBufferUpdate = args.skipBufferUpdate
    cfg.hyper_params.useDiscriminator = args.useDiscriminator

    if args.pretrainStep is not None: cfg.hyper_params.pretrainStep = args.pretrainStep
    if args.bcLossRatio is not None: cfg.hyper_params.bcLossRatio = args.bcLossRatio
    if args.rlLossRatio is not None: cfg.hyper_params.rlLossRatio = args.rlLossRatio
    if args.demoPath is not None: cfg.hyper_params.demo_path = args.demoPath
    if args.rewardThreshold is not None: cfg.hyper_params.rewardThreshold = args.rewardThreshold
    if args.bclossZoom is not None: cfg.hyper_params.bc_decay_zoom = args.bclossZoom
    if args.bclossFinalV is not None: cfg.hyper_params.bc_loss_FinalV = args.bclossFinalV
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
    cfg.env.gymnasium = True if 'Fetch' in args.env else False

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
            policy = TD3fD.TD3fD(kwargs, cfg)
        else:
            policy = TD3.TD3(**kwargs)
    elif "r2" in cfg.policy.lower():
        kwargs["policy_noise"] = cfg.noise_cfg.policy_noise * max_action
        kwargs["noise_clip"] = cfg.noise_cfg.noise_clip * max_action
        kwargs["policy_freq"] = cfg.hyper_params.policy_freq
        policy = R2.R2(kwargs, cfg)
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
        print("policy eval ->")
        eval_policy(policy, cfg.env.name, cfg.seed)
    # GO GO GO
    train(policy, cfg)


if __name__ == "__main__":
    main()
