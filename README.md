# Reinforcement Learning from Suboptimal Demonstrations based on Reward Relabeling （SIR3）

This is a PyTorch implementation of a novel training framework combining Self-Imitation learning (SIL) with Reward Relabeling based Reinforcement learning, SIR3 in short.
It is easy for readers to repetition the results in the main article in the widely used MuJoCo environments by following this instruction.

## Setup

You can install the Python liblaries by following the `requirements.txt` with Python 3.7.
Note that there are several components which are required to install manually (e.g., the MoJoCo).

## Example


### Train

You can use SIR3 to train policies in sparse-reward environments.

```shell
# with successful absorbing states
python run.py --env Reacher-v2 --policy SIR3 --isSparse --reLabeling
```
```shell
# without successful absorbing states
python run.py --env HalfCheetah-v2 --policy SIR3 --isSparse --reLabeling --reLabelingDone
```

### Enjoy 

You can use the trained policy to observe, save gifs and save demonstration tracks.
```shell
python enjoy.py --load_model checkpoint/pusher_v2/sir3/240101_666666 --save_demo --save_gif
```


## Reference

1. https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
2. https://github.com/sfujim/TD3
3. https://github.com/medipixel/rl_algorithms
