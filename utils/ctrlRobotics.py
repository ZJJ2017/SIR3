import gymnasium as gym
import keyboard

import os
import copy
import numpy as np
import pickle

env_name = "FetchReachDense-v2"
env = gym.make(env_name, render_mode='human')

observation, info = env.reset(seed=42)

move = 0.2

demos = []
demo_ep_r = []
tr_cnt = 0

ep_r = 0

for _ in range(1000):
    # action = env.action_space.sample()
    action = [0, 0, 0, 0]
    key_pressed = keyboard.read_event(suppress=True)
    if key_pressed.event_type == keyboard.KEY_DOWN:
        if key_pressed.name == 'up':
            print("up")
            action[0] += move
        elif key_pressed.name == 'down':
            print("down")
            action[0] -= move
        elif key_pressed.name == 'left':
            print("left")
            action[1] += move
        elif key_pressed.name == 'right':
            print("right")
            action[1] -= move
        elif key_pressed.name == 'w':
            print("up")
            action[2] += move
        elif key_pressed.name == 's':
            print("down")
            action[2] -= move
        elif key_pressed.name == 'q':
            break
    last_state = copy.deepcopy(observation)
    observation, reward, terminated, truncated, info = env.step(action)
    ep_r += reward
    env.render()
    demos.append((last_state, action, reward, observation, terminated or truncated, info))

    if terminated or truncated:
        tr_cnt += 1
        observation, info = env.reset()
        print(ep_r)
        demo_ep_r.append(ep_r)
        ep_r = 0

save_demo_path = './'
_path = f"{str(env_name).lower().split('-')[0]}_demo_r{int(np.mean(demo_ep_r))}_n{len(demos)}_t{tr_cnt}.pkl"
with open(os.path.join(save_demo_path, _path), "wb") as f:
    pickle.dump(demos, f)

env.close()



