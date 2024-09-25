#agent_2로 학습(target_policy 업데이트 방식), 그 뭐냐 입실론 출력
import gymnasium as gym
import math
import random
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import Agent_2
import Memory
from PIL import Image
import os
import plot  # 새로 만든 plot 모듈을 import

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

EPISODES = 500
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4
MEMORY_SIZE = 10000
UPDATE_PERIOD = 10


env = gym.make("CartPole-v1", render_mode = 'rgb_array')

# GPU를 사용할 경우
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_actions = env.action_space.n
state, info = env.reset()
n_observations = len(state)

memory = Memory.ReplayMemory(MEMORY_SIZE)
agent = Agent_2.DQNagent(n_observations, n_actions, memory)

episode_durations = []

is_ipython = 'inline' in plt.get_backend()

for i_episode in range(EPISODES+1):
    state, _ = env.reset()
    state = torch.tensor(state, device=device, dtype=torch.float32).unsqueeze(0)
    total_reward = 0
    for t in count():

        action = agent.select_act(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        total_reward += reward
        done = terminated or truncated

        if terminated:
            reward -= 10
            next_state = None
        else:
            next_state = torch.tensor(observation, device=device, dtype=torch.float32).unsqueeze(0)

        reward = torch.tensor([reward], device=device)
        memory.push(state, action, next_state, reward)
        state = next_state
        agent.learn()

        if done:
            print(f"Episode: {i_episode}, Total Reward: {total_reward:.2f}, Epsilon: {agent.eps_threshold:.2f}")
            episode_durations.append(t + 1)
            break

    if i_episode % UPDATE_PERIOD == 0:
        agent.update_target_network()


    if i_episode % 50 == 0:

        state, _ = env.reset()
        state = torch.tensor(state, device=device, dtype=torch.float32).unsqueeze(0)
        total_reward = 0
        frames = []
        for t in count():
            frame = env.render()
            frames.append(frame)

            action = agent.target_select_act(state)
            observation, reward, terminated, truncated, _ = env.step(action.item())

            total_reward += reward
            done = terminated or truncated
            if terminated:
                reward -= 10
                next_state = None
            else:
                next_state = torch.tensor(observation, device=device, dtype=torch.float32).unsqueeze(0)
            state = next_state

            if done:
                print(f"***Episode: {i_episode}, Total Reward: {total_reward:.2f}***")
                break

        for i, frame in enumerate(frames):
            img = Image.fromarray(frame)
            img.save(os.path.join('image', f'frame_target_{i_episode}_{i}.png'))  # 'image' 폴더에 저장



# 모든 에피소드가 끝난 후 한 번만 그래프 출력
print('Complete')
plot.plot_durations(episode_durations, is_ipython, show_result=True)
plt.ioff()
plt.show()

