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
import Agent
import Memory
from PIL import Image
import os
import plot  # 새로 만든 plot 모듈을 import
import glob
import os  # Import os module to handle file deletion
import re
from PIL import Image

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

EPISODES = 500
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.01
LR = 1e-4
MEMORY_SIZE = 10000


for repeat in range(5):
    env = gym.make("CartPole-v1", render_mode = 'rgb_array')

    # GPU를 사용할 경우
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_actions = env.action_space.n
    state, info = env.reset()
    n_observations = len(state)

    memory = Memory.ReplayMemory(MEMORY_SIZE)
    agent = Agent.DQNagent(n_observations, n_actions, memory)

    episode_durations = []

    is_ipython = 'inline' in plt.get_backend()

    # 텍스트 파일로 결과 저장
    with open(f"output_results_{repeat + 1}.txt", "w") as f:  # 결과를 저장할 파일 오픈
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
                agent.update_target_network()

                if done:
                    result = f"Episode: {i_episode}, Total Reward: {total_reward:.2f}, Epsilon: {agent.eps_threshold:.3f}\n"
                    print(result)
                    f.write(result)  # 파일에 결과 기록
                    episode_durations.append(t + 1)
                    break

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
                        result = f"***Episode: {i_episode}, Total Reward: {total_reward:.2f}\n"
                        print(result)
                        f.write(result)  # 파일에 결과 기록
                        break

                for i, frame in enumerate(frames):
                    img = Image.fromarray(frame)
                    img.save(os.path.join(f'frame_target_{repeat + 1}_{i_episode}_{i}.png'))  # 'image' 폴더에 저장

    # 모든 에피소드가 끝난 후 한 번만 그래프 출력
    print('Complete')
    with open(f"output_results_{repeat + 1}.txt", "a") as f:  # 마지막 결과도 파일에 기록
        f.write('Complete\n')

    plot.plot_durations(episode_durations, is_ipython, show_result=True)
    plt.ioff()
    plt.show()


import glob
import os  # Import os module to handle file deletion
import re
from PIL import Image


    # 숫자를 기준으로 정렬하는 함수
def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]


for g_repeat in range(5):
    # GIF 생성 코드
    for i in range(11):
        frames = []
        a = i * 50
        imgs = glob.glob(f"frame_target_{g_repeat + 1}_{a}_*.png")  # 이미지 파일 불러오기
        imgs.sort(key=natural_keys)  # 자연스러운 숫자 순서로 정렬

        for j in imgs:
            new_frame = Image.open(j)
            frames.append(new_frame)

        frame_count = len(frames)
        # GIF 저장 (fps를 제어하려면 duration을 조정)
        gif_path = os.path.join('image', f'cartpole_gif_repeat{g_repeat + 1}_episode{a}_{frame_count}.gif')
        frames[0].save(gif_path, format='GIF',
                       append_images=frames[1:],
                       save_all=True,
                       duration=50, loop=1)

        # PNG 파일 삭제
        for j in imgs:
            os.remove(j)





