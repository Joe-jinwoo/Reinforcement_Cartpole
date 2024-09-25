import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import Memory
import Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.005
EPS_DECAY = 3000
TAU = 0.005
LR = 1e-4
MEMORY_SIZE = 10000


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

env = gym.make('CartPole-v1')


class DQNagent:
    def __init__(self, n_observations, n_actions, memory):
        self.n_observations = n_observations
        self.n_actions = n_actions

        self.policy_net = Model.DQN(n_observations, n_actions).to(device)
        self.target_net = Model.DQN(n_observations, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), LR)
        self.steps_done = 0
        self.memory = memory
        self.sample = 0
        self.eps_threshold = 0

    def select_act(self, state):  # 행동 선택(담당)함수
        self.eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        self.sample = random.random()

        if self.sample > self.eps_threshold:  # 무작위 값 > 앱실론값 : 학습된 신경망이 옳다고 생각하는 쪽으로,
            with torch.no_grad():
                return self.policy_net(state).max(1).indices.view(1, 1)
        else:  # 무작위 값 < 앱실론값 : 무작위로 행동
            return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

    def target_select_act(self, state):  # 행동 선택(담당)함수
        with torch.no_grad():
            return self.target_net(state).max(1).indices.view(1, 1)

    def learn(self):  # 메모리에 쌓아둔 경험들을 재학습(replay)하며, 학습하는 함수
        if len(self.memory) < BATCH_SIZE:  # 메모리에 저장된 에피소드가 batch 크기보다 작으면 그냥 학습을 거름.
            return

        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))  # 기존의 batch를 요소별 리스트로 분리해줄 수 있게끔


        #다음 스테이트가 있는 애들을 표시해줌, 예를 들어, next_state가 {state1, None, state2}라면 non_final_mask는 {True, False, True}가 된다.
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=device, dtype=torch.bool)
        #다음 스테이트가 있는 애들끼리만 묶어준 것. 위에 예면 {state1, state2}가 됨.
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        current_q = self.policy_net(state_batch).gather(1, action_batch)

        #일단 다 0으로 만들어준 다음에
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        #다음 스테이트가 있는 애들은 그 그 스테이트의 q 최댓값 넣어주고
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        # 기대 Q 값 계산
        expected_q = (next_state_values * GAMMA) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(current_q, expected_q.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def update_target_network(self):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
        self.target_net.load_state_dict(target_net_state_dict)