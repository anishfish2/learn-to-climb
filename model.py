#Reference: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

from environment import environment
from climber import *
from collections import namedtuple, deque
from itertools import count
import math
import random
import matplotlib
from matplotlib.animation import FuncAnimation
from PIL import Image
import sys
import os
import shutil
import argparse
from tqdm import tqdm
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

BATCH_SIZE = 32 
GAMMA = 0.25
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-3

class DQN(nn.Module):

    def __init__(self, env_size, n_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 128, kernel_size=3, stride=1, padding=1)


        conv_output_size = self._get_conv_output_size(env_size)

        self.layer1 = nn.Linear(conv_output_size, 128)
        self.layer4 = nn.Linear(128, n_actions)

    def _get_conv_output_size(self, size):
        dummy_input = torch.zeros(1, 1, size, size)
        output = self.conv1(dummy_input)
        output = self.conv2(output)
        #output = self.conv3(output)
        return int(output.numel())

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        #x = F.relu(self.conv3(x))
        x = x.view(x.shape[0], -1)  # Flatten the tensor
        x = F.relu(self.layer1(x))
        return self.layer4(x)
        
Transition = namedtuple('Transition',
                            ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
def optimize_model(memory, policy_net, target_net, optimizer, device):
        if len(memory) < BATCH_SIZE:
            return
        transitions = memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
        optimizer.step()

def select_action(env, state, policy_net, device, steps_done):
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


