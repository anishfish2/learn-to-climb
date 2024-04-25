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

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 5000
TAU = 0.005
LR = 1e-4


def run(filename = 'testing', episodes = 5, size = 100, verbose = True, agent_energy = 500):

    is_ipython = 'inline' in matplotlib.get_backend()
    if is_ipython:
        from IPython import display
    plt.ion()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class DQN(nn.Module):

        def __init__(self, n_observations, n_actions):
            super(DQN, self).__init__()
            self.layer1 = nn.Linear(n_observations, 128)
            self.layer2 = nn.Linear(128, 128)
            self.layer4 = nn.Linear(128, n_actions)

        def forward(self, x):
            x = F.relu(self.layer1(x))
            x = F.relu(self.layer2(x))
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


    env = environment(size, [], 'saves/' + filename)

    for i in range(0, 90, 5):
        env.add_hold(np.asarray((55, 50)))
        env.add_hold(np.asarray((45, 50)))
        env.add_hold(np.asarray((i, i)))
        env.add_hold(np.asarray((i+10, i+10)))
        env.add_hold(np.asarray((i+10, i)))
        env.add_hold(np.asarray((i, i+10)))


    state = env.reset()
    env.render()
    n_actions = env.action_space.n
    n_observations = len(state)
    policy_net = DQN(n_observations, n_actions).to(device)
    target_net = DQN(n_observations, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(10000)

    steps_done = 0

    def select_action(state):
        global steps_done
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * steps_done / EPS_DECAY)
        steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


    reward_tracking = []

    def plot_rewards(show_result=False):
        plt.figure(1)
        reward_t = torch.tensor(reward_tracking, dtype=torch.float)
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.plot(reward_t.numpy())
        if len(reward_t) >= 100:
            means = reward_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated
        if is_ipython:
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())
        plt.savefig('saves/' + filename + '/rewards.png')

    def plot_state(save = False, show_result = False):
        env.render_run(save, show_result)


    def optimize_model():
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

    if torch.cuda.is_available():
        num_episodes = 600
    else:
        num_episodes = episodes

    for i_episode in tqdm(range(num_episodes)):
        if not verbose:
            print("Episode: ", i_episode)
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        for t in count():
            action = select_action(state)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            memory.push(state, action, next_state, reward)

            state = next_state

            optimize_model()

            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)


            if done:
                #reward_tracking.append(env.agent.distance_to_goal)
                reward_tracking.append(reward)
                if not verbose:
                    plot_rewards()
                break

            if not verbose:
                if i_episode == num_episodes - 1:
                    plot_state(False, False)
                
    plot_state(True, True)

    print('Complete')
    plot_rewards(show_result=True)
    plt.ioff()
    #plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Climbing Reinforcement Learning')
    parser.add_argument('-f', type=str, default='testing', help='name of the file')
    parser.add_argument('-e', type=int, default=5, help='number of episodes')
    parser.add_argument('-s', type=int, default=100, help='size of the environment')
    parser.add_argument('-v', type=bool, default=False, help='verbose')
    parser.add_argument('-ae', type=bool, default=False, help='agent energy')

    args = parser.parse_args()
    filename = args.f
    episodes = args.e
    size = args.s
    verbose = args.v
    agent_energy = args.ae

    steps_done = 0

    if not os.path.exists('saves/' + filename):
        os.makedirs('saves/' + filename)
    else:
        shutil.rmtree('saves/' + filename)
        os.makedirs('saves/' + filename)
    
    f = open('saves/' + filename + "/stats.txt","w+")
    f.write("Filename: " + filename + "\n")
    f.write("Episodes: " + str(episodes) + "\n")
    f.write("Size: " + str(size) + "\n")
    f.write("Agent Energy: " + str(agent_energy) + "\n")
    f.close()

    run(filename, episodes, size, verbose, agent_energy)

    