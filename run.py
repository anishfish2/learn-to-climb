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
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from model import *
from utils import *

def run(filename = 'testing', episodes = 5, size = 100, verbose = True, agent_energy = 500):

    is_ipython = 'inline' in matplotlib.get_backend()
    if is_ipython:
        from IPython import display
    plt.ion()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = environment(size, [], 'saves/' + filename, agent_energy)

    env.add_hold(np.asarray((45, 50)))
    env.add_hold(np.asarray((55, 50)))
    for i in range(0, 98):
        env.add_hold(np.asarray((i, i)))
        env.add_hold(np.asarray((i+1, i)))
        env.add_hold(np.asarray((i, i+1)))
        env.add_hold(np.asarray((i+2, i)))
        env.add_hold(np.asarray((i, i+2)))

    state = env.reset()
    n_actions = env.action_space.n
    env_size = env.size
    policy_net = DQN(env_size, n_actions).to(device)
    target_net = DQN(env_size, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(10000)



    reward_tracking = []


    num_episodes = episodes

    for i_episode in tqdm(range(num_episodes)):
        if not verbose:
            print("Episode: ", i_episode)
        
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        steps_done = 0
        for t in count():
            action = select_action(env, state, policy_net, device, steps_done)
            steps_done += 1
            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)

            if terminated:
                next_state = None
                break
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            memory.push(state, action, next_state, reward)

            state = next_state
            optimize_model(memory=memory, policy_net=policy_net, target_net=target_net, optimizer=optimizer, device=device)
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)

            reward_tracking.append(reward)

            if verbose:
                plot_rewards(show_result=True, reward_tracking=reward_tracking, filename=filename)
                plot_state(env, False, True)

    plot_state(env, True, True)
    print('Complete')
    plot_rewards(show_result=True, reward_tracking=reward_tracking, filename=filename)
    plt.ioff()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Climbing Reinforcement Learning')
    parser.add_argument('-f', type=str, default='testing', help='name of the file')
    parser.add_argument('-e', type=int, default=5, help='number of episodes')
    parser.add_argument('-s', type=int, default=100, help='size of the environment')
    parser.add_argument('-v', type=bool, default=False, help='verbose')
    parser.add_argument('-ae', type=int, default=500, help='agent starting energy')

    args = parser.parse_args()
    filename = args.f
    episodes = args.e
    size = args.s
    verbose = args.v
    agent_energy = args.ae
    
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
    f.write("Date:" + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")

    f.close()

    run(filename, episodes, size, verbose, agent_energy)

    