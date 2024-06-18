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

from model import *


is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
plt.ion()

def plot_rewards(show_result=False, reward_tracking=[], filename='testing'):
        plt.figure(1)
        reward_t = torch.tensor(reward_tracking, dtype=torch.float)
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.plot(reward_t.numpy(), color='gray')
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

def plot_state(env = None, save = False, show_result = False):
    env.render_run(save, show_result)