import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
from PIL import Image
import numpy as np
from climber import *
import math
import gym

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

ANGLE_CHANGE = math.radians(5)

class environment(gym.Env):
    def __init__(self, size = 50, holds = np.array([]), agent = climber()):
        self.size = size
        self.holds = holds
        self.agent = agent
        self.action_space = gym.spaces.Discrete(8) 

    def reset_agent(self):
        self.agent = climber()

    def add_hold(self, holds):
        self.holds.append(holds)

    def set_size(self, size):
        self.size = size

    def reset(self):
        self.reset_agent()
        return self.get_observation()

    def step(self, action):
        previous_torso_location = self.agent.torso.location

        if action == 0:
            self.agent.raise_left_arm(ANGLE_CHANGE)
        elif action == 1:
            self.agent.lower_left_arm(ANGLE_CHANGE)
        elif action == 2:
            self.agent.raise_right_arm(ANGLE_CHANGE)
        elif action == 3:
            self.agent.lower_right_arm(-ANGLE_CHANGE)
        elif action == 4:
            self.agent.grab_left_arm(self.holds)
        elif action == 5:
            self.agent.grab_right_arm(self.holds)
        elif action == 6:
            self.agent.release_left_arm()
        elif action == 7:
            self.agent.release_right_arm()

        new_torso_location = self.agent.torso.location

        reward = (new_torso_location[0] + new_torso_location[1]) ** 2

        self.agent.energy -= np.linalg.norm(new_torso_location - previous_torso_location)
        self.agent.energy -= 1

        done = False
        if (self.agent.torso.location[0] == self.size and self.agent.torso.location[1] == self.size) or self.agent.energy <= 0:
            done = True 
        
        return self.get_observation(), reward, done, None, None
    
    def render(self, mode = "human"):
        if len(self.holds) > 0:
            plt.plot([point[0] for point in self.holds], [point[1] for point in self.holds], 'ro')

        plt.plot(self.agent.torso.location[0], self.agent.torso.location[1], 'go')
        plt.plot([self.agent.torso.location[0], self.agent.torso.left_arm.location[0]], [self.agent.torso.location[1], self.agent.torso.left_arm.location[1]], linestyle = 'solid')
        plt.plot([self.agent.torso.location[0], self.agent.torso.right_arm.location[0]], [self.agent.torso.location[1], self.agent.torso.right_arm.location[1]], linestyle = 'solid')
        plt.plot(self.agent.torso.left_arm.location[0], self.agent.torso.left_arm.location[1], 'bo')
        plt.plot(self.agent.torso.right_arm.location[0], self.agent.torso.right_arm.location[1], 'yo')
    
        plt.xlim(0, self.size)
        plt.ylim(0, self.size)
        plt.show()

    def render_run(self, save = False, show_result = False):
        plt.figure(2)

        if len(self.holds) > 0:
            plt.plot([point[0] for point in self.holds], [point[1] for point in self.holds], 'ro')

        plt.plot(self.agent.torso.location[0], self.agent.torso.location[1], 'go')
        plt.plot([self.agent.torso.location[0], self.agent.torso.left_arm.location[0]], [self.agent.torso.location[1], self.agent.torso.left_arm.location[1]], linestyle = 'solid')
        plt.plot([self.agent.torso.location[0], self.agent.torso.right_arm.location[0]], [self.agent.torso.location[1], self.agent.torso.right_arm.location[1]], linestyle = 'solid')
        plt.plot(self.agent.torso.left_arm.location[0], self.agent.torso.left_arm.location[1], 'bo')
        plt.plot(self.agent.torso.right_arm.location[0], self.agent.torso.right_arm.location[1], 'yo')
    
        plt.xlim(0, self.size)
        plt.ylim(0, self.size)

        plt.pause(0.0001)  # pause a bit so that plots are updated
        if is_ipython:
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())
        plt.clf()

    def get_observation(self):
        return np.concatenate((self.agent.torso.location, self.agent.torso.left_arm.location, self.agent.torso.right_arm.location, [self.agent.torso.left_arm.holding], [self.agent.torso.right_arm.holding]))

if __name__ == "__main__":
    env = environment(100, [])
    env.set_size(100)
    env.add_hold(np.asarray((10, 10)))
    env.render()
    env.step()

