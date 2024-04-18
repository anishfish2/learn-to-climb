import matplotlib.pyplot as plt
import numpy as np
from climber import *
import math

ANGLE_CHANGE = math.radians(5)

class environment:
    def __init__(self, size = 50, holds = np.array([]), agent = climber()):
        self.size = size
        self.holds = holds
        self.agent = agent

    def reset_agent(self):
        self.agent = climber()

    def add_hold(self, holds):
        self.holds.append(holds)

    def set_size(self, size):
        self.size = size

    def reset(self):
        self.holds = []
        self.reset_agent()

    def step(self, action):
        print("stepping")

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

if __name__ == "__main__":
    env = environment(100, [])
    env.set_size(100)
    env.add_hold(np.asarray((10, 10)))
    env.render()
    env.step()
    env.render()
    env.step()
    env.render()
