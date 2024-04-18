import matplotlib.pyplot as plt
import numpy as np
from climber import *
import math

ANGLE_CHANGE = math.radians(5)

class environment:
    def __init__(self, size = 50, holds = np.array([]), agents = []):
        self.size = size
        self.holds = holds
        self.agents = agents

    def add_agent(self):
        self.agents.append(climber())

    def add_hold(self, holds):
        self.holds.append(holds)

    def set_size(self, size):
        self.size = size

    def step(self):
        self.agents[0].torso.left_arm.grab(self.holds)
        self.agents[0].raise_left_arm(ANGLE_CHANGE)
        self.agents[0].raise_left_arm(ANGLE_CHANGE)
        self.agents[0].raise_left_arm(ANGLE_CHANGE)
        self.agents[0].raise_left_arm(ANGLE_CHANGE)




    def render(self, mode = "human"):
        print(self.holds)
        if len(self.holds) > 0:
            plt.plot([point[0] for point in self.holds], [point[1] for point in self.holds], 'ro')
        
        if len(self.agents) > 0:
            for agent in self.agents:
                plt.plot(agent.torso.location[0], agent.torso.location[1], 'go')
                plt.plot([agent.torso.location[0], agent.torso.left_arm.location[0]], [agent.torso.location[1], agent.torso.left_arm.location[1]], linestyle = 'solid')
                plt.plot([agent.torso.location[0], agent.torso.right_arm.location[0]], [agent.torso.location[1], agent.torso.right_arm.location[1]], linestyle = 'solid')
                plt.plot(agent.torso.left_arm.location[0], agent.torso.left_arm.location[1], 'bo')
                plt.plot(agent.torso.right_arm.location[0], agent.torso.right_arm.location[1], 'yo')
        
        plt.xlim(0, self.size)
        plt.ylim(0, self.size)
        plt.show()

if __name__ == "__main__":
    env = environment(100, [], [])
    env.set_size(100)
    env.add_agent()
    env.add_hold(np.asarray((55, 50)))
    env.add_hold(np.asarray((60, 50)))
    env.add_hold(np.asarray((45, 50)))
    print(env.agents[0].torso.right_arm.holding)
    env.render()
    env.step()
    print(env.agents[0].torso.right_arm.holding)
    env.render()
    env.step()
    print(env.agents[0].torso.right_arm.holding)
    env.render()
