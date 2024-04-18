import matplotlib.pyplot as plt
import numpy as np
from climber import *

class environment:
    def __init__(self, size = 50, holds = np.array(), agents = []):
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
        self.agents[0].move_left_hand(math.radians(90))

    def render(self, mode = "human"):
        print(self.holds)
        if len(self.holds) > 0:
            plt.plot([point[0] for point in self.holds], [point[1] for point in self.holds], 'ro')
        
        if len(self.agents) > 0:
            for agent in self.agents:
                plt.plot(agent.torso.location[0], agent.torso.location[1], 'go')
                plt.plot([agent.torso.location[0], agent.torso.left_arm.location[0]], [agent.torso.location[1], agent.torso.left_arm.location[1]], linestyle = 'solid')
                plt.plot([agent.torso.location[0], agent.torso.right_arm.location[0]], [agent.torso.location[1], agent.torso.right_arm.location[1]], linestyle = 'solid')
                plt.plot(agent.torso.left_arm.location[0], agent.torso.left_arm.location[1], 'blue')
                plt.plot(agent.torso.right_arm.location[0], agent.torso.right_arm.location[1], 'blue')
        plt.show()

if __name__ == "__main__":
    env = environment(100, [], [])
    env.set_size(10)
    env.add_agent()
    env.add_hold(np.asarray((55, 50)))
    env.add_hold(np.asarray((45, 50)))
    env.render()
    env.step()
    env.render()
