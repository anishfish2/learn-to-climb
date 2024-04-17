import matplotlib.pyplot as plt
import numpy as np
from climber import *

class environment:
    def __init__(self, size = 50, holds = [], agents = []):
        self.size = size
        self.holds = holds
        self.agents = agents

    def add_agent(self, agent):
        self.agents.append(climber())

    def add_hold(self, holds):
        self.holds.append(holds)

    def set_size(self, size):
        self.size = size

    def step(self):
        pass

    def render(self, mode = "human"):
        print(self.holds)
        if len(self.holds) > 0:
            plt.plot([point[0] for point in self.holds], [point[1] for point in self.holds], 'ro')
        
        if len(self.agents) > 0:
            for agent in self.agents:
                print(agent.torso["location"])
                print(agent.left_hand["location"])
                print(agent.right_hand["location"])
                plt.plot(agent.torso["location"][0], agent.torso["location"][1], 'go')
                plt.plot([agent.torso["location"][0], agent.left_hand["location"][0]], [agent.torso["location"][1], agent.left_hand["location"][1]], linestyle = 'solid')
                plt.plot([agent.torso["location"][0], agent.right_hand["location"][0]], [agent.torso["location"][1], agent.right_hand["location"][1]], linestyle = 'solid')
                plt.plot(agent.left_hand["location"][0], agent.left_hand["location"][1], 'blue')
                plt.plot(agent.right_hand["location"][0], agent.right_hand["location"][1], 'blue')
        plt.show()

if __name__ == "__main__":
    env = environment(100, [], [])
    env.set_size(10)
    env.add_agent("agent1")
    env.add_hold(np.asarray((100, 100)))
    env.render()