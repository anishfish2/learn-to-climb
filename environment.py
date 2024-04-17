import matplotlib.pyplot as plt
import numpy as np

class environment:
    def __init__(self, size = 50, holds = [], agents = []):
        self.size = size
        self.holds = holds
        self.agents = agents

    def add_agent(self, agent):
        self.agents.append(agent)

    def add_hold(self, holds):
        self.holds.append(holds)

    def set_size(self, size):
        self.size = size

    def step(self):
        pass

    def render(self, mode = "human"):
        if len(self.holds) > 0:
            print([point[0] for point in self.holds], [point[1] for point in self.holds])
            plt.plot([point[0] for point in self.holds], [point[1] for point in self.holds], 'ro')
        
        if len(self.agents) > 0:
            plt.plot([point[0] for point in self.agents], [point[1] for point in self.agents], 'bo')
        plt.show()

if __name__ == "__main__":
    env = environment(100, [(5, 5), (60, 28), (15, 76)], [])
    print(env.size)
    print(env.holds)
    print(env.agents)
    env.set_size(10)
    print(env.size)
    env.add_agent("agent1")
    print(env.agents)
    env.add_hold([(100, 100)])
    print(env.holds)
    env.render()