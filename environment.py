import matplotlib.pyplot as plt
import matplotlib
import matplotlib.animation as animation
from PIL import Image
import numpy as np
from climber import *
import math
import gym
import os
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

ANGLE_CHANGE = math.radians(10)

class environment(gym.Env):
    def __init__(self, size = 50, holds = np.array([]), save_path = "", agent_energy = 500, goal = None, agent_starting_location = None):
        self.size = size
        self.holds = holds
        self.agent_starting_energy = agent_energy
        self.agent = climber(agent_energy = self.agent_starting_energy,location=np.asarray((self.size / 2, self.size / 2)))
        self.action_space = gym.spaces.Discrete(8) 
        self.animation_step = 0
        self.num_step = 0
        self.save_path = save_path
        self.observation_tracking = []
        self.holding_tracking = []
        self.action_tracking = []
        self.reward_tracking = []
        self.energy_tracking = []
        self.goal = goal if goal is not None else np.asarray((self.size, self.size))
        self.agent_starting_location = agent_starting_location if agent_starting_location is not None else np.asarray((self.size / 2, self.size / 2))

    def reset_agent(self):
        self.agent = climber(agent_energy = self.agent_starting_energy)
        self.agent.best_distance = None 

    def add_hold(self, holds):
        self.holds.append(holds)

    def set_size(self, size):
        self.size = size

    def reset(self):
        self.reset_agent()
        self.num_step = 0
        self.animation_step = 0
        self.observation_tracking = []
        self.holding_tracking = []
        self.action_tracking = []
        return self.get_observation()

    def step(self, action):
        previous_torso_location = self.agent.torso.location
        if action == 0:
            self.agent.raise_left_arm(ANGLE_CHANGE)
            self.action_tracking.append("Raise Left Arm")

        elif action == 1:
            self.agent.lower_left_arm(ANGLE_CHANGE)
            self.action_tracking.append("Lower Left Arm")

        elif action == 2:
            self.agent.raise_right_arm(ANGLE_CHANGE)
            self.action_tracking.append("Raise Right Arm")

        elif action == 3:
            self.agent.lower_right_arm(ANGLE_CHANGE)
            self.action_tracking.append("Lower Right Arm")

        elif action == 4:
            self.agent.grab_left_arm(self.holds)
            self.action_tracking.append("Grab Left Arm")

        elif action == 5:
            self.agent.grab_right_arm(self.holds)
            self.action_tracking.append("Grab Right Arm")

        elif action == 6:
            self.agent.release_left_arm()
            self.action_tracking.append("Release Left Arm")

        elif action == 7:
            self.agent.release_right_arm()
            self.action_tracking.append("Release Right Arm")

        new_torso_location = self.agent.torso.location
        new_distance = math.hypot(self.goal[0] - new_torso_location[0], self.goal[1] - new_torso_location[1])
        previous_distance = math.hypot(self.goal[0] - previous_torso_location[0], self.goal[1] - previous_torso_location[1])
        reward = 0
        if self.agent.best_distance is None:
            #reward = math.hypot(self.goal[0] - self.agent_starting_location[0], self.goal[1] - self.agent_starting_location[1]) - new_distance
            self.agent.best_distance = new_distance
            
        # elif new_distance < self.agent.best_distance:
        #     reward = self.agent.best_distance - new_distance
        #     reward = 5
        #     self.agent.best_distance = new_distance
        
        else:
            if new_distance < previous_distance:
                reward = 1
            
        self.agent.current_reward = reward
        self.agent.energy -= 1

        done = False
        if (self.agent.torso.location[0] == self.size and self.agent.torso.location[1] == self.size) or self.agent.energy <= 0:
            done = True 

        observation = self.get_observation()

        self.agent.torso.location_tracking.append(self.agent.torso.location)
        self.agent.torso.left_arm.location_tracking.append(self.agent.torso.left_arm.location)
        self.agent.torso.right_arm.location_tracking.append(self.agent.torso.right_arm.location)
        self.reward_tracking.append(reward)
        self.observation_tracking.append(observation)
        self.holding_tracking.append((self.agent.torso.left_arm.holding, self.agent.torso.right_arm.holding))
        self.energy_tracking.append(self.agent.energy)

        self.num_step += 1

        return observation, reward, done, None, None
    
    def update_anim(self, frame):
        self.ax.clear()
        i = self.animation_step % self.num_step
        self.ax.set_title((f"Step {i}, Energy {self.energy_tracking[i]}, Reward {round(self.reward_tracking[i], 4)}, Left {self.holding_tracking[i][0]}, Right {self.holding_tracking[i][1]}, Action {self.action_tracking[i]}"), wrap = True)

        if len(self.holds) > 0:
            self.ax.plot([point[0] for point in self.holds], [point[1] for point in self.holds], 'ro')

        self.ax.plot(self.agent.torso.location_tracking[i][0], self.agent.torso.location_tracking[i][1], 'go')
   
        self.ax.plot([self.agent.torso.location_tracking[i][0], self.agent.torso.left_arm.location_tracking[i][0]], [self.agent.torso.location_tracking[i][1], self.agent.torso.left_arm.location_tracking[i][1]], linestyle = 'solid')
        self.ax.plot([self.agent.torso.location_tracking[i][0], self.agent.torso.right_arm.location_tracking[i][0]], [self.agent.torso.location_tracking[i][1], self.agent.torso.right_arm.location_tracking[i][1]], linestyle = 'solid')

        self.ax.plot(self.agent.torso.left_arm.location_tracking[i][0], self.agent.torso.left_arm.location_tracking[i][1], 'bo')
        self.ax.plot(self.agent.torso.right_arm.location_tracking[i][0], self.agent.torso.right_arm.location_tracking[i][1], 'yo')
        
        self.animation_step += 1
        x1 = [0, self.size, self.size, 0, 0]
        y1 = [0, 0, self.size, self.size, 0]
        self.ax.plot(x1, y1, color='black')


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
        plt.clf()
        
        if len(self.holds) > 0:
            plt.plot([point[0] for point in self.holds], [point[1] for point in self.holds], 'ro')

        plt.plot(self.agent.torso.location[0], self.agent.torso.location[1], 'go')
        plt.plot([self.agent.torso.location[0], self.agent.torso.left_arm.location[0]], [self.agent.torso.location[1], self.agent.torso.left_arm.location[1]], linestyle = 'solid')
        plt.plot([self.agent.torso.location[0], self.agent.torso.right_arm.location[0]], [self.agent.torso.location[1], self.agent.torso.right_arm.location[1]], linestyle = 'solid')
        plt.plot(self.agent.torso.left_arm.location[0], self.agent.torso.left_arm.location[1], 'bo')
        plt.plot(self.agent.torso.right_arm.location[0], self.agent.torso.right_arm.location[1], 'yo')

        plt.xlim(0, self.size)
        plt.ylim(0, self.size)

        plt.title(f'Current Reward: {round(self.agent.current_reward, 4)} Energy: {self.agent.energy}')
        #plt.pause(5)

        if is_ipython:
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())

        if save:
            csv_dict = {}


            csv_dict["Left Arm Location X"] = [i[0] for i in self.agent.torso.left_arm.location_tracking]
            csv_dict["Left Arm Location Y"] = [i[1] for i in self.agent.torso.left_arm.location_tracking]
            csv_dict["Right Arm Location X"] = [i[0] for i in self.agent.torso.right_arm.location_tracking]
            csv_dict["Right Arm Location Y"] = [i[1] for i in self.agent.torso.right_arm.location_tracking]
            csv_dict["Holding Left"] = [i[0] for i in self.holding_tracking]
            csv_dict["Holding Right"] = [i[1] for i in self.holding_tracking]          
            csv_dict["Action"] = self.action_tracking
            
            self.fig, self.ax = plt.subplots()
            self.ax.set_xlim(0, self.size)
            self.ax.set_ylim(0, self.size)
            self.ax.grid(False)
            self.ax.set_aspect('equal', 'box')

            x1 = [0, self.size, self.size, 0, 0]
            y1 = [0, 0, self.size, self.size, 0]
            self.ax.plot(x1, y1, color='black')
            real_num = self.num_step
            ani = animation.FuncAnimation(self.fig, self.update_anim, frames=real_num, interval=50, blit=False, repeat=True)
            _vid_name = os.path.join(self.save_path + '/', 'vid.gif')
            ani.save(filename=_vid_name, writer="pillow")
            
            

            df = pd.DataFrame(csv_dict)
            df.to_csv(self.save_path + '/data.csv')

        
    def get_observation(self):
        
       # Discretize observation space

       

        state_of_env = np.zeros(shape=(self.size,self.size))

        state_of_env[-int(self.agent.torso.location[1] + .5)][int(self.agent.torso.location[0] + .5)] = 1

        if self.agent.torso.left_arm.holding:
            state_of_env[-int(self.agent.torso.left_arm.location[1] + .5)][int(self.agent.torso.left_arm.location[0] + .5)] = 2
        else:
            state_of_env[-int(self.agent.torso.left_arm.location[1] + .5)][int(self.agent.torso.left_arm.location[0] + .5)] = 3
        if self.agent.torso.right_arm.holding:
            state_of_env[-int(self.agent.torso.right_arm.location[1] + .5)][int(self.agent.torso.right_arm.location[0] + .5)] = 4
        else:
            state_of_env[-int(self.agent.torso.right_arm.location[1] + .5)][int(self.agent.torso.right_arm.location[0] + .5)] = 5

        for hold in self.holds:
            if state_of_env[-hold[1]][hold[0]] == 1:
                state_of_env[-hold[1]][hold[0]] = 9
            elif state_of_env[-hold[1]][hold[0]] == 2:
                state_of_env[-hold[1]][hold[0]] = 10
            elif state_of_env[-hold[1]][hold[0]] == 3:
                state_of_env[-hold[1]][hold[0]] = 11
            elif state_of_env[-hold[1]][hold[0]] == 4:
                state_of_env[-hold[1]][hold[0]] = 12
            elif state_of_env[-hold[1]][hold[0]] == 5:
                state_of_env[-hold[1]][hold[0]] = 13 
            else:
                state_of_env[-hold[1]][hold[0]] = 14


        self.observation_tracking.append(state_of_env)

        return state_of_env
 

if __name__ == "__main__":
    env = environment(15, [], "testing_environment/")
    env.set_size(100)
    env.add_hold(np.asarray((14, 10)))
    env.render_run(show_result=True)
    env.step(7)
    env.step(0)
    env.step(0)

