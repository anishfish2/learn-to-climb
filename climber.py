import matplotlib.pyplot as plt
import numpy as np
import math


LIMB_LENGTH = 5

class body_part:
    def __init__(self, location = np.asarray((0, 0)), angle = 0):
        self.location = location
        self.location_tracking = []
        self.angle = angle

class torso(body_part):
    def __init__(self, location = np.asarray((0, 0)), angle = 0):
        super().__init__(location, angle)
        self.right_arm = arm(location + np.asarray((5, 0)), angle = 0)
        self.left_arm = arm(location + np.asarray((-5, 0)), angle = math.radians(180))

class arm(body_part):
    def __init__(self, location = np.asarray((0, 0)), angle = 0):
        super().__init__(location, angle)
        self.holding = False
        self.angle = angle

    def release(self):
        self.holding = False
    
    def grab(self, holds = []):
        for hold in holds:
            if np.linalg.norm(self.location - hold) < 2:
                self.holding = True

class climber:
    def __init__(self, location = np.asarray((50, 50)), agent_energy = 500):
        self.torso = torso(location)
        self.energy = agent_energy
        self.current_reward = 0
        self.location_tracking = []
        
        self.observation_radius = 10
        self.distance_to_goal = math.hypot(location[0] - self.torso.location[0], location[1] - self.torso.location[1])

    def raise_left_arm(self, angle):
        if not self.torso.left_arm.holding:
            self.torso.left_arm.angle -= angle
            self.torso.left_arm.location = np.asarray((self.torso.location[0] + LIMB_LENGTH * np.cos(self.torso.left_arm.angle), self.torso.location[1] + LIMB_LENGTH * np.sin(self.torso.left_arm.angle)))
        elif not self.torso.right_arm.holding:
            self.torso.left_arm.angle -= angle
            self.torso.location = np.asarray((self.torso.left_arm.location[0] - LIMB_LENGTH * np.cos(self.torso.left_arm.angle), self.torso.left_arm.location[1] - LIMB_LENGTH * np.sin(self.torso.left_arm.angle)))
            self.torso.right_arm.location = np.asarray((self.torso.location[0] + LIMB_LENGTH * np.cos(self.torso.right_arm.angle), self.torso.location[1] + LIMB_LENGTH * np.sin(self.torso.right_arm.angle)))

    def lower_left_arm(self, angle):
        if not self.torso.left_arm.holding:
            self.torso.left_arm.angle += angle
            self.torso.left_arm.location = np.asarray((self.torso.location[0] + LIMB_LENGTH * np.cos(self.torso.left_arm.angle), self.torso.location[1] + LIMB_LENGTH * np.sin(self.torso.left_arm.angle)))
        elif not self.torso.right_arm.holding:
            self.torso.left_arm.angle += angle
            self.torso.location = np.asarray((self.torso.left_arm.location[0] - LIMB_LENGTH * np.cos(self.torso.left_arm.angle), self.torso.left_arm.location[1] - LIMB_LENGTH * np.sin(self.torso.left_arm.angle)))
            self.torso.right_arm.location = np.asarray((self.torso.location[0] + LIMB_LENGTH * np.cos(self.torso.right_arm.angle), self.torso.location[1] + LIMB_LENGTH * np.sin(self.torso.right_arm.angle)))

    def raise_right_arm(self, angle):
        if not self.torso.right_arm.holding:
            self.torso.right_arm.angle += angle
            self.torso.right_arm.location = np.asarray((self.torso.location[0] + LIMB_LENGTH * np.cos(self.torso.right_arm.angle), self.torso.location[1] + LIMB_LENGTH * np.sin(self.torso.right_arm.angle)))
        elif not self.torso.left_arm.holding:
            self.torso.right_arm.angle += angle
            self.torso.location = np.asarray((self.torso.right_arm.location[0] - LIMB_LENGTH * np.cos(self.torso.right_arm.angle), self.torso.right_arm.location[1] - LIMB_LENGTH * np.sin(self.torso.right_arm.angle)))
            self.torso.left_arm.location = np.asarray((self.torso.location[0] + LIMB_LENGTH * np.cos(self.torso.left_arm.angle), self.torso.location[1] + LIMB_LENGTH * np.sin(self.torso.left_arm.angle)))

    def lower_right_arm(self, angle):
        if not self.torso.right_arm.holding:
            self.torso.right_arm.angle -= angle
            self.torso.right_arm.location = np.asarray((self.torso.location[0] + LIMB_LENGTH * np.cos(self.torso.right_arm.angle), self.torso.location[1] + LIMB_LENGTH * np.sin(self.torso.right_arm.angle)))
        elif not self.torso.left_arm.holding:
            self.torso.right_arm.angle -= angle
            self.torso.location = np.asarray((self.torso.right_arm.location[0] - LIMB_LENGTH * np.cos(self.torso.right_arm.angle), self.torso.right_arm.location[1] - LIMB_LENGTH * np.sin(self.torso.right_arm.angle)))
            self.torso.left_arm.location = np.asarray((self.torso.location[0] + LIMB_LENGTH * np.cos(self.torso.left_arm.angle), self.torso.location[1] + LIMB_LENGTH * np.sin(self.torso.left_arm.angle)))

    def grab_right_arm(self, holds):
        self.torso.right_arm.grab(holds)
    
    def grab_left_arm(self, holds):
        self.torso.left_arm.grab(holds)

    def release_right_arm(self):
        self.torso.right_arm.release()

    def release_left_arm(self):
        self.torso.left_arm.release()