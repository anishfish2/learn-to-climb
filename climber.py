import matplotlib.pyplot as plt
import numpy as np
import math

class body_part:
    def __init__(self, location = np.asarray((0, 0)), angle = 0):
        self.location = location
        self.angle = angle

class torso(body_part):
    def __init__(self, location = np.asarray((0, 0)), angle = 0):
        super().__init__(location, angle)
        self.right_arm = right_arm(location + np.asarray((5, 0)), angle = 0)
        self.left_arm = left_arm(location + np.asarray((-5, 0)), angle = 0)

class left_arm(body_part):
    def __init__(self, location = np.asarray((0, 0)), angle = 0):
        super().__init__(location, angle)
        self.holding = None

    def release(self):
        self.holding = False
    
    def grab(self):
        self.holding = True

class right_arm(body_part):
    def __init__(self, location = np.asarray((0, 0)), angle = 0):
        super().__init__(location, angle)
        self.holding = None

    def release(self):
        self.holding = False
    
    def grab(self):
        self.holding = True

class climber:
    def __init__(self, location = np.asarray((50, 50))):
        self.torso = torso(location)

    def move_left_hand(self, angle):
        self.torso.left_arm.angle = angle
        self.torso.left_arm.location = (self.torso.location[0] + 5 * np.cos(angle), self.torso.location[1] + 5 * np.sin(angle))

    def move_right_hand(self, angle):
        self.torso.right_arm.angle = angle
        self.torso.right_arm.location = (self.torso.location[0] + 5 * np.cos(angle), self.torso.location[1] + 5 * np.sin(angle))
