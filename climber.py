import matplotlib.pyplot as plt
import numpy as np

class limb:
    def __init__(self, location = np.asarray((0, 0)), angle = 0):
        self.location = location
        self.angle = angle

class left_arm(limb):
    def __init__(self, location = np.asarray((0, 0)), angle = 0):
        super().__init__(location, angle)

class right_arm(limb):
    def __init__(self, location = np.asarray((0, 0)), angle = 0):
        super().__init__(location, angle)

class climber:
    def __init__(self, location = np.asarray((50, 50))):
        self.torso = {"location": location}
        self.left_hand = {"location": location + np.asarray((-5, 0)), "angle": 0}
        self.right_hand = {"location": location + np.asarray((5, 0)), "angle": 0}

    def move_left_hand(self, angle):
        self.left_hand["angle"] = angle
        self.left_hand["location"] = location
