import matplotlib.pyplot as plt
import numpy as np
import math


LIMB_LENGTH = 5

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
        self.holding = False
        self.angle = angle

    def release(self):
        self.holding = False
    
    def grab(self, holds = []):
        for hold in holds:
            if np.linalg.norm(self.location - hold) < 1:
                self.holding = True


class right_arm(body_part):
    def __init__(self, location = np.asarray((0, 0)), angle = 0):
        super().__init__(location, angle)
        self.holding = False
        self.angle = angle

    def release(self):
        self.holding = False
    
    def grab(self, holds):
        for hold in holds:
            if np.linalg.norm(self.location - hold) < 1:
                self.holding = True
        

class climber:
    def __init__(self, location = np.asarray((50, 50))):
        self.torso = torso(location)

    def raise_left_hand(self, angle, release = False):
        if not self.torso.left_arm.holding:
            self.torso.left_arm.angle += angle
            self.torso.left_arm.location = np.asarray((self.torso.location[0] - LIMB_LENGTH * np.cos(self.torso.left_arm.angle), self.torso.location[1] + LIMB_LENGTH * np.sin(self.torso.left_arm.angle)))
        else:
            if release:
                self.torso.left_arm.release()
                self.torso.left_arm.angle += angle
                self.torso.left_arm.location = np.asarray((self.torso.location[0] - LIMB_LENGTH * np.cos(self.torso.left_arm.angle), self.torso.location[1] + LIMB_LENGTH * np.sin(self.torso.left_arm.angle)))
            else:
                if not self.torso.right_arm.holding:
                    self.torso.location = np.asarray((self.torso.left_arm.location[0] - LIMB_LENGTH * np.cos(angle), self.torso.left_arm.location[1] + LIMB_LENGTH * np.sin(angle)))
                    self.torso.right_arm.location = np.asarray((self.torso.location[0] - LIMB_LENGTH * np.cos(self.torso.right_arm.angle), self.torso.location[1] + LIMB_LENGTH * np.sin(self.torso.right_arm.angle)))

    def lower_left_hand(self, angle, release = False):
        if not self.torso.left_arm.holding:
            self.torso.left_arm.angle -= angle
            self.torso.left_arm.location = np.asarray((self.torso.location[0] - LIMB_LENGTH * np.cos(self.torso.left_arm.angle), self.torso.location[1] + LIMB_LENGTH * np.sin(self.torso.left_arm.angle)))
        else:
            if release:
                self.torso.left_arm.release()
                self.torso.left_arm.angle -= angle
                self.torso.left_arm.location = np.asarray((self.torso.location[0] - LIMB_LENGTH * np.cos(self.torso.left_arm.angle), self.torso.location[1] + LIMB_LENGTH * np.sin(self.torso.left_arm.angle)))
            else:
                if not self.torso.right_arm.holding:
                    self.torso.location = np.asarray((self.torso.left_arm.location[0] - LIMB_LENGTH * np.cos(-1 * angle), self.torso.left_arm.location[1] + LIMB_LENGTH * np.sin(-1 * angle)))
                    self.torso.right_arm.location = np.asarray((self.torso.location[0] - LIMB_LENGTH * np.cos(self.torso.right_arm.angle), self.torso.location[1] + LIMB_LENGTH * np.sin(self.torso.right_arm.angle)))

    def raise_right_hand(self, angle, release = False):
        if not self.torso.right_arm.holding:
            self.torso.right_arm.angle += angle
            self.torso.right_arm.location = np.asarray((self.torso.location[0] + LIMB_LENGTH * np.cos(self.torso.right_arm.angle), self.torso.location[1] + LIMB_LENGTH * np.sin(self.torso.right_arm.angle)))
        else:
            if release:
                self.torso.right_arm.release()
                self.torso.right_arm.angle += angle
                self.torso.right_arm.location = np.asarray((self.torso.location[0] + LIMB_LENGTH * np.cos(self.torso.right_arm.angle), self.torso.location[1] + LIMB_LENGTH * np.sin(self.torso.right_arm.angle)))
            else:
                if not self.torso.left_arm.holding:
                    self.torso.location = np.asarray((self.torso.right_arm.location[0] + LIMB_LENGTH * np.cos(angle), self.torso.right_arm.location[1] + LIMB_LENGTH * np.sin(angle)))
                    self.torso.left_arm.location = np.asarray((self.torso.location[0] + LIMB_LENGTH * np.cos(self.torso.left_arm.angle), self.torso.location[1] + LIMB_LENGTH * np.sin(self.torso.left_arm.angle)))

    def lower_right_hand(self, angle, release = False):
        if not self.torso.right_arm.holding:
            self.torso.right_arm.angle -= angle
            self.torso.right_arm.location = np.asarray((self.torso.location[0] + LIMB_LENGTH * np.cos(self.torso.right_arm.angle), self.torso.location[1] + LIMB_LENGTH * np.sin(self.torso.right_arm.angle)))
        else:
            if release:
                self.torso.right_arm.release()
                self.torso.right_arm.angle -= angle
                self.torso.right_arm.location = np.asarray((self.torso.location[0] + LIMB_LENGTH * np.cos(self.torso.right_arm.angle), self.torso.location[1] + LIMB_LENGTH * np.sin(self.torso.right_arm.angle)))
            else:
                if not self.torso.left_arm.holding:
                    self.torso.location = np.asarray((self.torso.right_arm.location[0] + LIMB_LENGTH * np.cos(-1 * angle), self.torso.right_arm.location[1] + LIMB_LENGTH * np.sin(-1 * angle)))
                    self.torso.left_arm.location = np.asarray((self.torso.location[0] + LIMB_LENGTH * np.cos(self.torso.left_arm.angle), self.torso.location[1] + LIMB_LENGTH * np.sin(self.torso.left_arm.angle)))