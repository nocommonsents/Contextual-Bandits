__author__ = 'bixlermike'

import numpy as np
import random as rn
import time

from exploChallenge.policies.ContextualBanditPolicy import ContextualBanditPolicy


def rargmax(x):
    m = np.amax(x)
    indices = np.nonzero(x == m)[0]
    return rn.choice(indices)

class MostRecent(ContextualBanditPolicy):

    def __init__(self):
        self.counts = {}
        self.values = {}
        self.arrival_times = {}
        return

    def getActionToPerform(self, visitor,possibleActions):

        for action in possibleActions:
            if action.getID() not in self.arrival_times:
                self.arrival_times[action.getID()] = int(time.time())
                self.counts[action.getID()] = 0
                self.values[action.getID()] = 0

        all_times = [self.arrival_times[a.getID()] for a in possibleActions]
        action = possibleActions[rargmax(all_times)]

        return action

    def updatePolicy(self, content, chosen_arm, reward):
        self.counts[chosen_arm.getID()] += 1
        n = self.counts[chosen_arm.getID()]
        value = self.values[chosen_arm.getID()]

        new_value = value + (reward - value) / (n + 1)
        self.values[chosen_arm.getID()] = new_value

        return


