__author__ = 'dai.shi'

import random as rn
import numpy as np
import operator

from exploChallenge.policies.ContextualBanditPolicy import ContextualBanditPolicy

def rargmax(x):
    m = np.amax(x)
    indices = np.nonzero(x == m)[0]
    return rn.choice(indices)

class eGreedy(ContextualBanditPolicy):


    def __init__(self, epsilon):
        self.epsilon = epsilon
        self.counts = {}
        self.values = {}

    def getEpsilon(self):
        return self.epsilon


    def getActionToPerform(self, visitor,possibleActions):

        for action in possibleActions:
            if action.getID() not in self.counts:
                self.counts[action.getID()] = 1.0
                self.values[action.getID()] = 1.0

        psvalues = [self.values[a.getID()] for a in possibleActions]

        if rn.random() > self.epsilon:
            action = possibleActions[rargmax(psvalues)]
            return action

        else:
            randomIndex = rn.randint(0, len(possibleActions) - 1)
            return possibleActions[randomIndex]

    def updatePolicy(self, content, chosen_arm, reward):
        self.counts[chosen_arm.getID()] += 1
        n = self.counts[chosen_arm.getID()]
        value = self.values[chosen_arm.getID()]

        new_value = value + (reward - value) / (n + 1)
        self.values[chosen_arm.getID()] = new_value

        return


