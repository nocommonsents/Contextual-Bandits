__author__ = 'bixlermike'
# Derived from algorithms from John White and ideas from Dai Shi

# Final verification 8 Apr 2015

import random
import operator
import math
import numpy as np

from exploChallenge.policies.ContextualBanditPolicy import ContextualBanditPolicy


def rargmax(x):
    m = np.amax(x)
    indices = np.nonzero(x == m)[0]
    return random.choice(indices)

class UCB1(ContextualBanditPolicy):


    def __init__(self):
        self.counts = {}
        self.values = {}
        return

    def getActionToPerform(self, visitor, possibleActions):
        arm_values = {}
        arm_counts = {}

        for action in possibleActions:
            if action.getID() not in self.counts:
                self.counts[action.getID()] = 1.0
                self.values[action.getID()] = 1.0
            arm_values[action.getID()] = self.values[action.getID()]
            arm_counts[action.getID()] = self.counts[action.getID()]

        total_count = sum(arm_counts[count] for count in arm_counts)
        # Calculate upper confidence bound of each arm
        ucb_values = [ self.values[a.getID()] + math.sqrt((2 * math.log(total_count)) / float(arm_counts[a.getID()])) for a in possibleActions]
        #print ucb_values
        action = possibleActions[rargmax(ucb_values)]
        return action


    def updatePolicy(self, content, chosen_arm, reward):
        try:
            self.counts[chosen_arm.getID()] = self.counts[chosen_arm.getID()] + 1
            n = self.counts[chosen_arm.getID()]

            value = self.values[chosen_arm.getID()]
            new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
            self.values[chosen_arm.getID()] = new_value
            return
        except:
            return


