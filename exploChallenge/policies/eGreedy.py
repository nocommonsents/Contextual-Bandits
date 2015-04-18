__original_authors__ = 'dai.shi', 'John White'
__revised__ = 'bixlermike'

# Final verification 8 Apr 2015

import math
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


    def getActionToPerform(self, visitor, possibleActions):

        for action in possibleActions:
            if action.getID() not in self.counts:
                # Avoid divide by zero error, but learns more quickly than counts, values of 1.0 to start
                self.counts[action.getID()] = 0.001
                self.values[action.getID()] = 0.001

        arm_values = [self.values[a.getID()] for a in possibleActions]

        # Exploit with probability 1-epsilon
        if rn.random() > self.epsilon:
            #print str(rargmax(arm_values)) + " " + str(possibleActions[rargmax(arm_values)]) + "\n"
            return possibleActions[rargmax(arm_values)]
        # Explore with probability epsilon
        else:
            randomIndex = rn.randint(0, len(possibleActions) - 1)
            return possibleActions[randomIndex]

    def updatePolicy(self, content, chosen_arm, reward):
        self.counts[chosen_arm.getID()] += 1
        n = self.counts[chosen_arm.getID()]
        value = self.values[chosen_arm.getID()]
        # Calculate revised AER of this arm
        new_value = ((n-1) / float(n)) * value + (1/float(n)) * reward
        self.values[chosen_arm.getID()] = new_value

        return


