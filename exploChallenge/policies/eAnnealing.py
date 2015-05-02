__author__ = 'bixlermike'

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

class eAnnealing(ContextualBanditPolicy):


    def __init__(self):
        self.epsilon = {}
        self.trials = 1
        self.counts = {}
        self.values = {}

    def getEpsilon(self):
        return self.epsilon


    def getActionToPerform(self, visitor, possibleActions):
        for action in possibleActions:
            if action.getID() not in self.counts:
                # Avoid divide by zero error
                self.counts[action.getID()] = 0.001
                self.values[action.getID()] = 0.001

        arm_values = [self.values[a.getID()] for a in possibleActions]
        # Adjust probability of exploration lower as algorithm progresses
        self.epsilon = 1 / math.log(self.trials + 0.0000001)
        self.trials += 1
        # Exploit
        if rn.random() > self.epsilon:
            action = possibleActions[rargmax(arm_values)]
            return action
        # Explore
        else:
            randomIndex = rn.randint(0, len(possibleActions) - 1)
            return possibleActions[randomIndex]

    def updatePolicy(self, content, chosen_arm, reward):
        try:
            self.counts[chosen_arm.getID()] += 1
            n = self.counts[chosen_arm.getID()]
            value = self.values[chosen_arm.getID()]

            # Calculate revised AER of this arm
            new_value = ((n-1) / float(n)) * value + (1/float(n)) * reward
            self.values[chosen_arm.getID()] = new_value

            return
        except:
            return


