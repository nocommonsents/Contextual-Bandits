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

class eAnnealingMod(ContextualBanditPolicy):


    def __init__(self):
        self.epsilon = {}
        self.trials = 1
        self.counts = {}
        self.values = {}
        self.random_values = {}
        self.scaled_values = {}

    def getScaledValues(self, visitor, possibleActions):
        self.epsilon = 1 / math.log(self.trials + 0.0000001)
        self.trials += 1
        for action in possibleActions:
            if action.getID() not in self.counts:
                # Avoid divide by zero error
                self.counts[action.getID()] = 0.001
                self.values[action.getID()] = 0.001

        # Exploiting
        if rn.random() > self.epsilon:
            z = sum(math.exp(self.values[v]) for v in self.values)
            #print "Exploiting - z is equal to: " + str(z)

            for ac in possibleActions:
                self.scaled_values[ac.getID()] = [math.exp(self.values[ac.getID()])/z]
            #print "Exploiting scaled values are: " + str(self.scaled_values)
            return self.scaled_values

        # Exploring
        else:
            for article in possibleActions:
                self.random_values[article.getID()] = rn.random()
            z = sum(math.exp(self.random_values[v]) for v in self.values)
            #print "Exploring - z is equal to: " + str(z)

            for ac in possibleActions:
                 self.scaled_values[ac.getID()] = [math.exp(self.random_values[ac.getID()])/z]
            #print "Exploiting scaled values are: " + str(self.scaled_values)
            return self.scaled_values

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


