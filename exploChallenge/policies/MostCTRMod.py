__author__ = 'bixlermike'
# Influenced by Dai Shi's MostClicked model

import math
import random as rn
import numpy as np

from exploChallenge.policies.ContextualBanditPolicy import ContextualBanditPolicy

def rargmax(x):
    m = np.amax(x)
    indices = np.nonzero(x == m)[0]
    return rn.choice(indices)

class MostCTRMod(ContextualBanditPolicy):

    def __init__(self):
        self.counts = {}
        self.rates = {}
        self.scaled_click_rates = {}
        return

    def getCTRs(self, visitor,possibleActions):

        for action in possibleActions:
            if action.getID() not in self.counts:
                self.rates[action.getID()] = 1.0
                self.counts[action.getID()] = 1.0

        z = sum(math.exp(self.rates[v]) for v in self.rates)
        for ac in possibleActions:
            self.scaled_click_rates[ac.getID()] = [math.exp(self.rates[ac.getID()])/(z)]
        return self.scaled_click_rates

    def updatePolicy(self, content, chosen_arm, reward):
        self.counts[chosen_arm.getID()] += 1
        n = self.counts[chosen_arm.getID()]

        rate = self.rates[chosen_arm.getID()]
        new_rate = rate + (reward - rate) / (n + 1)
        self.rates[chosen_arm.getID()] = new_rate
        return