__author__ = 'dai.shi'
__revised__ = 'bixlermike'
import random as rn
import numpy as np

from exploChallenge.policies.ContextualBanditPolicy import ContextualBanditPolicy

def rargmax(x):
    m = np.amax(x)
    indices = np.nonzero(x == m)[0]
    return rn.choice(indices)

class MostCTR(ContextualBanditPolicy):

    def __init__(self):
        self.counts = {}
        self.rates = {}
        self.total_count = 0
        return

    def getActionToPerform(self, visitor,possibleActions):

        for action in possibleActions:
            if action.getID() not in self.counts:
                self.rates[action.getID()] = 1.0
                self.counts[action.getID()] = 1.0

        click_rates = [self.rates[a.getID()] for a in possibleActions]

        action = possibleActions[rargmax(click_rates)]
        return action

    def updatePolicy(self, content, chosen_arm, reward):
        try:
            self.counts[chosen_arm.getID()] += 1
            n = self.counts[chosen_arm.getID()]
            rate = self.rates[chosen_arm.getID()]
            new_rate = ((n-1) / float(n)) * rate + (1/float(n)) * reward
            #new_rate_2 = rate + (reward - rate) / (n + 1)
            #print "New rate: " + str(new_rate)
            #print "Old rate: " + str(new_rate_2)
            self.total_count += 1
            # if self.total_count % 100 == 0:
            #     print "Delta: " + str(new_rate_2-new_rate)
            self.rates[chosen_arm.getID()] = new_rate
        except:
            pass
        return