__author__ = 'bixlermike'

# BinomialUCI algoritm from https://www.ki.tu-berlin.de/fileadmin/fg135/publikationen/Seiler_2013_MLR.pdf, page 8
# Using 95% confidence interval, 2-tailed -> z-value = 1.96
import math
import numpy as np
import random
import re

from exploChallenge.policies.ContextualBanditPolicy import ContextualBanditPolicy
from exploChallenge.policies.RidgeRegressor import RidgeRegressor

from exploChallenge.policies.MostCTR import MostCTR
from exploChallenge.policies.NaiveBayesContextual import NaiveBayesContextual
from exploChallenge.policies.LinUCB import LinUCB
from exploChallenge.policies.eGreedyContextual import eGreedyContextual

def rargmax(x):
    m = np.amax(x)
    indices = np.nonzero(x == m)[0]
    return random.choice(indices)

class BinomialUCI(ContextualBanditPolicy):


    def __init__(self):
        # Create an object from each class to use for ensemble model
        self.z_value = 1.96
        self.counts = {}
        self.successes = {}
        self.ucis = {}
        self.num_trials = 0
        self.chosen_policy = None

    #@Override
    def getActionToPerform(self, visitor, possibleActions):
        for action in possibleActions:
            if action.getID() not in self.counts:
                #print "New article: " + str(action.getID())
                self.counts[action.getID()] = 1.0
                self.successes[action.getID()] = 1.0
                self.ucis[action.getID()] = 1.0

        self.num_trials += 1

        #print self.policy_ucis
        current_uci_values = [self.ucis[a.getID()] for a in possibleActions]
        #print current_uci_values
        return possibleActions[rargmax(current_uci_values)]

#@Override
    def updatePolicy(self, content, chosen_arm, reward, *possibleActions):

        self.counts[chosen_arm.getID()] += 1

        if reward is True:
            self.successes[chosen_arm.getID()] += 1
        term_one = (self.successes[chosen_arm.getID()]/self.counts[chosen_arm.getID()])
        term_two = ((self.z_value * self.z_value) / (2*self.num_trials))
        term_three = (self.z_value / math.sqrt(self.counts[chosen_arm.getID()]))
        radicand = term_one * (1-term_one) + (self.z_value * self.z_value)/(4*self.counts[chosen_arm.getID()])
        numerator = term_one + term_two + term_three * math.sqrt(radicand)
        denominator = (1 + ((self.z_value * self.z_value)/self.counts[chosen_arm.getID()]))

        new_value = numerator / denominator
        self.ucis[chosen_arm.getID()] = new_value
        #print "Chosen article is: " + str(chosen_arm.getID())
        #print "After update UCIs are: " + str(self.ucis) + "\n"

        return


