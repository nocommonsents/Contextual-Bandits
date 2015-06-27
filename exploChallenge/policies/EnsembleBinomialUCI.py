__author__ = 'bixlermike'
# UCI from https://www.ki.tu-berlin.de/fileadmin/fg135/publikationen/Seiler_2013_MLR.pdf, page 8
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

class EnsembleBinomialUCI(ContextualBanditPolicy):


    def __init__(self, regressor):
        # Create an object from each class to use for ensemble model
        self.regressor = regressor
        self.z_value = 1.96
        self.policy_one = MostCTR()
        self.policy_two = NaiveBayesContextual()
        #self.policy_three = eGreedyContextual(0.1, RidgeRegressor(np.eye(136), np.zeros(136)))
        self.policies = [self.policy_one, self.policy_two]
        self.policy_counts = {}
        self.policy_successes = {}
        self.policy_ucis = {}
        
        self.num_trials = 0
        self.chosen_policy = None

    #@Override
    def getActionToPerform(self, visitor, possibleActions):

        for i in self.policies:
            if str(i) not in self.policy_counts:
                self.policy_counts[str(i)] = 1.0
                self.policy_successes[str(i)] = 1.0
                self.policy_ucis[str(i)] = 1.0

        self.num_trials += 1

        #print self.policy_ucis
        current_uci_values = [self.policy_ucis[str(a)] for a in self.policies]
        #print current_uci_values
        self.chosen_policy = str(self.policies[rargmax(current_uci_values)])

        #print "Policy chosen was " + str(self.chosen_policy) + "\n"

        if (re.match('<exploChallenge\.policies\.MostCTR',self.chosen_policy)):
            return self.policy_one.getActionToPerform(visitor, possibleActions)
        elif (re.match('<exploChallenge\.policies\.NaiveBayes',self.chosen_policy)):
            return self.policy_two.getActionToPerform(visitor, possibleActions)
        else:
            print "Error in getActionToPerform!"
            return

#@Override
    def updatePolicy(self, content, chosen_arm, reward, *possibleActions):

        try:
            self.policy_one.updatePolicy(content, chosen_arm, reward)
        except:
            pass
        try:
            self.policy_two.updatePolicy(content, chosen_arm, reward)
        except:
            pass

        #print "Chosen policy is: " + str(self.chosen_policy)
        self.policy_counts[str(self.chosen_policy)] += 1

        if reward is True:
            self.policy_successes[str(self.chosen_policy)] += 1
        term_one = (self.policy_successes[str(self.chosen_policy)]/self.policy_counts[str(self.chosen_policy)])
        term_two = ((self.z_value * self.z_value) / (2*self.num_trials))
        term_three = (self.z_value / math.sqrt(self.policy_counts[str(self.chosen_policy)]))
        radicand = term_one * (1-term_one) + (self.z_value * self.z_value)/(4*self.policy_counts[str(self.chosen_policy)])
        numerator = term_one + term_two + term_three * math.sqrt(radicand)
        denominator = (1 + ((self.z_value * self.z_value)/self.policy_counts[str(self.chosen_policy)]))

        new_value = numerator / denominator
        self.policy_ucis[str(self.chosen_policy)] = new_value
        #print "After update UCIs are: " + str(self.policy_ucis) + "\n"

        return


