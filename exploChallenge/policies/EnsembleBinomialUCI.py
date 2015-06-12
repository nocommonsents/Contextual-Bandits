__author__ = 'bixlermike'
# UCI from https://www.ki.tu-berlin.de/fileadmin/fg135/publikationen/Seiler_2013_MLR.pdf, page 8
# Using 95% confidence interval, 2-tailed -> z-value = 1.96
import math
import numpy as np
import random
import re

from exploChallenge.policies.ContextualBanditPolicy import ContextualBanditPolicy
from exploChallenge.policies.NaiveBayesContextual import NaiveBayesContextual
from exploChallenge.policies.MostCTR import MostCTR
from exploChallenge.policies.eAnnealing import eAnnealing
from exploChallenge.policies.RidgeRegressor import RidgeRegressor

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
        self.policies = [self.policy_one, self.policy_two]
        self.policy_one_count = 1.0
        self.policy_two_count = 1.0
        self.policy_counts = [self.policy_one_count, self.policy_two_count]
        self.policy_one_successes = 1.0
        self.policy_two_successes = 1.0
        self.policy_successes = [self.policy_one_successes, self.policy_two_successes]
        self.policy_one_uci = 0
        self.policy_two_uci = 0
        self.policy_ucis = [self.policy_one_uci, self.policy_two_uci]
        
        self.num_trials = 0

        self.chosen_policy = None

    #@Override
    def getActionToPerform(self, visitor, possibleActions):

        self.num_trials += 1

        max_uci_index = rargmax(self.policy_ucis)
        #print max_uci_index
        self.chosen_policy = str(self.policies[max_uci_index])
        #print "Policy chosen was " + str(self.chosen_policy)

        if (re.match('<exploChallenge\.policies\.MostCTR',self.chosen_policy)):
            return self.policy_one.getActionToPerform(visitor, possibleActions)
        elif (re.match('<exploChallenge\.policies\.Naive',self.chosen_policy)):
            return self.policy_two.getActionToPerform(visitor, possibleActions)
        else:
            print "Error in getActionToPerform!"
            return

#@Override
    def updatePolicy(self, content, chosen_arm, reward, *possibleActions):

        if (re.match('<exploChallenge\.policies\.MostCTR',self.chosen_policy)):
            self.policy_one_count += 1
            if reward is True:
                self.policy_one_successes += 1
            term_one = (self.policy_one_successes/self.policy_one_count)
            term_two = ((self.z_value * self.z_value) / (2*self.num_trials))
            term_three = (self.z_value / math.sqrt(self.policy_one_count))
            radicand = term_one * (1-term_one) + (self.z_value * self.z_value)/(4*self.policy_one_count)
            numerator = term_one + term_two + term_three * math.sqrt(radicand)
            denominator = (1 + ((self.z_value * self.z_value)/self.policy_one_count))
            self.policy_one_uci = numerator / denominator
        elif (re.match('<exploChallenge\.policies\.NaiveBayes',self.chosen_policy)):
            self.policy_two_count += 1
            if reward is True:
                self.policy_two_successes += 1
            term_one = (self.policy_two_successes/self.policy_two_count)
            term_two = ((self.z_value * self.z_value) / (2*self.num_trials))
            term_three = (self.z_value / math.sqrt(self.policy_two_count))
            radicand = term_one * (1-term_one) + (self.z_value * self.z_value)/(4*self.policy_two_count)
            numerator = term_one + term_two + term_three * math.sqrt(radicand)
            denominator = (1 + ((self.z_value * self.z_value)/self.policy_two_count))
            self.policy_two_uci = numerator / denominator
        else:
            print "Error with updatePolicy in EnsembleEAnnealingUpdateAll!"


        try:
            self.policy_one.updatePolicy(content, chosen_arm, reward)
        except:
            pass
        try:
            self.policy_two.updatePolicy(content, chosen_arm, reward)
        except:
            pass

        return