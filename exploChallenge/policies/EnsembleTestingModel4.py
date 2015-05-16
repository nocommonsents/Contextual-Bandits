__author__ = 'bixlermike'

import numpy as np
import random
import re
from scipy.stats import beta

from exploChallenge.policies.ContextualBanditPolicy import ContextualBanditPolicy
from exploChallenge.policies.NaiveBayesContextual import NaiveBayesContextual
from exploChallenge.policies.MostCTR import MostCTR
from exploChallenge.policies.eAnnealing import eAnnealing
from exploChallenge.policies.RidgeRegressor import RidgeRegressor

def rargmax(x):
    m = np.amax(x)
    indices = np.nonzero(x == m)[0]
    return random.choice(indices)

class EnsembleTestingModel4(ContextualBanditPolicy):


    def __init__(self, regressor):
        # Create an object from each class to use for ensemble model
        self.regressor = regressor
        self.prior_alpha = 1.0
        self.prior_beta = 1.0
        self.policy_one = eAnnealing()
        self.policy_two = MostCTR()
        self.policy_three = NaiveBayesContextual()
        self.policies = [self.policy_one, self.policy_two, self.policy_three]
        #self.policy_one_count = 0
        #self.policy_two_count = 0
        #self.policy_three_count = 0
        #self.policy_counts = [self.policy_one_count, self.policy_two_count, self.policy_three_count]
        self.policy_counts = {}
        #self.policy_one_successes = 0
        #self.policy_two_successes = 0
        #self.policy_three_successes = 0
        self.policy_successes = {}
        self.chosen_policy = None

    #@Override
    def getActionToPerform(self, visitor, possibleActions):
        sampled_theta = []
        dist = {}
        #Construct beta distribution for posterior

        for i in self.policies:
            if i not in self.policy_counts:
                self.policy_counts[str(i)] = 0
                self.policy_successes[str(i)] = 0
            #print i
            dist = beta(self.prior_alpha+self.policy_successes[str(i)],
                        self.prior_beta+self.policy_counts[str(i)]-self.policy_successes[str(i)])

            #Draw sample from beta distribution
            sampled_theta += [dist.rvs()]
            #print sampled_theta

        #print "Best index: " + str(sampled_theta.index(max(sampled_theta)))
        # Return the index of the sample with the largest value
        if (sampled_theta.index(max(sampled_theta)) == 0):
            self.chosen_policy =  str(self.policies[0])
        elif (sampled_theta.index(max(sampled_theta)) == 1):
            self.chosen_policy = str(self.policies[1])
        elif (sampled_theta.index(max(sampled_theta)) == 2):
            self.chosen_policy = str(self.policies[2])
        #print "Policy chosen was " + str(self.chosen_policy)

        if (re.match('<exploChallenge\.policies\.eAnnealing',self.chosen_policy)):
            return self.policy_one.getActionToPerform(visitor, possibleActions)
        elif (re.match('<exploChallenge\.policies\.MostCTR',self.chosen_policy)):
            return self.policy_two.getActionToPerform(visitor, possibleActions)
        elif (re.match('<exploChallenge\.policies\.Naive',self.chosen_policy)):
            return self.policy_three.getActionToPerform(visitor, possibleActions)
        else:
            print "Error in getActionToPerform!"
            return

#@Override
    def updatePolicy(self, content, chosen_arm, reward):

        try:
            self.policy_one.updatePolicy(content, chosen_arm, reward)
        except:
            pass
        try:
            self.policy_two.updatePolicy(content, chosen_arm, reward)
        except:
            pass
        try:
            self.policy_three.updatePolicy(content, chosen_arm, reward)
        except:
            pass

        self.policy_counts[self.chosen_policy] += 1
        if reward is True:
           self.policy_successes[str(self.chosen_policy)] += 1

        return