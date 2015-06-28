__author__ = 'bixlermike'

import numpy as np
import random
import re
from scipy.stats import beta

rand = np.random.rand
from exploChallenge.policies.ContextualBanditPolicy import ContextualBanditPolicy
from exploChallenge.policies.RidgeRegressor import RidgeRegressor

from exploChallenge.policies.MostCTR import MostCTR
from exploChallenge.policies.NaiveBayesContextual import NaiveBayesContextual
from exploChallenge.policies.LinUCB import LinUCB
from exploChallenge.policies.eGreedyContextual import eGreedyContextual
from exploChallenge.policies.SoftmaxContextual import SoftmaxContextual
def rargmax(x):
    m = np.amax(x)
    indices = np.nonzero(x == m)[0]
    return random.choice(indices)

class EnsembleBayesianUpdateAllModel(ContextualBanditPolicy):


    def __init__(self, regressor):
        # Create an object from each class to use for ensemble model
        self.regressor = regressor
        self.prior_alpha = 1.0
        self.prior_beta = 1.0
        self.policy_one = MostCTR()
        self.policy_two = NaiveBayesContextual()
        self.policy_three = eGreedyContextual(0.1, RidgeRegressor(np.eye(136), np.zeros(136)))
        #self.policy_four = LinUCB(0.1)
        #self.policy_five = SoftmaxContextual(0.1, RidgeRegressor(np.eye(136), np.zeros(136)))
        self.policies = [self.policy_one, self.policy_two, self.policy_three]

        self.policy_counts = {}
        self.policy_successes = {}
        self.chosen_policy = None

    #@Override
    def getActionToPerform(self, visitor, possibleActions):
        sampled_theta = []
        dist = {}
        #Construct beta distribution for posterior

        for i in self.policies:
            if str(i) not in self.policy_counts:
                self.policy_counts[str(i)] = 0
                self.policy_successes[str(i)] = 0

            # print str(i)
            # print self.policy_successes[str(i)]
            # print self.policy_counts[str(i)]
            # print "\n"
            dist = beta(self.prior_alpha+self.policy_successes[str(i)],
                        self.prior_beta+self.policy_counts[str(i)]-self.policy_successes[str(i)])
            #Draw sample from beta distribution
            sampled_theta += [dist.rvs()]

        #print str(self.policies) + " " + str(sampled_theta)
        #print "Best index: " + str(sampled_theta.index(max(sampled_theta)))
        # Return the policy corresponding to the index of the sample with the largest value
        self.chosen_policy = str(self.policies[(sampled_theta.index(max(sampled_theta)))])
        #print str(self.chosen_policy)  + "\n"

        if (re.match('<exploChallenge\.policies\.MostCTR',self.chosen_policy)):
            return self.policy_one.getActionToPerform(visitor, possibleActions)
        elif (re.match('<exploChallenge\.policies\.Naive',self.chosen_policy)):
            return self.policy_two.getActionToPerform(visitor, possibleActions)
        elif (re.match('<exploChallenge\.policies\.eGreedyContextual',self.chosen_policy)):
            return self.policy_three.getActionToPerform(visitor, possibleActions)
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
        try:
            self.policy_three.updatePolicy(content, chosen_arm, reward)
        except:
            pass
        self.policy_counts[str(self.chosen_policy)] += 1
        #print self.policy_counts
        if reward is True:
            self.policy_successes[str(self.chosen_policy)] += 1


        return