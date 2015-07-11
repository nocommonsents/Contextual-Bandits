__author__ = 'bixlermike'

import numpy as np
import random
import re
from scipy.stats import beta

rand = np.random.rand
from exploChallenge.policies.ContextualBanditPolicy import ContextualBanditPolicy
from exploChallenge.policies.RidgeRegressor import RidgeRegressor

from exploChallenge.policies.BinomialUCI import BinomialUCI
from exploChallenge.policies.MostRecent import MostRecent
from exploChallenge.policies.MostCTR import MostCTR
from exploChallenge.policies.Softmax import Softmax
from exploChallenge.policies.eAnnealing import eAnnealing
from exploChallenge.policies.UCB1 import UCB1
from exploChallenge.policies.LinUCB import LinUCB
from exploChallenge.policies.NaiveBayesContextual import NaiveBayesContextual
from exploChallenge.policies.SoftmaxContextual import SoftmaxContextual
from exploChallenge.policies.eAnnealingContextual import eAnnealingContextual
def rargmax(x):
    m = np.amax(x)
    indices = np.nonzero(x == m)[0]
    return random.choice(indices)

class EnsembleBayesianUpdateAllModel(ContextualBanditPolicy):


    def __init__(self, regressor):
        self.regressor = regressor
        self.prior_alpha = 1.0
        self.prior_beta = 1.0
        # Create an object from each class to use for ensemble model
        self.policy_one = BinomialUCI()
        self.policy_two = MostRecent()
        self.policy_three = MostCTR()
        self.policy_four = Softmax(0.01)
        self.policy_five = eAnnealing()
        self.policy_six = UCB1()
        self.policy_seven = LinUCB(0.1)
        self.policy_eight = NaiveBayesContextual()
        self.policy_nine = SoftmaxContextual(0.01, RidgeRegressor(np.eye(136), np.zeros(136)))
        self.policy_ten = eAnnealingContextual(RidgeRegressor(np.eye(136), np.zeros(136)))
        self.policies = [self.policy_one, self.policy_two, self.policy_three, self.policy_four, self.policy_five,
                         self.policy_six, self.policy_seven, self.policy_eight, self.policy_nine, self.policy_ten]
        self.chosen_policy = None

        self.policy_counts = {}
        self.policy_successes = {}
        self.policy_runtimes = {}
        self.start_time = 0
        self.end_time = 0
        self.total_updates = 0
        self.policy_AER_to_runtime_ratios = {}
        for i in self.policies:
            self.policy_runtimes[str(i)] = 0
            self.policy_counts[str(i)] = 0
            self.policy_successes[str(i)] = 0
            self.policy_AER_to_runtime_ratios[str(i)] = 0
        self.chosen_policy = None

    #@Override
    def getActionToPerform(self, visitor, possibleActions):
        sampled_theta = []
        dist = {}
        #Construct beta distribution for posterior

        for i in self.policies:
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
        self.chosen_policy = self.policies[(sampled_theta.index(max(sampled_theta)))]
        #print str(self.chosen_policy)  + "\n"
        return self.chosen_policy.getActionToPerform(visitor, possibleActions)

#@Override
    def updatePolicy(self, content, chosen_arm, reward, *possibleActions):
        for p in self.policies:
            try:
                #print "Updating policy: " + str(p)
                p.updatePolicy(content, chosen_arm, reward)
            except:
                #print "Error updating: " + str(self.chosen_policy) + " for chosen arm " + str(chosen_arm) + "."
                pass

        self.policy_counts[str(self.chosen_policy)] += 1
        #print "Counts for: " + str(self.chosen_policy) + " is " + str(self.policy_counts[str(self.chosen_policy)])
        if reward is True:
            self.policy_successes[str(self.chosen_policy)] += 1


        return