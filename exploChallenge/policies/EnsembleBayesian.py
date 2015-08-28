__author__ = 'bixlermike'

import numpy as np
import random
from scipy.stats import beta
import time

rand = np.random.rand
from exploChallenge.policies.ContextualBanditPolicy import ContextualBanditPolicy
from exploChallenge.policies.RidgeRegressor import RidgeRegressor

from exploChallenge.policies.BinomialUCI import BinomialUCI
from exploChallenge.policies.MostCTR import MostCTR
from exploChallenge.policies.Softmax import Softmax
from exploChallenge.policies.UCB1 import UCB1
from exploChallenge.policies.LinUCB import LinUCB
from exploChallenge.policies.NaiveBayesContextual import NaiveBayesContextual
from exploChallenge.policies.SoftmaxContextual import SoftmaxContextual

#output_file = open("banditPolicyProportionsVsEvalNumber.txt", "a+")
output_file = open("testPolicyCountsVsEvalNumber.txt", "a+")

def rargmax(x):
    m = np.amax(x)
    indices = np.nonzero(x == m)[0]
    return random.choice(indices)

class EnsembleBayesian(ContextualBanditPolicy):


    def __init__(self, regressor):
        self.regressor = regressor
        self.prior_alpha = 1.0
        self.prior_beta = 1.0
        # Create an object from each class to use for ensemble model
        self.policy_one = BinomialUCI()
        self.policy_two = MostCTR()
        self.policy_three = Softmax(0.1)
        self.policy_four = UCB1()
        self.policy_five = LinUCB(0.1)
        self.policy_six = NaiveBayesContextual()
        self.policy_seven = SoftmaxContextual(0.1, RidgeRegressor(np.eye(136), np.zeros(136)))
        self.policies = [self.policy_one, self.policy_two, self.policy_three, self.policy_four, self.policy_five,
                         self.policy_six, self.policy_seven]
        self.policy_nicknames = ["BinomialUCI", "MostCTR", "Softmax0.1", "UCB1", "LinUCB(0.1)", "NaiveBayesContextual",
                                 "SoftmaxContextual0.1"]
        self.policy_counts = {}
        self.policy_successes = {}
        self.policy_runtimes = {}
        self.start_time = 0
        self.end_time = 0
        self.updates = 0
        self.policy_runtime_to_count_ratios = {}
        for i in self.policies:
            self.policy_runtimes[str(i)] = 0
            self.policy_counts[str(i)] = 0
            self.policy_successes[str(i)] = 0
            self.policy_runtime_to_count_ratios[str(i)] = 0.01
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
        self.start_time = time.clock()
        return self.chosen_policy.getActionToPerform(visitor, possibleActions)

#@Override
    def updatePolicy(self, content, chosen_arm, reward, *possibleActions):
        self.end_time = time.clock()
        elapsed_time = self.end_time - self.start_time
        #print "Elapsed time: " + str(elapsed_time)
        self.policy_runtimes[str(self.chosen_policy)] += elapsed_time
        self.policy_counts[str(self.chosen_policy)] += 1
        self.policy_runtime_to_count_ratios[str(self.chosen_policy)] = self.policy_runtimes[str(self.chosen_policy)] \
                                                                     /self.policy_counts[str(self.chosen_policy)]
        self.updates += 1
        for p in self.policies:
            try:
                #print "Updating policy: " + str(p)
                p.updatePolicy(content, chosen_arm, reward)
            except:
                #print "Error updating: " + str(self.chosen_policy) + " for chosen arm " + str(chosen_arm) + "."
                pass

        #print "Counts for: " + str(self.chosen_policy) + " is " + str(self.policy_counts[str(self.chosen_policy)])
        if reward is True:
            self.policy_successes[str(self.chosen_policy)] += 1

        if (self.updates % 100 == 0):
            for i in self.policies:
                print str("EnsembleBayesianUpdateAll") + "," + str(self.policy_nicknames[self.policies.index(i)]) + "," + \
                      str(self.updates) + "," + str(float(self.policy_counts[str(i)])/self.updates)
                output_file.write(str("EnsembleBayesianUpdateAll") + "," + str(self.policy_nicknames[self.policies.index(i)]) + ","
                                  + str(self.updates) + "," + str(float(self.policy_counts[str(i)])/self.updates) + "\n")