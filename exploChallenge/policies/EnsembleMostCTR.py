__author__ = 'bixlermike'

import math
import numpy as np
import random
import time

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

class EnsembleMostCTR(ContextualBanditPolicy):


    def __init__(self):
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
        self.policy_runtime_to_count_ratios = {}
        self.start_time = 0
        self.end_time = 0
        self.trials = 7.0
        self.updates = 7.0
        for i in self.policies:
            self.policy_runtimes[str(i)] = 0
            self.policy_counts[str(i)] = 1.0
            self.policy_successes[str(i)] = 1.0
            self.policy_runtime_to_count_ratios[str(i)] = 0.01
        self.chosen_policy = None

    #@Override
    def getActionToPerform(self, visitor, possibleActions):

        self.trials += 1
        policy_click_rates = [(self.policy_successes[str(a)]/self.policy_counts[str(a)]) for a in self.policies]
        #policy_adjusted_click_rates = [(self.policy_successes[str(a)]/self.policy_counts[str(a)]/
        #                                self.policy_runtime_to_count_ratios[str(a)]) for a in self.policies]
        self.chosen_policy = self.policies[rargmax(policy_click_rates)]
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
        #print "Updating policy " + str(self.chosen_policy)
        for p in self.policies:
            try:
                #print "Updating policy: " + str(p)
                p.updatePolicy(content, chosen_arm, reward)
            except:
                #print "Error updating: " + str(self.chosen_policy) + " for chosen arm " + str(chosen_arm) + "."
                pass

        #print "Chosen policy is: " + str(self.chosen_policy)

        if reward is True:
            self.policy_successes[str(self.chosen_policy)] += 1

        if (self.updates % 100 == 0):
            for i in self.policies:
                #print str(self.policy_nicknames[self.policies.index(i)]) + " " + str(self.policy_runtime_to_count_ratios[str(i)])
                print str("EnsembleMostCTRUpdateAll") + "," + str(self.policy_nicknames[self.policies.index(i)]) + "," + \
                      str(self.updates) + "," + str(float(self.policy_counts[str(i)])/self.updates)
                output_file.write(str("EnsembleMostCTRUpdateAll") + "," + str(self.policy_nicknames[self.policies.index(i)]) + ","
                                  + str(self.updates) + "," + str(float(self.policy_counts[str(i)])/self.updates) + "\n")

