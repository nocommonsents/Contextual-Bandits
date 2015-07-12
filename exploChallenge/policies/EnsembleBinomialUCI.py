__author__ = 'bixlermike'
# UCI from https://www.ki.tu-berlin.de/fileadmin/fg135/publikationen/Seiler_2013_MLR.pdf, page 8
# Using 95% confidence interval, 2-tailed -> z-value = 1.96
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

#output_file = open("banditPolicyCountsVsEvalNumber.txt", "a+")
output_file = open("testPolicyCountsVsEvalNumber.txt", "a+")

def rargmax(x):
    m = np.amax(x)
    indices = np.nonzero(x == m)[0]
    return random.choice(indices)

class EnsembleBinomialUCI(ContextualBanditPolicy):


    def __init__(self, regressor):
        self.regressor = regressor
        self.z_value = 1.96
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
        self.policy_ucis = {}
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
            self.policy_ucis[str(i)] = 1.0
            self.policy_runtime_to_count_ratios[str(i)] = 0
        self.chosen_policy = None

    #@Override
    def getActionToPerform(self, visitor, possibleActions):

        self.trials += 1

        #print self.policy_ucis
        current_uci_values = [self.policy_ucis[str(a)] for a in self.policies]
        #print current_uci_values
        self.chosen_policy = self.policies[rargmax(current_uci_values)]
        #print "Chosen policy: " + str(self.chosen_policy) + "\n"
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
        term_one = (self.policy_successes[str(self.chosen_policy)]/self.policy_counts[str(self.chosen_policy)])
        term_two = ((self.z_value * self.z_value) / (2*self.trials))
        term_three = (self.z_value / math.sqrt(self.policy_counts[str(self.chosen_policy)]))
        radicand = term_one * (1-term_one) + (self.z_value * self.z_value)/(4*self.policy_counts[str(self.chosen_policy)])
        numerator = term_one + term_two + term_three * math.sqrt(radicand)
        denominator = (1 + ((self.z_value * self.z_value)/self.policy_counts[str(self.chosen_policy)]))

        new_value = numerator / denominator
        self.policy_ucis[str(self.chosen_policy)] = new_value
        #print "After update UCIs are: " + str(self.policy_ucis) + "\n"

        if (self.updates % 100 == 0):
            for i in self.policies:
                print str("EnsembleBinomialUCIUpdateAll") + "," + str(self.policy_nicknames[self.policies.index(i)]) + "," + \
                      str(self.updates) + "," + str(float(self.policy_counts[str(i)])/self.updates)
                output_file.write(str("EnsembleBinomialUCIUpdateAll") + "," + str(self.policy_nicknames[self.policies.index(i)]) + ","
                                  + str(self.updates) + "," + str(float(self.policy_counts[str(i)])/self.updates))

