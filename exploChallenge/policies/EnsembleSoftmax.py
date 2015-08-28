__author__ = 'bixlermike'

# Final verification 1 Jun 2015

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

output_file = open("banditPolicyProportionsVsEvalNumber.txt", "a+")
#output_file = open("testPolicyCountsVsEvalNumber.txt", "a+")

def categorical_draw(probs):
    z = random.random()
    cum_prob = 0.0
    for p in probs:
        prob = probs[p]
        cum_prob += prob
        if cum_prob > z:
            return p
    return probs.iterkeys().next()

class EnsembleSoftmax(ContextualBanditPolicy):

    def __init__(self, temp):
        self.temperature = temp
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
        self.policy_runtimes = {}
        self.policy_counts = {}
        self.policy_scores = {}
        self.policy_runtime_to_count_ratios = {}
        self.start_time = 0
        self.end_time = 0
        self.updates = 7.0
        self.trials = 7.0
        for i in self.policies:
            self.policy_runtimes[str(i)] = 0
            self.policy_counts[str(i)] = 1.0
            self.policy_scores[str(i)] = 1.0
            self.policy_runtime_to_count_ratios[str(i)] = 0.01
        self.chosen_policy = None

    def getTemp(self):
        return self.temperature

    #@Override
    def getActionToPerform(self, visitor, possibleActions):
        self.trials += 1
        policy_probs = {}
        adjusted_policy_probs = {}

        #print self.policy_runtime_to_count_ratios
        policy_scores = [math.exp(self.policy_scores[str(a)]/self.temperature) for a in self.policy_scores]
        adjusted_policy_scores = [math.exp(self.policy_scores[str(a)]/self.temperature/
                                           math.exp(self.policy_runtime_to_count_ratios[str(a)])) for a in self.policy_scores]
        #print "Policy scores: " + str(policy_scores)
        z = sum(policy_scores)
        #print z

        # Calculate the probability that each arm will be selected
        for v in self.policies:
            policy_probs[v] = math.exp(self.policy_scores[str(v)] / self.temperature) / z
            #adjusted_policy_probs[v] = math.exp(self.policy_scores[str(v)] / self.temperature/\
            #                           math.exp(self.policy_runtime_to_count_ratios[str(v)]) / z)
            #print policy_probs

        # Generate random number and see which bin it falls into to select arm
        self.chosen_policy = categorical_draw(policy_probs)
        #print self.chosen_policy
        #print "Chosen policy: " + str(self.chosen_policy)
        #self.start_time = time.clock()
        return self.chosen_policy.getActionToPerform(visitor, possibleActions)

    #@Override
    def updatePolicy(self, content, chosen_arm, reward, *possibleActions):
        #print "Updating policy " + str(self.chosen_policy)
        #self.end_time = time.clock()
        #elapsed_time = self.end_time - self.start_time
        #print "Elapsed time: " + str(elapsed_time)
        #self.policy_runtimes[str(self.chosen_policy)] += elapsed_time
        self.policy_counts[str(self.chosen_policy)] += 1
        #self.policy_runtime_to_count_ratios[str(self.chosen_policy)] = self.policy_runtimes[str(self.chosen_policy)] \
        #                                                            /self.policy_counts[str(self.chosen_policy)]
        self.updates += 1
        for p in self.policies:
            try:
                #print "Updating policy: " + str(p)
                p.updatePolicy(content, chosen_arm, reward)
            except:
                #print "Error updating: " + str(self.chosen_policy) + " for chosen arm " + str(chosen_arm) + "."
                pass

        old_value = self.policy_scores[str(self.chosen_policy)]
        n = self.policy_counts[str(self.chosen_policy)]
        new_value = ((n - 1) / float(n)) * old_value + reward * (1 / float(n))
        #print "Old value: " + str(old_value)
        #print "New value:" + str(new_value)
        self.policy_scores[str(self.chosen_policy)] = new_value


        # if (self.trials % 100 == 0):
        #     #print "Scores are: " + str(self.policy_scores)
        #     print "Counts are: " + str(self.policy_counts)
        if (self.updates % 100 == 0):
            for i in self.policies:
                print str("EnsembleSoftmax0.01UpdateAll") + "," + str(self.policy_nicknames[self.policies.index(i)]) + "," + \
                      str(self.updates) + "," + str(float(self.policy_counts[str(i)])/self.updates)
                output_file.write(str("EnsembleSoftmax0.01UpdateAll") + "," + str(self.policy_nicknames[self.policies.index(i)]) + ","
                                  + str(self.updates) + "," + str(float(self.policy_counts[str(i)])/self.updates) + "\n")

