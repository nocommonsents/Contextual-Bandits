__author__ = 'bixlermike'

# Final verification 1 Jun 2015

import math
import numpy as np
import random as rn
import time

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
    return rn.choice(indices)

class EnsembleEAnnealingUpdateAllModel(ContextualBanditPolicy):


    def __init__(self):
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
        self.policy_scores = {}
        self.policy_counts = {}
        self.policy_runtimes = {}
        self.policy_AER_to_runtime_ratios = {}
        self.start_time = 0
        self.end_time = 0
        self.total_updates = 0
        self.trials = 1
        for i in self.policies:
            self.policy_counts[str(i)] = 1.0
            self.policy_scores[str(i)] = 1.0
            self.policy_runtimes[str(i)] = 0
            self.policy_AER_to_runtime_ratios[str(i)] = 0
        self.chosen_policy = None

    #@Override
    def getActionToPerform(self, visitor, possibleActions):
        random_number = rn.random()
        self.epsilon = 1 / math.log(self.trials + 0.0000001)
        self.trials += 1
        #print "Epsilon is: " + str(self.epsilon)

        if random_number > self.epsilon:
            #print "Exploiting!"
            policy_values = [self.policy_scores[str(a)] for a in self.policies]
            self.chosen_policy = self.policies[rargmax(policy_values)]
            #print "Chosen policy: " + str(self.chosen_policy)
            self.start_time = time.clock()
            return self.chosen_policy.getActionToPerform(visitor, possibleActions)
        else:
            #print "Exploring"
            self.chosen_policy =  rn.choice(self.policies)
            #print "Chosen policy: " + str(self.chosen_policy)
            self.start_time = time.clock()
            return self.chosen_policy.getActionToPerform(visitor, possibleActions)


    #@Override
    def updatePolicy(self, content, chosen_arm, reward, *possibleActions):
        self.end_time = time.clock()
        elapsed_time = self.end_time - self.start_time
        #print "Elapsed time: " + str(elapsed_time)
        self.policy_runtimes[str(self.chosen_policy)] += elapsed_time
        self.policy_counts[str(self.chosen_policy)] += 1
        self.policy_AER_to_runtime_ratios[str(self.chosen_policy)] = self.policy_runtimes[str(self.chosen_policy)] \
                                                                     /self.policy_counts[str(self.chosen_policy)]
        self.total_updates += 1
        #print self.policy_scores
        #print "Updating policy " + str(self.chosen_policy)
        for p in self.policies:
            try:
                #print "Updating policy: " + str(p)
                p.updatePolicy(content, chosen_arm, reward)
            except:
                #print "Error updating: " + str(self.chosen_policy) + " for chosen arm " + str(chosen_arm) + "."
                pass

        new_value = ((self.policy_counts[str(self.chosen_policy)] - 1) / float(self.policy_counts[str(self.chosen_policy)])) * \
                    self.policy_scores[str(self.chosen_policy)] + reward * (1 / float(self.policy_counts[str(self.chosen_policy)]))
        self.policy_scores[str(self.chosen_policy)] = new_value
        #print "Scores are: " + str(self.policy_scores)
        #print "Counts are: " + str(self.policy_counts)
        if (self.total_updates % 500 == 0):
            print "All average AER to runtime ratios: " + str(self.policy_AER_to_runtime_ratios)