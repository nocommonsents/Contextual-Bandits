__author__ = 'bixlermike'

# Final verification 1 Jun 2015

import math
import numpy as np
import random
import re

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


def categorical_draw(probs):
    z = random.random()
    cum_prob = 0.0
    for p in probs:
        prob = probs[p]
        cum_prob += prob
        if cum_prob > z:
            return p
    return probs.iterkeys().next()

class EnsembleSoftmaxUpdateAllModel(ContextualBanditPolicy):

    def __init__(self, temp):
        self.temperature = temp
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
        self.policy_runtimes = {}
        self.policy_counts = {}
        self.policy_scores = {}
        self.policy_AER_to_runtime_ratios = {}
        self.start_time = 0
        self.end_time = 0
        self.total_updates = 0
        self.trials = 0
        for i in self.policies:
            self.policy_runtimes[str(i)] = 0
            self.policy_counts[str(i)] = 0
            self.policy_scores[str(i)] = 0
            self.policy_AER_to_runtime_ratios[str(i)] = 0
        self.chosen_policy = None

    def getTemp(self):
        return self.temperature

    #@Override
    def getActionToPerform(self, visitor, possibleActions):
        self.trials += 1
        policy_probs = {}

        policy_scores = [math.exp(self.policy_scores[str(a)]/self.temperature) for a in self.policy_scores]
        #print "Policy scores: " + str(policy_scores)
        z = sum(policy_scores)
        #print z

        # Calculate the probability that each arm will be selected
        for v in self.policies:
            policy_probs[v] = math.exp(self.policy_scores[str(v)] / self.temperature) / z

        # Generate random number and see which bin it falls into to select arm
        self.chosen_policy = categorical_draw(policy_probs)
        #print self.chosen_policy
        #print "Chosen policy: " + str(self.chosen_policy)
        return self.chosen_policy.getActionToPerform(visitor, possibleActions)

    #@Override
    def updatePolicy(self, content, chosen_arm, reward, *possibleActions):
        #print "Updating policy " + str(self.chosen_policy)
        for p in self.policies:
            try:
                #print "Updating policy: " + str(p)
                p.updatePolicy(content, chosen_arm, reward)
            except:
                print "Error updating: " + str(self.chosen_policy) + " for chosen arm " + str(chosen_arm) + "."
                pass

        self.policy_counts[str(self.chosen_policy)] += 1

        old_value = self.policy_scores[str(self.chosen_policy)]
        n = self.policy_counts[str(self.chosen_policy)]
        new_value = ((n - 1) / float(n)) * old_value + reward * (1 / float(n))
        #print "Old value: " + str(old_value)
        #print "New value:" + str(new_value)
        self.policy_scores[str(self.chosen_policy)] = new_value


        if (self.trials % 100 == 0):
            #print "Scores are: " + str(self.policy_scores)
            print "Counts are: " + str(self.policy_counts)
        return

