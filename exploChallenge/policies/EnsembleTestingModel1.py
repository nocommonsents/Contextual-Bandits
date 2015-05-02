__author__ = 'bixlermike'

import math
import numpy as np
import random
import re

from exploChallenge.policies.ContextualBanditPolicy import ContextualBanditPolicy
from exploChallenge.policies.eAnnealingContextualMod import eAnnealingContextualMod
from exploChallenge.policies.NaiveBayesContextualMod import NaiveBayesContextualMod
from exploChallenge.policies.MostCTRMod import MostCTRMod
from exploChallenge.policies.RidgeRegressor import RidgeRegressor

def rargmax(x):
    m = np.amax(x)
    indices = np.nonzero(x == m)[0]
    return random.choice(indices)

class EnsembleTestingModel1(ContextualBanditPolicy):


    def __init__(self, regressor):
        # Create an object from each class to use for ensemble model
        self.regressor = regressor
        self.policy_one = NaiveBayesContextualMod()
        self.policy_two = MostCTRMod()
        self.policies = [self.policy_one, self.policy_two]
        self.policy_one_count = 0
        self.policy_two_count = 0
        self.policy_counts = [self.policy_one_count, self.policy_two_count]
        self.policy_one_scores = {}
        self.policy_two_scores = {}
        self.total_scores = {}
        self.final_scaled_values = {}
        self.chosen_policy = None

    #@Override
    def getActionToPerform(self, visitor, possibleActions):

        policy_one_scaled_values = self.policy_one.getScaledIndices(visitor, possibleActions)
        #print policy_one_scaled_values
        policy_two_scaled_values =  self.policy_two.getScaledIndices(visitor, possibleActions)
        #print policy_two_scaled_values
        #print "\n"
        for act in possibleActions:
            self.policy_one_scores[act.getID()] = float(policy_one_scaled_values[act.getID()][0])
            self.policy_two_scores[act.getID()] = float(policy_two_scaled_values[act.getID()][0])

        policy_one_scores_list = [self.policy_one_scores[v.getID()] for v in possibleActions]
        policy_two_scores_list = [self.policy_two_scores[vi.getID()] for vi in possibleActions]

        max_policy_one_score = max(policy_one_scores_list)

        #print "Max policy one: " + str(max_policy_one_score) + " and max policy two: " + str(max_policy_two_score)
        #print "Delta is: " + str(delta)
        # If the policy one score beats the policy two score by at least 10%
        if (max_policy_one_score > max_policy_two_score and float(delta)/max_policy_two_score > 0.1):
           # print "Here1 " + str(max_policy_one_score) + " " + str(max_policy_two_score)
            return possibleActions[rargmax(policy_one_scores_list)]
        else:
            #print "Here3 "
            return possibleActions[rargmax(policy_three_scores_list)]

#@Override
    def updatePolicy(self, content, chosen_arm, reward):

        self.policy_one.updatePolicy(content, chosen_arm, reward)
        self.policy_two.updatePolicy(content, chosen_arm, reward)

        return