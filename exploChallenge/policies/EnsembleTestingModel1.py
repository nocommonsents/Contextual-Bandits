__author__ = 'bixlermike'

import numpy as np
import random

from exploChallenge.policies.ContextualBanditPolicy import ContextualBanditPolicy
from exploChallenge.policies.NaiveBayesContextualMod import NaiveBayesContextualMod
from exploChallenge.policies.MostCTRMod import MostCTRMod
from exploChallenge.policies.eAnnealingMod import eAnnealingMod
from exploChallenge.policies.RidgeRegressor import RidgeRegressor

def rargmax(x):
    m = np.amax(x)
    indices = np.nonzero(x == m)[0]
    return random.choice(indices)

class EnsembleTestingModel1(ContextualBanditPolicy):


    def __init__(self, regressor):
        # Create an object from each class to use for ensemble model
        self.regressor = regressor
        self.policy_two = MostCTRMod()
        self.policy_three = NaiveBayesContextualMod()
        self.policies = [self.policy_two, self.policy_three]
        self.policy_two_count = 0
        self.policy_three_count = 0
        self.policy_counts = [self.policy_two_count, self.policy_three_count]
        self.policy_two_scores = {}
        self.policy_three_scores = {}
        self.total_scores = {}
        self.chosen_policy = None

    #@Override
    def getActionToPerform(self, visitor, possibleActions):

        policy_two_scaled_values = self.policy_two.getCTRs(visitor, possibleActions)
        policy_three_scaled_values = self.policy_three.getScaledIndices(visitor, possibleActions)

        #print "Policy two: " + str(policy_two_scaled_values)
        #print "Policy three: " + str(policy_three_scaled_values)
        #print "\n"
        for act in possibleActions:
            self.policy_two_scores[act.getID()] = float(policy_two_scaled_values[act.getID()][0])
            self.policy_three_scores[act.getID()] = float(policy_three_scaled_values[act.getID()][0])

        policy_two_total_score = sum(self.policy_two_scores[v] for v in self.policy_two_scores)
        policy_three_total_score = sum(self.policy_three_scores[v] for v in self.policy_three_scores)

        policy_two_scores_list = [self.policy_two_scores[z.getID()]/policy_two_total_score for z in possibleActions]
        policy_three_scores_list = [self.policy_three_scores[z.getID()]/policy_three_total_score for z in possibleActions]

        #total_scores = [policy_one_scores_list[x.getID()] + policy_two_scores_list[x.getID()] + policy_three_scores_list[x.getID()] for x in possibleActions]

        #print policy_one_scores_list
        #print policy_two_scores_list
        #print policy_three_scores_list
        #print "\n"

        max_policy_two_score = max(policy_two_scores_list)
        max_policy_three_score = max(policy_three_scores_list)

        #print "Max policy one score: " + str(max_policy_one_score)
        #print "Max policy two score: " + str(max_policy_two_score)
        #print "Max policy three score: " + str(max_policy_three_score)

        if (max_policy_three_score >= 1.1 * max_policy_two_score):
            return possibleActions[rargmax(policy_three_scores_list)]

        else:
            return possibleActions[rargmax(policy_two_scores_list)]



#@Override
    def updatePolicy(self, content, chosen_arm, reward, *possibleActions):

        try:
            self.policy_two.updatePolicy(content, chosen_arm, reward)
        except:
            pass
        try:
            self.policy_three.updatePolicy(content, chosen_arm, reward)
        except:
            pass