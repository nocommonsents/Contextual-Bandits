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
        self.policy_one = NaiveBayesContextualMod()
        self.policy_two = MostCTRMod()
        self.policy_three = eAnnealingMod()
        self.policies = [self.policy_one, self.policy_two, self.policy_three]
        self.policy_one_count = 0
        self.policy_two_count = 0
        self.policy_three_count = 0
        self.policy_counts = [self.policy_one_count, self.policy_two_count, self.policy_three_count]
        self.policy_one_scores = {}
        self.policy_two_scores = {}
        self.policy_three_scores = {}
        self.total_scores = {}
        self.chosen_policy = None

    #@Override
    def getActionToPerform(self, visitor, possibleActions):

        policy_one_scaled_values = self.policy_one.getScaledIndices(visitor, possibleActions)
        policy_two_scaled_values = self.policy_two.getCTRs(visitor, possibleActions)
        policy_three_scaled_values = self.policy_three.getScaledValues(visitor, possibleActions)

        #print policy_one_scaled_values
        #print "\n"
        for act in possibleActions:
            self.policy_one_scores[act.getID()] = float(policy_one_scaled_values[act.getID()][0])
            self.policy_two_scores[act.getID()] = float(policy_two_scaled_values[act.getID()][0])
            self.policy_three_scores[act.getID()] = float(policy_three_scaled_values[act.getID()][0])

        policy_one_total_score = sum(self.policy_one_scores[v] for v in self.policy_one_scores)
        policy_two_total_score = sum(self.policy_two_scores[v] for v in self.policy_two_scores)
        policy_three_total_score = sum(self.policy_three_scores[v] for v in self.policy_three_scores)

        policy_one_scores_list = [self.policy_one_scores[z.getID()]/policy_one_total_score for z in possibleActions]
        policy_two_scores_list = [self.policy_two_scores[z.getID()]/policy_two_total_score for z in possibleActions]
        policy_three_scores_list = [self.policy_three_scores[z.getID()]/policy_three_total_score for z in possibleActions]

        #total_scores = [policy_one_scores_list[x.getID()] + policy_two_scores_list[x.getID()] + policy_three_scores_list[x.getID()] for x in possibleActions]

        #print policy_one_scores_list
        #print policy_two_scores_list
        #print policy_three_scores_list
        #print "\n"

        max_policy_one_score = max(policy_one_scores_list)
        max_policy_two_score = max(policy_two_scores_list)
        max_policy_three_score = max(policy_three_scores_list)

        #print "Max policy one score: " + str(max_policy_one_score)
        #print "Max policy two score: " + str(max_policy_two_score)
        #print "Max policy three score: " + str(max_policy_three_score)

    # If score from one policy is more than 10% better than any score in the other two, select the article with the high score
        if (max_policy_one_score >= max_policy_two_score and max_policy_one_score >= max_policy_three_score):
            percent_improv = min((max_policy_one_score - max_policy_two_score)/max_policy_two_score,(max_policy_one_score - max_policy_three_score)/max_policy_three_score)
            if (percent_improv > 0.1):
                #print "Policy one is winner!"
                return possibleActions[rargmax(policy_one_scores_list)]
        elif (max_policy_two_score >= max_policy_one_score and max_policy_two_score >= max_policy_three_score):
            percent_improv = min((max_policy_two_score - max_policy_one_score)/max_policy_one_score,(max_policy_two_score - max_policy_three_score)/max_policy_three_score)
            if (percent_improv > 0.1):
                #print "Policy two is winner!"
                return possibleActions[rargmax(policy_two_scores_list)]
        elif (max_policy_three_score >= max_policy_one_score and max_policy_three_score >= max_policy_two_score):
            percent_improv = min((max_policy_three_score - max_policy_one_score)/max_policy_one_score,(max_policy_three_score - max_policy_two_score)/max_policy_two_score)
            if (percent_improv > 0.1):
                #print "Policy three is winner!"
                return possibleActions[rargmax(policy_three_scores_list)]
        else:
            print "Issue in calculating percent improvement!"

        # If there's no policy with a score that beats others easily, pick the policy with the best CTR from our previous tests (MostCTR)
        return possibleActions[rargmax(policy_two_scores_list)]

#@Override
    def updatePolicy(self, content, chosen_arm, reward):

        try:
            self.policy_one.updatePolicy(content, chosen_arm, reward)
            self.policy_two.updatePolicy(content, chosen_arm, reward)
            self.policy_three.updatePolicy(content, chosen_arm, reward)
        except:
            pass