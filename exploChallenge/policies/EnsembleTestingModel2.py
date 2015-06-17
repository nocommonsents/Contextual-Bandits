__author__ = 'bixlermike'

import numpy as np
import random
import time

from exploChallenge.policies.ContextualBanditPolicy import ContextualBanditPolicy
from exploChallenge.policies.NaiveBayesContextualMod import NaiveBayesContextualMod
from exploChallenge.policies.MostCTRMod import MostCTRMod
from exploChallenge.policies.eAnnealingMod import eAnnealingMod
from exploChallenge.policies.RidgeRegressor import RidgeRegressor

def rargmax(x):
    m = np.amax(x)
    indices = np.nonzero(x == m)[0]
    return random.choice(indices)

class EnsembleTestingModel2(ContextualBanditPolicy):


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
        self.policy_two_scaled_scores = {}
        self.policy_three_scaled_scores = {}
        self.total_scores = {}
        self.arrival_times = {}
        self.chosen_policy = None

    #@Override
    def getActionToPerform(self, visitor, possibleActions):

        # for action in possibleActions:
        #     if action.getID() not in self.arrival_times:
        #         self.arrival_times[action.getID()] = int(time.time())
        #
        # all_times = [self.arrival_times[a.getID()] for a in possibleActions]

        policy_two_scaled_values = self.policy_two.getCTRs(visitor, possibleActions)
        policy_three_scaled_values = self.policy_three.getScaledIndices(visitor, possibleActions)

        #print "\n"
        for act in possibleActions:
            self.policy_two_scores[act.getID()] = float(policy_two_scaled_values[act.getID()][0])
            self.policy_three_scores[act.getID()] = float(policy_three_scaled_values[act.getID()][0])

        policy_two_total_score = sum(self.policy_two_scores[v] for v in self.policy_two_scores)
        policy_three_total_score = sum(self.policy_three_scores[v] for v in self.policy_three_scores)

        for z in possibleActions:
            self.policy_two_scaled_scores[z.getID()] = self.policy_two_scores[z.getID()]/policy_two_total_score
            self.policy_three_scaled_scores[z.getID()] = self.policy_three_scores[z.getID()]/policy_three_total_score
            #self.total_scores[z.getID()] =

        total_scores = [self.policy_two_scaled_scores[a.getID()] + self.policy_three_scaled_scores[a.getID()] for a in possibleActions]

        #print total_scores
        #print rargmax(total_scores)
        #print "\n\n"


        return possibleActions[rargmax(total_scores)]

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
        return