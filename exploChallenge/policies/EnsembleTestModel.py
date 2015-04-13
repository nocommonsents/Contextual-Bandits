__author__ = 'bixlermike'

import numpy as np
import random
import re

from exploChallenge.policies.ContextualBanditPolicy import ContextualBanditPolicy
from exploChallenge.policies.eGreedyContextualMod import eGreedyContextualMod
from exploChallenge.policies.NaiveBayesContextualMod import NaiveBayesContextualMod
from exploChallenge.policies.LinUCBMod import LinUCBMod
from exploChallenge.policies.RidgeRegressor import RidgeRegressor

def rargmax(x):
    m = np.amax(x)
    indices = np.nonzero(x == m)[0]
    return random.choice(indices)

class EnsembleTestModel(ContextualBanditPolicy):


    def __init__(self, epsilon, regressor):
        # Create an object from each class to use for ensemble model
        self.epsilon = epsilon
        self.regressor = regressor
        self.policy_one = eGreedyContextualMod(0.1, RidgeRegressor(np.eye(136), np.zeros(136)))
        self.policy_two = NaiveBayesContextualMod()
        self.policy_three = LinUCBMod()
        self.policies = [self.policy_one, self.policy_two, self.policy_three]
        self.policy_one_count = 0
        self.policy_two_count = 0
        self.policy_three_count = 0
        self.policy_one_score = 0
        self.policy_two_score = 0
        self.policy_three_score = 0
        self.policy_counts = [self.policy_one_count, self.policy_two_count, self.policy_three_count]
        self.policy_scores = [self.policy_one_score, self.policy_two_score, self.policy_three_score]
        self.final_scaled_values = {}
        self.chosen_policy = None

    #@Override
    def getActionToPerform(self, visitor, possibleActions):

        random_number = random.random()

        if random_number > self.epsilon:
            total_scores = {}
            policy_one_scaled_values = self.policy_one.getScaledRegressorValues(visitor, possibleActions)
            print policy_one_scaled_values
            policy_two_scaled_values =  self.policy_two.getScaledIndices(visitor, possibleActions)
            print policy_two_scaled_values
            policy_three_scaled_values = self.policy_three.getScaledPaValues(visitor, possibleActions)
            print policy_three_scaled_values

            #total_scores = ((policy_one_scaled_values[v.getID()][0] + policy_two_scaled_values[v.getID()][0] + policy_three_scaled_values[v.getID()][0]) for v in possibleActions)
            #for v in possibleActions:
                #temp_score = policy_one_scaled_values[v.getID()][0] + policy_two_scaled_values[v.getID()][0] + policy_three_scaled_values[v.getID()][0]
                #total_scores[v.getID()] = float(temp_score)
            total_scores = [float(policy_one_scaled_values[v.getID()][0]) + float(policy_two_scaled_values[v.getID()][0]) + float(policy_three_scaled_values[v.getID()][0]) for v in possibleActions]

            print str(total_scores) + "\n"
            print str(max(total_scores)) + " " + str(rargmax(total_scores))
            return possibleActions[rargmax(total_scores)]

        else:
            randomIndex = random.randint(0, len(possibleActions) - 1)
            return possibleActions[randomIndex]

    #@Override
    def updatePolicy(self, content, chosen_arm, reward):

        self.policy_one.updatePolicy(content, chosen_arm, reward)
        self.policy_two.updatePolicy(content, chosen_arm, reward)
        self.policy_three.updatePolicy(content, chosen_arm, reward)

        return