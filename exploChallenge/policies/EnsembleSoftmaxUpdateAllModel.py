__author__ = 'bixlermike'

# Final verification 1 Jun 2015

import math
import numpy as np
import random
import re

from exploChallenge.policies.ContextualBanditPolicy import ContextualBanditPolicy
from exploChallenge.policies.RidgeRegressor import RidgeRegressor

from exploChallenge.policies.NaiveBayesContextual import NaiveBayesContextual
from exploChallenge.policies.BinomialUCI import BinomialUCI
from exploChallenge.policies.UCB1 import UCB1
from exploChallenge.policies.EXP3 import EXP3


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
        # Create an object from each class to use for ensemble model
        self.policy_one = NaiveBayesContextual()
        self.policy_two = BinomialUCI()
        self.policy_three = UCB1()
        self.policy_four = EXP3(0.5)
        self.policies = [self.policy_one, self.policy_two, self.policy_three, self.policy_four]
        self.policy_counts = {}
        self.policy_scores = {}
        self.chosen_policy = None
        self.temperature = temp
        self.total_counts = 0

    def getTemp(self):
        return self.temperature

    #@Override
    def getActionToPerform(self, visitor, possibleActions):
        self.total_counts += 1
        policy_probs = {}

        for i in self.policies:
            if str(i) not in self.policy_counts:
                self.policy_counts[str(i)] = 0
                self.policy_scores[str(i)] = 0

        policy_scores = [math.exp(self.policy_scores[str(a)]/self.temperature) for a in self.policy_scores]
        #print "Policy scores: " + str(policy_scores)
        z = sum(policy_scores)
        #print z

        # Calculate the probability that each arm will be selected
        for v in self.policies:
            policy_probs[v] = math.exp(self.policy_scores[str(v)] / self.temperature) / z

        # Generate random number and see which bin it falls into to select arm
        self.chosen_policy = str(categorical_draw(policy_probs))
        #print self.chosen_policy
        #print "ID equals: " + str(self.chosen_policy)

        if (re.match('<exploChallenge\.policies\.NaiveBayes',self.chosen_policy)):
            return self.policy_one.getActionToPerform(visitor, possibleActions)
        elif (re.match('<exploChallenge\.policies\.BinomialUCI',self.chosen_policy)):
            return self.policy_two.getActionToPerform(visitor, possibleActions)
        elif (re.match('<exploChallenge\.policies\.UCB1',self.chosen_policy)):
            return self.policy_three.getActionToPerform(visitor, possibleActions)
        elif (re.match('<exploChallenge\.policies\.EXP3',self.chosen_policy)):
            return self.policy_four.getActionToPerform(visitor, possibleActions)
        else:
            print "Error in getActionToPerform!"
            return

    #@Override
    def updatePolicy(self, content, chosen_arm, reward, *possibleActions):
        #print "Updating policy " + str(self.chosen_policy)
        try:
            self.policy_one.updatePolicy(content, chosen_arm, reward)
        except:
            pass
        try:
            self.policy_two.updatePolicy(content, chosen_arm, reward)
        except:
            pass
        try:
            self.policy_three.updatePolicy(content, chosen_arm, reward)
        except:
            pass
        try:
            self.policy_four.updatePolicy(content, chosen_arm, reward)
        except:
            pass

        self.policy_counts[str(self.chosen_policy)] += 1

        old_value = self.policy_scores[str(self.chosen_policy)]
        n = self.policy_counts[str(self.chosen_policy)]
        new_value = ((n - 1) / float(n)) * old_value + reward * (1 / float(n))
        #print "Old value: " + str(old_value)
        #print "New value:" + str(new_value)
        self.policy_scores[str(self.chosen_policy)] = new_value


        if (self.total_counts % 100 == 0):
            print "Scores are: " + str(self.policy_scores)
            print "Counts are: " + str(self.policy_counts)
        return

