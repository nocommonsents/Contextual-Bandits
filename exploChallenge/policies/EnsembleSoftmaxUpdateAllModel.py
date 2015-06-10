__author__ = 'bixlermike'

# Final verification 1 Jun 2015

import math
import numpy as np
import random
import re

from exploChallenge.policies.ContextualBanditPolicy import ContextualBanditPolicy
from exploChallenge.policies.eAnnealing import eAnnealing
from exploChallenge.policies.MostCTR import MostCTR
from exploChallenge.policies.NaiveBayesContextual import NaiveBayesContextual


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
        self.policy_one = MostCTR()
        self.policy_two = NaiveBayesContextual()
        self.policies = [self.policy_one, self.policy_two]
        self.policy_counts = {}
        self.policy_scores = {}
        self.chosen_policy = None
        self.temperature = temp

    def getTemp(self):
        return self.temperature

    #@Override
    def getActionToPerform(self, visitor, possibleActions):

        probs = self.policy_scores
        for i in self.policies:
            if str(i) not in self.policy_counts:
                self.policy_counts[str(i)] = 0
                self.policy_scores[str(i)] = 0

        z = sum([math.exp(self.policy_scores[v] / self.temperature) for v in self.policy_scores])
        #print z

        # Calculate the probability that each arm will be selected
        for v in self.policy_scores:
            probs[v]= math.exp(self.policy_scores[v] / self.temperature) / z

        # Generate random number and see which bin it falls into to select arm
        #print probs
        self.chosen_policy = str(categorical_draw(probs))
        #print self.chosen_policy
        #print "ID equals: " + str(self.chosen_policy)

        if (re.match('<exploChallenge\.policies\.MostCTR',self.chosen_policy)):
            return self.policy_one.getActionToPerform(visitor, possibleActions)
        elif (re.match('<exploChallenge\.policies\.Naive',self.chosen_policy)):
            return self.policy_two.getActionToPerform(visitor, possibleActions)
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


        self.policy_counts[str(self.chosen_policy)] += 1

        if reward is True:
            self.policy_scores[str(self.chosen_policy)] = ((self.policy_counts[str(self.chosen_policy)] - 1) / float(self.policy_counts[str(self.chosen_policy)])) * \
                                                          self.policy_scores[str(self.chosen_policy)] + (1 / float(self.policy_counts[str(self.chosen_policy)]))
            print "Scores are: " + str(self.policy_scores)
            print "Counts are: " + str(self.policy_counts)
        return

