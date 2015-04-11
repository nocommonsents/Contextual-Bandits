__author__ = 'ftruzzi'
__revised__ = 'bixlermike'

# Final verification 8 Apr 2015

# Naive Bayesian approach based on https://explochallenge.inria.fr/wp-content/uploads/2012/05/paper3.pdf (section 2.3)
import math
import numpy as np
import random as rn
from exploChallenge.policies.ContextualBanditPolicy import ContextualBanditPolicy


def rargmax(vector):
    """ Argmax that chooses randomly among eligible maximum indices. """
    m = np.amax(vector)
    indices = np.nonzero(vector == m)[0]
    return rn.choice(indices)

def rargmin(x):
    m = np.amin(x)
    indices = np.nonzero(x == m)[0]
    return rn.choice(indices)

class NaiveBayesContextualMod(ContextualBanditPolicy):
    def __init__(self):
        self.d = 136
        self.clicks = {}
        self.selections = {}
        self.clicks_per_feature = {}
        self.selections_per_feature = {}
        self.choice = {}
        self.indices = {}
        self.scaled_indices = {}

    def getScaledIndices(self, visitor, possibleActions):

        for article in possibleActions:
            if article.getID() not in self.clicks:
                self.clicks[article.getID()] = 1.0
                self.selections[article.getID()] = 1.0
                self.clicks_per_feature[article.getID()] = np.ones(self.d)
                self.selections_per_feature[article.getID()] = np.ones(self.d)

        for a in possibleActions:
            self.indices[a.getID()] = self.clicks[a.getID()] / self.selections[a.getID()] * np.prod(
            self.clicks_per_feature[a.getID()] / self.clicks_per_feature[a.getID()])
        # Proportion calculation for each feature
        z = sum(math.exp(self.indices[v]) for v in self.indices)
        #print z
        # Calculate the probability that each arm will be selected
        for act in possibleActions:
            self.scaled_indices[act.getID()] = [math.exp(self.clicks[act.getID()] / self.selections[act.getID()] * np.prod(
                self.clicks_per_feature[act.getID()] / self.clicks_per_feature[act.getID()]))/(z)]
        #print self.scaled_indices
        return self.scaled_indices

    def updatePolicy(self, context, action, reward):
        try:
            self.clicks[action.getID()] += reward
            self.selections[action.getID()] += 1.0
            for f, p in enumerate(context.getFeatures()):
                # Feature is "on" and there is a reward given the article
                self.clicks_per_feature[action.getID()][f] += p * float(reward)
                # Feature is present in the article
                self.selections_per_feature[action.getID()][f] += p
            return
        except:
            return