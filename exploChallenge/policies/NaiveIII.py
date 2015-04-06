__author__ = 'ftruzzi'

# My naive bayes approach
import numpy as np
import random as rn
from collections import defaultdict
from exploChallenge.policies.ContextualBanditPolicy import ContextualBanditPolicy


def rargmax(vector):
    """ Argmax that chooses randomly among eligible maximum indices. """
    m = np.amax(vector)
    indices = np.nonzero(vector == m)[0]
    return rn.choice(indices)


class Naive3(ContextualBanditPolicy):
    def __init__(self):
        self.d = 136
        self.clicks = {}
        self.selections = {}
        self.clicksPerFeature = {}
        self.selectionsPerFeature = {}
        self.choice = {}
        self.used = {}
        self.t = 0
    def getActionToPerform(self, visitor, possibleActions):

        for article in possibleActions:
            if article.getID() not in self.clicks:
                self.clicks[article.getID()] = 1.0 #Optimistic
                self.selections[article.getID()] = 1.0
                self.clicksPerFeature[article.getID()] = np.ones(self.d)
                self.selectionsPerFeature[article.getID()] = np.ones(self.d)
                self.used[article.getID()] = 0

            #print sum(visitor.getFeatures()*self.clicksPerFeature[article.getID()])

        #if np.random.random() > 0.1:
        indices = [self.clicks[a.getID()] / self.selections[a.getID()] * np.prod(
                self.clicksPerFeature[a.getID()] / self.clicksPerFeature[a.getID()]) for a in possibleActions]
        choice = possibleActions[rargmax(indices)]
        # else:
        #     choice = rn.choice(possibleActions)

        self.t += 1
        self.used[choice.getID()] += 1
        return choice

    def updatePolicy(self, context, action, reward):
        try:
            self.clicks[action.getID()] += reward
            self.selections[action.getID()] += 1.0
            for f, p in enumerate(context.getFeatures()):
                # Feature is "on" and there is a reward given the article
                self.clicksPerFeature[action.getID()][f] += p * float(reward)
                # Feature is present is given the article
                self.selectionsPerFeature[action.getID()][f] += p

                #self.pai[a.getID()] += features * (float(reward) - self.pai[a.getID()]) / (1.0 + self.selections[a.getID()])
            return
        except:
            return