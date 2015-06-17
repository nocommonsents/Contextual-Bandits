__author__ = 'dai.shi'

import random as rn
import operator
import math
import numpy as np

from exploChallenge.policies.ContextualBanditPolicy import ContextualBanditPolicy

def rargmax(x):
    m = np.amax(x)
    indices = np.nonzero(x == m)[0]
    return rn.choice(indices)

class LinearBayes(ContextualBanditPolicy):


    def __init__(self):
        self.clicksPerFeature = {}
        self.selectionsPerFeature = {}
        self.clicksPerArticle = {}
        self.selectionsPerArticle = {}
        self.FeatureClicks = np.ones(136)
        self.FeatureSelections = np.ones(136)
        return
    #     def __init__(self):
#         self.clicksPerFeature = {}
#         self.selectionsPerFeature = {}
#         self.clicksPerArticle = {}
#         self.selectionPerArticle = {}
#         self.featureClickRate = {}
#         for feature in xrange(136):
#             self.featureClickRate[feature] = 0

    def getActionToPerform(self, visitor,possibleActions):
        pai = {}
        pa = {}


        for action in possibleActions:

            if action.getID() not in self.clicksPerFeature:
                self.clicksPerFeature[action.getID()] = np.ones(136)
                self.selectionsPerFeature[action.getID()] = np.ones(136)
                self.clicksPerArticle[action.getID()] = 1
                self.selectionsPerArticle[action.getID()] = 1

            pa[action.getID()] = self.clicksPerArticle[action.getID()]/self.selectionsPerArticle[action.getID()]
            #pai[action.getID()] = self.clicksPerFeature[action.getID()]/self.selectionsPerFeature[action.getID()]
            pai[action.getID()] = self.clicksPerFeature[action.getID()]/self.selectionsPerFeature[action.getID()]
        pi = self.FeatureClicks/self.FeatureSelections

        #probs = [sum(visitor.getFeatures()*(pai[action.getID()]/pi))*pa[action.getID()] for action in possibleActions]
        probs = [sum(visitor.getFeatures()*pai[action.getID()])*pa[action.getID()] for action in possibleActions]

        action = possibleActions[rargmax(probs)]
        return action

#         for article in possibleActions:
#             if article.getID() not in self.clicksPerFeature:
#                 self.clicksPerFeature[article.getID()] = defaultdict(int)
#                 self.selectionsPerFeature[article.getID()] = defaultdict(int)
#                 self.clicksPerArticle[article.getID()] = 1
#                 self.selectionPerArticle[article.getID()] = 1
#
#         # corrigir quando nao estiver com sono
#
#         sumOfclicks = np.asarray([sum(self.clicksPerFeature[a.getID()].values() + [1]) for a in possibleActions])
#         sumOfSelections = np.asarray([sum(self.selectionsPerFeature[a.getID()].values() + [1]) for a in possibleActions])
#         Pai = sumOfclicks/sumOfSelections
#
#         Pa = np.asarray([self.clicksPerArticle[a.getID()] / self.selectionPerArticle[a.getID()] for a in possibleActions])
#         Pi = np.asarray([self.clicksPerFeature[a.getID()] / self.selectionsPerFeature[a.getID()] for a in possibleActions])
#
#         action = possibleActions[np.argmax((sumOfclicks / sumOfSelections) * Pa / Pi)]
#
#
#         #action = possibleActions[np.argmax([((1.0 * x) / y) for x, y in zip(sumOfclicks, sumOfSelections)])]
#
#         return action

    def updatePolicy(self, content, chosen_arm, reward):
        self.selectionsPerFeature[chosen_arm.getID()] += content.getFeatures()
        self.FeatureSelections += content.getFeatures()
        self.selectionsPerArticle[chosen_arm.getID()] += 1
        if reward is True:
            self.FeatureClicks += content.getFeatures()
            self.clicksPerFeature[chosen_arm.getID()] += content.getFeatures()
            self.clicksPerArticle[chosen_arm.getID()] += 1
        return

#         self.selectionPerArticle[a.getID()] += 1
#         if reward is True:
#             self.clicksPerArticle += 1
#         for f, p in enumerate(c.getFeatures()):
#             if p is 1:
#                 self.featureClickRate[f] += 1
#                 if reward is True:
#                     self.clicksPerFeature[a.getID()][f] += 1.0
#                 self.selectionsPerFeature[a.getID()][f] += 1.0


