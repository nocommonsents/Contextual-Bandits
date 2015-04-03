__author__ = 'bixlermike'

import random as rn
import numpy as np
import operator

from exploChallenge.policies.ContextualBanditPolicy import ContextualBanditPolicy
from exploChallenge.policies.RidgeRegressor import RidgeRegressor

class eGreedyContextual(ContextualBanditPolicy):
#    def __init__(self, epsilon, regressor):
#        self._epsilon = epsilon
#        self._regressor = regressor

    def __init__(self, epsilon, regressor):
        self.epsilon = epsilon
        self.regressor = regressor
        self.d = 136
        self.regressor_predictions = {}
        #self.clicksPerFeature = {}


    def getEpsilon(self):
        return self.epsilon


    def decide(self, actions):


        explore = np.random.uniform(0.0, 1.0) <= self.epsilon
        if explore:
            i = np.random.randint(len(actions))
            return actions[i]
        else:
            return max(actions, key=self.regressor.predict)


    def getActionToPerform(self, visitor, possibleActions):

        features = np.zeros((self.d, 1))
        #print features
        features[visitor.getFeatures()] = 1.0
        #print visitor.getFeatures()
        #print features
        #self.clicksPerFeature[article.getID()] = np.ones(136)

        if rn.random() > self.epsilon:
            for a in possibleActions:
                #self.clicksPerFeature[a.getID()] = np.ones(136)
                #print sum(visitor.getFeatures()*self.clicksPerFeature[a.getID()])
                print a.getID()
                self.regressor_predictions[a.getID()] = self.regressor.predict(visitor.getFeatures())
                print self.regressor_predictions[a.getID()]
            return max(possibleActions, key=self.regressor.predict)

        else:
            randomIndex = rn.randint(0, len(possibleActions) - 1)
            return possibleActions[randomIndex]


    def learn(self, content, chosen_arm, reward):
        self.regressor.update(chosen_arm, reward)


