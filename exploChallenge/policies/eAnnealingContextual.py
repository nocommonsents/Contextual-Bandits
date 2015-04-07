__author__ = 'bixlermike'

import random as rn
import math
import numpy as np
import operator

from exploChallenge.policies.ContextualBanditPolicy import ContextualBanditPolicy
from exploChallenge.policies.RidgeRegressor import RidgeRegressor

def rargmax(x):
    m = np.amax(x)
    indices = np.nonzero(x == m)[0]
    return rn.choice(indices)

class eAnnealingContextual(ContextualBanditPolicy):

    def __init__(self, regressor):
        self.regressor = regressor
        self.d = 136
        self.trials = 1
        self.regressor_predictions = {}
        # A dxd identity matrix
        self.A = {}
        # Inverse of A
        self.AI = {}
        # A dxd zeroes matrix
        self.b = {}
        # Holds feature vector
        self.x = {}
        # Transpose of feature vector
        self.xT = {}
        # A inverse times b
        self.theta = {}
        # Transpose of theta
        self.thetaT = {}


    def decide(self, possibleActions):
        explore = rn.random() <= self.epsilon
        if rn.random() > self.epsilon:
            return max(possibleActions, key=self.regressor.predict)

        else:
            randomIndex = rn.randint(0, len(possibleActions) - 1)
        return possibleActions[randomIndex]


    def getActionToPerform(self, visitor, possibleActions):

        xT = np.array([visitor.getFeatures()])
        x = np.transpose(xT)
        self.epsilon = 1 / math.log(self.trials + 0.0000001)
        self.trials += 1
        self.ID_to_article = {}
        self.x = x
        self.xT = xT
        ## Exploit
        if rn.random() > self.epsilon:
            for article in possibleActions:
                if article.getID() not in self.A:
                    self.A[article.getID()] = np.identity(self.d)
                    self.b[article.getID()] = np.zeros((self.d, 1))
                    self.AI[article.getID()] = np.identity(self.d)
                    self.theta[article.getID()] = np.dot(self.AI[article.getID()], self.b[article.getID()])
                    self.thetaT[article.getID()] = np.transpose(self.theta[article.getID()])
                self.ID_to_article[article.getID()] = article
                # Completes calculation of theta
                self.theta[article.getID()] = np.dot(self.AI[article.getID()], self.b[article.getID()])
                self.thetaT[article.getID()] = np.transpose(self.theta[article.getID()])

                # Now use estimated feature coefficients to predict which article is best given the contextual information
                self.regressor_predictions[article.getID()] = float(np.dot(self.thetaT[article.getID()], x))
            max_key = max(self.regressor_predictions.iteritems(), key=operator.itemgetter(1))[0]
            self.a_max = self.ID_to_article[max_key]
            return self.a_max

        ## Explore
        else:
            randomIndex = rn.randint(0, len(possibleActions) - 1)
            return possibleActions[randomIndex]


    def updatePolicy(self, content, chosen_arm, reward):
        # updatePolicy
        if reward == 1:
            self.rewards = 1
        else:
            self.rewards = 0
        # Part of theta calculation equivalent to X * X tranpose + identity matrix
        self.A[chosen_arm.getID()] += np.outer(self.x, self.x) + np.identity(self.d)
        # Equivalent to X tranpose * y (reward)
        self.b[chosen_arm.getID()] += self.rewards * self.x
        # Need to do inverse of A for final calculation of theta
        self.AI[chosen_arm.getID()] = np.linalg.inv(self.A[chosen_arm.getID()])

