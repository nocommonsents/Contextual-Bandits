__author__ = 'bixlermike'

# Final verification 8 Apr 2015

# Ridge regression estimates the weight coefficient vector as:
# Theta = (xT * x + I)^-1 * xT * y
# x: Feature vector, xT = transpose of feature vector, I = identity matrix, y = reward vector (all 0s or 1s)
# Here we use A = (xT * x + I)^-1 and B = (xT * y), so theta is then theta = A^-1 * b
# Get predictions for each article by multiplying weight vector (theta) * feature vector (x)
import random as rn
import numpy as np

from exploChallenge.policies.ContextualBanditPolicy import ContextualBanditPolicy
from exploChallenge.policies.RidgeRegressor import RidgeRegressor

def rargmax(x):
    m = np.amax(x)
    indices = np.nonzero(x == m)[0]
    return rn.choice(indices)

class eGreedyContextual(ContextualBanditPolicy):

    def __init__(self, epsilon, regressor):
        self.epsilon = epsilon
        self.regressor = regressor
        self.d = 136
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

    def getEpsilon(self):
        return self.epsilon


    def getActionToPerform(self, visitor, possibleActions):
        xT = np.array([visitor.getFeatures()])
        x = np.transpose(xT)
        self.x = x
        self.xT = xT
        # Set up dictionaries for any articles not seen previously
        for article in possibleActions:
            if article.getID() not in self.A:
                self.A[article.getID()] = np.identity(self.d)
                self.b[article.getID()] = np.zeros((self.d, 1))
                self.AI[article.getID()] = np.identity(self.d)
                self.theta[article.getID()] = np.dot(self.AI[article.getID()], self.b[article.getID()])
                self.thetaT[article.getID()] = np.transpose(self.theta[article.getID()])
        ## Exploit
        if rn.random() > self.epsilon:
            for article in possibleActions:
                # Completes calculation of theta
                self.theta[article.getID()] = np.dot(self.AI[article.getID()], self.b[article.getID()])
                self.thetaT[article.getID()] = np.transpose(self.theta[article.getID()])
                # Now use estimated feature coefficients to predict which article is best given the contextual information
                self.regressor_predictions[article.getID()] = float(np.dot(self.thetaT[article.getID()], x))

            regressor_values = [self.regressor_predictions[a.getID()] for a in possibleActions]
            return possibleActions[rargmax(regressor_values)]

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
        # Part of theta calculation equivalent to x * x tranpose + identity matrix
        self.A[chosen_arm.getID()] += np.outer(self.x, self.x) + np.identity(self.d)
        # Equivalent to x transpose * y (reward)
        self.b[chosen_arm.getID()] += self.rewards * self.x
        # Need to do inverse of A for final calculation of theta
        self.AI[chosen_arm.getID()] = np.linalg.inv(self.A[chosen_arm.getID()])
        #print str(chosen_arm.getID()) + " " + str(self.thetaT[chosen_arm.getID()])

