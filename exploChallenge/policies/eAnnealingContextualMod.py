__author__ = 'bixlermike'

# Final verification 8 Apr 2015

# Ridge regression estimates the weight coefficient vector as:
# Theta = (xT * x + I)^-1 * xT * y
# x: Feature vector, xT = transpose of feature vector, I = identity matrix, y = reward vector (all 0s or 1s)
# Here we use A = (xT * x + I)^-1 and B = (xT * y), so theta is then theta = A^-1 * b
# Get predictions for each article by multiplying weight vector (theta) * feature vector (x)
import math
import numpy as np
import random

from exploChallenge.policies.ContextualBanditPolicy import ContextualBanditPolicy
from exploChallenge.policies.RidgeRegressor import RidgeRegressor

def rargmax(x):
    m = np.amax(x)
    indices = np.nonzero(x == m)[0]
    return random.choice(indices)

class eAnnealingContextualMod(ContextualBanditPolicy):

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

        self.scaled_regressor_values = {}


    def getEpsilon(self):
        return self.epsilon


    def getScaledRegressorValues(self, visitor, possibleActions):
        xT = np.array([visitor.getFeatures()])
        x = np.transpose(xT)
        self.x = x
        self.xT = xT
        self.epsilon = 1 / math.log(self.trials + 0.0000001)
        self.trials += 1
        # Set up dictionaries for any articles not seen previously
        for article in possibleActions:
            if article.getID() not in self.A:
                self.A[article.getID()] = np.identity(self.d)
                self.b[article.getID()] = np.zeros((self.d, 1))
                self.AI[article.getID()] = np.identity(self.d)
                self.theta[article.getID()] = np.dot(self.AI[article.getID()], self.b[article.getID()])
                self.thetaT[article.getID()] = np.transpose(self.theta[article.getID()])
        ## Exploit
        if random.random() > self.epsilon:
            for article in possibleActions:
                # Completes calculation of theta
                self.theta[article.getID()] = np.dot(self.AI[article.getID()], self.b[article.getID()])
                self.thetaT[article.getID()] = np.transpose(self.theta[article.getID()])
                # Now use estimated feature coefficients to predict which article is best given the contextual information
                self.regressor_predictions[article.getID()] = float(np.dot(self.thetaT[article.getID()], x))
            z = sum(math.exp(self.regressor_predictions[v]) for v in self.regressor_predictions)
            #print z
            for ac in possibleActions:
                self.scaled_regressor_values[ac.getID()] = [math.exp(self.regressor_predictions[ac.getID()])/z]
            return self.scaled_regressor_values

        ## Explore
        else:
            for article in possibleActions:
                self.regressor_predictions[article.getID()] = random.random()
            z = sum(math.exp(self.regressor_predictions[v]) for v in self.regressor_predictions)
            for ac in possibleActions:
                self.scaled_regressor_values[ac.getID()] = [math.exp(self.regressor_predictions[ac.getID()])/z]
            #print self.scaled_regressor_values
            return self.scaled_regressor_values



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

