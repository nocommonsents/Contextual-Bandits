__author__ = 'bixlermike'

# Ridge regression estimates the weight coefficient vector as:
# Theta = (xT * x + I)^-1 * xT * y
# x: Feature vector, xT = transpose of feature vector, I = identity matrix, y = reward vector (all 0s or 1s)
# Here we use A = (xT * x + I)^-1 and B = (xT * y), so theta is then theta = A^-1 * b
# Get predictions for each article by multiplying weight vector (theta) * feature vector (x)
import math
import numpy as np
import random as rn

from exploChallenge.policies.ContextualBanditPolicy import ContextualBanditPolicy
from exploChallenge.policies.RidgeRegressor import RidgeRegressor

def categorical_draw(probs):
    z = rn.random()
    cum_prob = 0.0
    for p in probs:
        prob = probs[p]
        cum_prob += prob
        if cum_prob > z:
            return p
    return probs.iterkeys().next()


class SoftmaxContextual(ContextualBanditPolicy):

    def __init__(self, temp, regressor):
        self.temperature = temp
        self.regressor = regressor
        self.predicted_arm_values = {}
        self.d = 136
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

    def getTemp(self):
        return self.temperature


    def getActionToPerform(self, visitor, possibleActions):
        xT = np.array([visitor.getFeatures()])
        x = np.transpose(xT)
        regressor_values = {}
        arm_probs = {}
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

            # Now use estimated feature coefficients to predict which article is best given the contextual information
            self.predicted_arm_values[article.getID()] = float(np.dot(self.thetaT[article.getID()], x))

        arm_scores = [math.exp(self.predicted_arm_values[a.getID()]/self.temperature) for a in possibleActions]
        #print "Arm scores: " + str(arm_scores)
        # Normalization factor z
        z = sum(arm_scores)
        # Calculate the probability that each arm will be selected
        for v in possibleActions:
            arm_probs[v.getID()]= math.exp(self.predicted_arm_values[v.getID()] / self.temperature) / z
        #print "Arm probs: " + str(arm_probs)
        # Generate random number and see which bin it falls into to select arm
        id = categorical_draw(arm_probs)
        for action in possibleActions:
            if action.getID() == id:
                return action


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

