__author__ = 'bixlermike'
#!/usr/bin/env python2.7

# LinUCB with disjoint linear models
# Algorithm details and psuedocode at http://www.research.rutgers.edu/~lihong/pub/Li10Contextual.pdf
# Using Ridge-regression as opposed to least-squares solution that could also be explored

# Final verification 8 Apr 2015

import operator
import numpy as np
import random as rn
from scipy import linalg

def rargmax(x):
    m = np.amax(x)
    indices = np.nonzero(x == m)[0]
    return rn.choice(indices)

class LinUCBUpdate:
    def __init__(self):
        # Alpha parameter
        self.alpha = 0.1
        # Dimension of user feature vector
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
        self.thetaT = {}

        self.pa = {}
        self.rewards = 0

    def getAlpha(self):
        return self.alpha

    def getActionToPerform(self, visitor, possibleActions):
        # getActionToPerform
        # Want x to be self.d by 1, so create transpose first
        xT = np.array([visitor.getFeatures()])
        x = np.transpose(xT)

        for article in possibleActions:
            # init collection of matrix/vector Aa, Ba, ba
            if article.getID() not in self.A:
                self.A[article.getID()] = np.identity(self.d)
                self.b[article.getID()] = np.zeros((self.d, 1))
                self.AI[article.getID()] = np.identity(self.d)
                self.theta[article.getID()] = np.dot(self.AI[article.getID()], self.b[article.getID()])
                self.thetaT[article.getID()] = np.transpose(self.theta[article.getID()])

            self.theta[article.getID()] = np.dot(self.AI[article.getID()],(self.b[article.getID()]))
            self.thetaT[article.getID()] = np.transpose(self.theta[article.getID()])
            self.pa[article.getID()] = [float(np.dot(self.thetaT[article.getID()], x)) + float(self.alpha * np.sqrt(np.dot(xT, (np.dot(self.AI[article.getID()], x)))))]
            self.pa[article.getID()] = float(self.pa[article.getID()][0])

        pa_values = [self.pa[a.getID()] for a in possibleActions]
        self.x = x
        self.xT = xT
        self.pa = {}
        return possibleActions[rargmax(pa_values)]

    def updatePolicy(self, c, chosen_arm, reward):
        # updatePolicy
        if reward == 1:
            self.rewards = 1
        else:
            self.rewards = 0
        #print "Update to " + str(chosen_arm) + " " + str(chosen_arm.getID())
        # Part of theta calculation equivalent to x * x tranpose + identity matrix (identity not used variant here)
        self.A[chosen_arm.getID()] += np.dot(self.x,self.xT)
        # Equivalent to x transpose * y (reward)
        self.b[chosen_arm.getID()] += self.rewards * self.x
        # Need to do inverse of A for final calculation of theta
        self.AI[chosen_arm.getID()] = linalg.inv(self.A[chosen_arm.getID()])



