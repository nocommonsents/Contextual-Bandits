__author__ = 'bixlermike'
#!/usr/bin/env python2.7

import numpy as np
import random as rn
from scipy import linalg

class LinUCBUpdate:
    def __init__(self):
        # Alpha parameter
        self.alpha = 1
        # Dimension of user feature vector
        self.d = 29
        # A dxd identity matrix
        self.A = {}
        # Inverse of A
        self.AI = {}
        # A dxd zeroes matrix
        self.b = {}
        # Chooses best article
        self.a_max = 0
        # Holds feature vector
        self.x = {}
        # Transpose of feature vector
        self.xT = {}
        # A inverse times b
        self.theta = {}
        self.thetaT = {}

        self.rewards = 0
        # linUCB

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
                self.theta[article.getID()] = np.zeros((self.d,1))
                self.thetaT[article.getID()] = np.transpose(self.theta[article.getID()])
            #print x
            print self.thetaT[article.getID()].shape
            print x.shape
            pa = np.array([float(np.dot(self.thetaT[article.getID()], x)) + self.alpha * np.sqrt(np.dot(xT, (np.dot(self.AI[article.getID()], x))))])
        self.a_max = possibleActions[divmod(pa.argmax(), pa.shape[0])[1]]
        #self.theta[self.a_max.getID()] = self.AI[self.a_max.getID()]*(self.b[self.a_max.getID()])
        self.x = x
        self.xT = xT
        #print self.a_max

        return self.a_max


    def updatePolicy(self, c, a, reward):
        # updatePolicy
        if reward == 1:
            self.rewards = 1
        else:
            self.rewards = 0

        self.AI[self.a_max.getID()] = linalg.inv(self.A[self.a_max.getID()])
        self.A[self.a_max.getID()] += np.dot(self.x,self.xT)
        self.b[self.a_max.getID()] += self.rewards * self.x
        self.theta[self.a_max.getID()] = np.dot(self.AI[self.a_max.getID()],(self.b[self.a_max.getID()]))
        self.thetaT[self.a_max.getID()] = np.transpose(self.theta[self.a_max.getID()])


