__author__ = 'bixlermike'
#!/usr/bin/env python2.7

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
        # Chooses best article
        self.a_max = 0
        # Holds feature vector
        self.x = {}
        # Transpose of feature vector
        self.xT = {}
        # A inverse times b
        self.theta = {}
        self.thetaT = {}

        self.ID_to_article = {}
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
                self.ID_to_article[article.getID()] = article
            #pa = np.array([float(np.dot(self.thetaT[article.getID()], x)) + self.alpha * np.sqrt(np.dot(xT, (np.dot(self.AI[article.getID()], x))))])
            #self.pa[article.getID()] = np.array([float(np.dot(self.thetaT[article.getID()], x)) + self.alpha * np.sqrt(np.dot(xT, (np.dot(self.AI[article.getID()], x))))])
            self.pa[article.getID()] = [float(np.dot(self.thetaT[article.getID()], x)) + float(self.alpha * np.sqrt(np.dot(xT, (np.dot(self.AI[article.getID()], x)))))]
            self.pa[article.getID()] = float(self.pa[article.getID()][0])
            #print self.pa[article.getID()]
        #print "Value is " + str(self.pa[article.getID()])
        #print "\n"
        #print self.pa
        max_key = max(self.pa.iteritems(), key=operator.itemgetter(1))[0]
        #print max_key
        #print self.ID_to_article[max_key]
        self.a_max = self.ID_to_article[max_key]
        #randomIndex = rn.randint(0, len(possibleActions) - 1)
        #self.a_max = possibleActions[randomIndex]
        self.x = x
        self.xT = xT
        self.pa = {}
        #print "Action is " + str(self.a_max) + " " + str(self.a_max.getID())
        return self.a_max


    def updatePolicy(self, c, chosen_arm, reward):
        # updatePolicy
        if reward == 1:
            self.rewards = 1
        else:
            self.rewards = 0
        #print "Update to " + str(chosen_arm) + " " + str(chosen_arm.getID())
        self.AI[chosen_arm.getID()] = linalg.inv(self.A[chosen_arm.getID()])
        self.A[chosen_arm.getID()] += np.dot(self.x,self.xT)
        self.b[chosen_arm.getID()] += self.rewards * self.x
        self.theta[chosen_arm.getID()] = np.dot(self.AI[chosen_arm.getID()],(self.b[chosen_arm.getID()]))
        self.thetaT[chosen_arm.getID()] = np.transpose(self.theta[chosen_arm.getID()])


