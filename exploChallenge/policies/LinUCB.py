#!/usr/bin/env python2.7

import numpy as np
import math
from scipy import linalg


# lin UCB
class LinUCB:
    def __init__(self):
        # upper bound coefficient
        self.alpha = 3 # if worse -> 2.9, 2.8 1 + np.sqrt(np.log(2/delta)/2)
        self.r1 = 0.5 # if worse -> 0.7, 0.8
        self.r0 = -20 # if worse, -19, -21
        # dimension of user features = d
        self.d = 136
        # Aa : collection of matrix to compute disjoint part for each article a, d*d
        self.Aa = {}
        # AaI : store the inverse of all Aa matrix
        self.AaI = {}
        # ba : collection of vectors to compute disjoin part, d*1
        self.ba = {}

        self.a_max = 0

        self.theta = {}
        self.x = None
        self.xT = None
        # linUCB

    def getActionToPerform(self, visitor, possibleActions):
        # getActionToPerform
        xaT = np.array([visitor.getFeatures()])
        xa = np.transpose(xaT)
        #print xaT
        #print xa

        for article in possibleActions:
            # init collection of matrix/vector Aa, Ba, ba
            if article.getID() not in self.Aa:
                self.Aa[article.getID()] = np.identity(self.d)
                self.ba[article.getID()] = np.zeros((self.d, 1))
                self.AaI[article.getID()] = np.identity(self.d)
                self.theta[article.getID()] = np.zeros((self.d, 1))


        for article in possibleActions:
            #print xaT.shape
            #print self.theta[article.getID()].shape
            #print float(np.dot(xaT, self.theta[article.getID()]))
            pa = np.array([float(np.dot(xaT, self.theta[article.getID()]) + self.alpha * np.sqrt(np.dot(xaT.dot(self.AaI[article.getID()]), xa)))])
        self.a_max = possibleActions[divmod(pa.argmax(), pa.shape[0])[1]]
        self.x = xa
        self.xT = xaT

        return self.a_max


    def updatePolicy(self, c, a, reward):
        # updatePolicy
        if reward == -1:
            pass
        elif reward == 1 or reward == 0:
            if reward == 1:
                r = self.r1
            else:
                r = self.r0
            #print self.x.shape
            #print self.xT.shape

            # Calculation works...problem likely in self.a_max
            #self.a_max = str(self.a_max)
            #print self.Aa[self.a_max]
            #self.Aa[chosen_article] += float(np.dot(self.x, self.xT))
            self.Aa[self.a_max.getID()] += self.x.dot(self.xT)
            self.ba[self.a_max.getID()] += r * self.x
            self.AaI[self.a_max.getID()] = linalg.solve(self.Aa[self.a_max.getID()], np.identity(self.d))
            self.theta[self.a_max.getID()] = self.AaI[self.a_max.getID()].dot(self.ba[self.a_max.getID()])
        else:
            # error
            pass

    #def __hash__(self): return hash(id(self))
    #def __eq__(self, x): return x is self
    #def __ne__(self, x): return x is not self

