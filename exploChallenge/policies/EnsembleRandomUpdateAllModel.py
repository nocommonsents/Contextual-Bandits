__author__ = 'bixlermike'

# Final verification 1 Jun 2015

import numpy as np
import random

from exploChallenge.policies.ContextualBanditPolicy import ContextualBanditPolicy
from exploChallenge.policies.RidgeRegressor import RidgeRegressor

from exploChallenge.policies.BinomialUCI import BinomialUCI
from exploChallenge.policies.MostRecent import MostRecent
from exploChallenge.policies.MostCTR import MostCTR
from exploChallenge.policies.Softmax import Softmax
from exploChallenge.policies.eAnnealing import eAnnealing
from exploChallenge.policies.UCB1 import UCB1
from exploChallenge.policies.LinUCB import LinUCB
from exploChallenge.policies.NaiveBayesContextual import NaiveBayesContextual
from exploChallenge.policies.SoftmaxContextual import SoftmaxContextual
from exploChallenge.policies.eAnnealingContextual import eAnnealingContextual

class EnsembleRandomUpdateAllModel(ContextualBanditPolicy):


    def __init__(self):
        # Create an object from each class to use for ensemble model
        self.policy_one = BinomialUCI()
        self.policy_two = MostCTR()
        self.policy_three = Softmax(0.01)
        self.policy_four = UCB1()
        self.policy_five = LinUCB(0.1)
        self.policy_six = NaiveBayesContextual()
        self.policy_seven = SoftmaxContextual(0.01, RidgeRegressor(np.eye(136), np.zeros(136)))
        self.policies = [self.policy_one, self.policy_two, self.policy_three, self.policy_four, self.policy_five,
                     self.policy_six, self.policy_seven]
        self.chosen_policy = None

    #@Override
    def getActionToPerform(self, visitor, possibleActions):
        self.chosen_policy =  random.choice(self.policies)
        #print "Chosen policy: " + str(self.chosen_policy) + "\n"
        return self.chosen_policy.getActionToPerform(visitor, possibleActions)

    #@Override
    def updatePolicy(self, content, chosen_arm, reward, *possibleActions):
        #print "Updating policy " + str(self.chosen_policy)
        for p in self.policies:
            try:
                #print "Updating policy: " + str(p)
                p.updatePolicy(content, chosen_arm, reward)
            except:
                #print "Error updating: " + str(self.chosen_policy) + " for chosen arm " + str(chosen_arm) + "."
                pass