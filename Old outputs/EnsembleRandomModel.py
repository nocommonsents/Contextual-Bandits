__author__ = 'bixlermike'

# Final verification 2 May 2015

import numpy as np
import random
import re

from exploChallenge.policies.ContextualBanditPolicy import ContextualBanditPolicy
from exploChallenge.policies.MostCTR import MostCTR
from exploChallenge.policies.UCB1 import UCB1
from exploChallenge.policies.eAnnealingContextual import eAnnealingContextual
from exploChallenge.policies.NaiveBayesContextual import NaiveBayesContextual
from exploChallenge.policies.EXP3 import EXP3
from exploChallenge.policies.LinUCB import LinUCB
from exploChallenge.policies.RidgeRegressor import RidgeRegressor

class EnsembleRandomModel(ContextualBanditPolicy):


    def __init__(self):
        # Create an object from each class to use for ensemble model
        self.policy_one = MostCTR()
        self.policy_two = UCB1()
        self.policy_three = eAnnealingContextual(RidgeRegressor(np.eye(136), np.zeros(136)))
        self.policy_four = NaiveBayesContextual()
        self.policy_five = EXP3(0.5)
        self.policy_six = LinUCB(0.1)
        self.policies = [self.policy_one, self.policy_two, self.policy_three, self.policy_four, self.policy_five, self.policy_six]
        self.chosen_policy = None

    #@Override
    def getActionToPerform(self, visitor, possibleActions):
        self.chosen_policy =  str(random.choice(self.policies))
        if (re.match('<exploChallenge\.policies\.MostCTR',self.chosen_policy)):
            #print "Choice is MostCTR"
            return self.policy_one.getActionToPerform(visitor, possibleActions)
        elif (re.match('<exploChallenge\.policies\.UCB1',self.chosen_policy)):
            #print "Choice is UCB1"
            return self.policy_two.getActionToPerform(visitor, possibleActions)
        elif (re.match('<exploChallenge\.policies\.eAnnealingContextual',self.chosen_policy)):
            #print "Choice is eAnnealingContextual"
            return self.policy_three.getActionToPerform(visitor, possibleActions)
        elif (re.match('<exploChallenge\.policies\.NaiveBayesContextual',self.chosen_policy)):
            #print "Choice is NaiveBayes"
            return self.policy_four.getActionToPerform(visitor, possibleActions)
        elif (re.match('<exploChallenge\.policies\.EXP3',self.chosen_policy)):
            #print "Choice is EXP3"
            return self.policy_five.getActionToPerform(visitor, possibleActions)
        elif (re.match('<exploChallenge\.policies\.LinUCB',self.chosen_policy)):
            #print "Choice is LinUCB'
            return self.policy_six.getActionToPerform(visitor, possibleActions)
        else:
            print "Error in getActionToPerform!"
            return

    #@Override
    def updatePolicy(self, content, chosen_arm, reward, *possibleActions):
        #print "Updating policy " + str(self.chosen_policy)

        if (re.match('<exploChallenge\.policies\.MostCTR',self.chosen_policy)):
            try:
                self.policy_one.updatePolicy(content, chosen_arm, reward)
            except:
                pass
        elif (re.match('<exploChallenge\.policies\.UCB1',self.chosen_policy)):
            try:
                self.policy_two.updatePolicy(content, chosen_arm, reward)
            except:
                pass
        elif (re.match('<exploChallenge\.policies\.eAnnealingContextual',self.chosen_policy)):
            try:
                self.policy_three.updatePolicy(content, chosen_arm, reward)
            except:
                pass
        elif (re.match('<exploChallenge\.policies\.NaiveBayes',self.chosen_policy)):
            try:
                self.policy_four.updatePolicy(content, chosen_arm, reward)
            except:
                pass
        elif (re.match('<exploChallenge\.policies\.EXP3',self.chosen_policy)):
            try:
                self.policy_five.updatePolicy(content, chosen_arm, reward, possibleActions)
            except:
                pass
        elif (re.match('<exploChallenge\.policies\.LinUCB',self.chosen_policy)):
            try:
                self.policy_six.updatePolicy(content, chosen_arm, reward)
            except:
                pass
        else:
            print "Error in updatePolicy!"
        return