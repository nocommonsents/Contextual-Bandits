__author__ = 'bixlermike'

# Final verification 1 Jun 2015

import numpy as np
import random

from exploChallenge.policies.ContextualBanditPolicy import ContextualBanditPolicy
from exploChallenge.policies.RidgeRegressor import RidgeRegressor

from exploChallenge.policies.BinomialUCI import BinomialUCI
from exploChallenge.policies.MostCTR import MostCTR
from exploChallenge.policies.Softmax import Softmax
from exploChallenge.policies.UCB1 import UCB1
from exploChallenge.policies.LinUCB import LinUCB
from exploChallenge.policies.NaiveBayesContextual import NaiveBayesContextual
from exploChallenge.policies.SoftmaxContextual import SoftmaxContextual

output_file = open("banditPolicyProportionsVsEvalNumber.txt", "a+")
#output_file = open("testPolicyCountsVsEvalNumber.txt", "a+")

class EnsembleRandomModel(ContextualBanditPolicy):

# Candidates:
# Context-free: BinomialUCI, EXP3, MostCTR, Softmax0.01, EAnnealing, UCB1
# Contextual: LinUCB, NaiveBayes, SoftmaxContextual0.01, EAnnealingContextual

    def __init__(self):
        # Create an object from each class to use for ensemble model
        self.policy_one = BinomialUCI()
        self.policy_two = MostCTR()
        self.policy_three = Softmax(0.1)
        self.policy_four = UCB1()
        self.policy_five = LinUCB(0.1)
        self.policy_six = NaiveBayesContextual()
        self.policy_seven = SoftmaxContextual(0.1, RidgeRegressor(np.eye(136), np.zeros(136)))
        self.policies = [self.policy_one, self.policy_two, self.policy_three, self.policy_four, self.policy_five,
                         self.policy_six, self.policy_seven]
        self.policy_nicknames = ["BinomialUCI", "MostCTR", "Softmax0.1", "UCB1", "LinUCB(0.1)", "NaiveBayesContextual",
                                 "SoftmaxContextual0.1"]
        self.chosen_policy = None
        self.policy_counts = {}
        self.trials = 0
        self.updates = 0
        for i in self.policies:
            self.policy_counts[str(i)] = 0

    #@Override
    def getActionToPerform(self, visitor, possibleActions):
        self.chosen_policy =  random.choice(self.policies)
        #print "Chosen policy: " + str(self.chosen_policy)
        return self.chosen_policy.getActionToPerform(visitor, possibleActions)

    #@Override
    def updatePolicy(self, content, chosen_arm, reward, *possibleActions):
        self.updates += 1
        self.policy_counts[str(self.chosen_policy)] += 1
        try:
            #print "Updating: " + str(self.chosen_policy)
            self.chosen_policy.updatePolicy(content, chosen_arm, reward)
        except:
            print "Error updating: " + str(self.chosen_policy) + " for chosen arm " + str(chosen_arm) + "."
            pass
        if (self.updates % 100 == 0):
            for i in self.policies:
                print str("EnsembleRandom") + "," + str(self.policy_nicknames[self.policies.index(i)]) + "," + \
                      str(self.updates) + "," + str(float(self.policy_counts[str(i)])/self.updates)
                output_file.write(str("EnsembleRandom") + "," + str(self.policy_nicknames[self.policies.index(i)]) + ","
                + str(self.updates) + "," + str(float(self.policy_counts[str(i)])/self.updates) + "\n")