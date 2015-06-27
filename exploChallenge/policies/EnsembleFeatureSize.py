__author__ = 'bixlermike'

import numpy as np
import random

from exploChallenge.policies.ContextualBanditPolicy import ContextualBanditPolicy

from exploChallenge.policies.MostCTR import MostCTR
from exploChallenge.policies.NaiveBayesContextual import NaiveBayesContextual

def rargmax(x):
    m = np.amax(x)
    indices = np.nonzero(x == m)[0]
    return random.choice(indices)

class EnsembleFeatureSize(ContextualBanditPolicy):


    def __init__(self):
        # Create an object from each class to use for ensemble model
        self.policy_one = MostCTR()
        self.policy_two = NaiveBayesContextual()
        self.chosen_policy = None
        #self.count_no_context = 0
        #self.count = 0

    #@Override
    def getActionToPerform(self, visitor, possibleActions):

        a = visitor.getFeatures()
        num_features = sum(a)
        #self.count += 1
        if (num_features == 1):
        #    print a
        #    self.count_no_context += 1
        #    print str(self.count_no_context) + " " + str(self.count)
            return self.policy_one.getActionToPerform(visitor, possibleActions)
        else:
            return self.policy_two.getActionToPerform(visitor, possibleActions)

#@Override
    def updatePolicy(self, content, chosen_arm, reward, *possibleActions):

        try:
            self.policy_one.updatePolicy(content, chosen_arm, reward)
        except:
            pass
        try:
            self.policy_two.updatePolicy(content, chosen_arm, reward)
        except:
            pass

        return


