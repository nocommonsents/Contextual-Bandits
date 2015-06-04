__author__ = 'bixlermike'

# Final verification 1 Jun 2015

import random
import re

from exploChallenge.policies.ContextualBanditPolicy import ContextualBanditPolicy
from exploChallenge.policies.MostCTR import MostCTR
from exploChallenge.policies.NaiveBayesContextual import NaiveBayesContextual

class EnsembleRandomUpdateAllModel(ContextualBanditPolicy):


    def __init__(self):
        # Create an object from each class to use for ensemble model
        self.policy_one = MostCTR()
        self.policy_two = NaiveBayesContextual()
        self.policies = [self.policy_one, self.policy_two]
        self.chosen_policy = None

    #@Override
    def getActionToPerform(self, visitor, possibleActions):
        self.chosen_policy =  str(random.choice(self.policies))
        if (re.match('<exploChallenge\.policies\.MostCTR',self.chosen_policy)):
            #print "Choice is MostCTR"
            return self.policy_one.getActionToPerform(visitor, possibleActions)
        elif (re.match('<exploChallenge\.policies\.NaiveBayesContextual',self.chosen_policy)):
            #print "Choice is NaiveBayes"
            return self.policy_two.getActionToPerform(visitor, possibleActions)
        else:
            print "Error in getActionToPerform!"
            return

    #@Override
    def updatePolicy(self, content, chosen_arm, reward, *possibleActions):
        #print "Updating policy " + str(self.chosen_policy)

        try:
            self.policy_one.updatePolicy(content, chosen_arm, reward)
        except:
            pass
        try:
            self.policy_two.updatePolicy(content, chosen_arm, reward)
        except:
            pass
        return