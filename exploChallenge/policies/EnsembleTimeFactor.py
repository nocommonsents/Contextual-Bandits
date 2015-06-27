__author__ = 'bixlermike'

import datetime
import numpy as np
import random

from exploChallenge.policies.ContextualBanditPolicy import ContextualBanditPolicy

from exploChallenge.policies.MostCTR import MostCTR
from exploChallenge.policies.NaiveBayesContextual import NaiveBayesContextual

def rargmax(x):
    m = np.amax(x)
    indices = np.nonzero(x == m)[0]
    return random.choice(indices)

class EnsembleTimeFactor(ContextualBanditPolicy):


    def __init__(self):
        # Create an object from each class to use for ensemble model
        self.policy_one = MostCTR()
        self.policy_two = NaiveBayesContextual()
        self.chosen_policy = None

    #@Override
    def getActionToPerform(self, visitor, possibleActions):

        timestamp = visitor.getTimestamp()
        print timestamp
        t = datetime.datetime.fromtimestamp(float(timestamp))
        fmt = "%Y-%m-%d %H:%M:%S"
        fmt_hours = "%H"
        formatted_time = t.strftime(fmt)
        formatted_time_hours = t.strftime(fmt_hours)
        print formatted_time_hours

        return self.policy_one.getActionToPerform(visitor, possibleActions)

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


