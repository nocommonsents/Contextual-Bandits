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
        self.current_policy = None
        self.total_counts = 0
        self.counts = {}
        self.values = {}

        for num in range(0,24):
            if (num < 10):
                num = "0" + str(num)
            key1 = str("MostCTR") + " " + str(num)
            key2 = str("NaiveBayes") + " " + str(num)
            self.counts[key1] = 0
            self.values[key1] = 0
            self.counts[key2] = 0
            self.values[key2] = 0
        self.timestamp = 0

    #@Override
    def getActionToPerform(self, visitor, possibleActions):

        self.timestamp = visitor.getTimestamp()
        temp = random.random()
        if (temp <= 0.5):
            self.current_policy = "MostCTR"
            return self.policy_one.getActionToPerform(visitor, possibleActions)
        else:
            self.current_policy = "NaiveBayes"
            return self.policy_two.getActionToPerform(visitor, possibleActions)

        #@Override
    def updatePolicy(self, content, chosen_arm, reward, *possibleActions):

        self.total_counts += 1
        #print self.timestamp
        t = datetime.datetime.fromtimestamp(float(self.timestamp))
        fmt = "%Y-%m-%d %H:%M:%S"
        fmt_hours = "%H"
        formatted_time = t.strftime(fmt)
        formatted_time_hours = t.strftime(fmt_hours)
        #print formatted_time_hours

        try:
            if (self.current_policy == "MostCTR"):
                key = str("MostCTR") + " " + str(formatted_time_hours)
                self.counts[key] += 1
                n = self.counts[key]
                value = self.values[key]
                # Calculate revised AER of this arm
                new_value = ((n-1) / float(n)) * value + (1/float(n)) * reward
                self.values[key] = new_value
            self.policy_one.updatePolicy(content, chosen_arm, reward)

        except:
            pass
        try:
            if (self.current_policy == "NaiveBayes"):
                key = str("NaiveBayes") + " " + str(formatted_time_hours)
                self.counts[key] += 1
                n = self.counts[key]
                value = self.values[key]
                # Calculate revised AER of this arm
                new_value = ((n-1) / float(n)) * value + (1/float(n)) * reward
                self.values[key] = new_value
            self.policy_two.updatePolicy(content, chosen_arm, reward)
        except:
            pass

        if (self.total_counts % 2000 == 0):
            for i in self.values:
                print i, self.values[i]
        return


