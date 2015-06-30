__author__ = 'bixlermike'

# Final verification 1 Jun 2015

import math
import numpy as np
import random as rn
import re

from exploChallenge.policies.ContextualBanditPolicy import ContextualBanditPolicy
from exploChallenge.policies.RidgeRegressor import RidgeRegressor

from exploChallenge.policies.NaiveBayesContextual import NaiveBayesContextual
from exploChallenge.policies.LinUCB import LinUCB
from exploChallenge.policies.eGreedyContextual import eGreedyContextual

def rargmax(x):
    m = np.amax(x)
    indices = np.nonzero(x == m)[0]
    return rn.choice(indices)

class EnsembleEAnnealingUpdateAllModel(ContextualBanditPolicy):


    def __init__(self):
        # Create an object from each class to use for ensemble model
        self.policy_one = NaiveBayesContextual()
        self.policy_two = LinUCB(0.1)
        self.policy_three = eGreedyContextual(0.1, RidgeRegressor(np.eye(136), np.zeros(136)))
        #self.policies = [self.policy_one, self.policy_two]
        self.policies = [self.policy_one, self.policy_two, self.policy_three]
        self.policy_scores = {}
        self.policy_counts = {}
        self.chosen_policy = None
        self.trials = 1

    #@Override
    def getActionToPerform(self, visitor, possibleActions):
        random_number = rn.random()
        self.epsilon = 1 / math.log(self.trials + 0.0000001)
        self.trials += 1
        #print "Epsilon is: " + str(self.epsilon)
        for i in self.policies:
            if str(i) not in self.policy_counts:
                self.policy_counts[str(i)] = 1.0
                self.policy_scores[str(i)] = 1.0


        if random_number > self.epsilon:
            print "Exploiting!"
            policy_values = [self.policy_scores[str(a)] for a in self.policies]
            self.chosen_policy = str(self.policies[rargmax(policy_values)])
            print self.policy_scores
        else:
            print "Exploring"
            self.chosen_policy =  str(rn.choice(self.policies))
            print self.policy_scores

        print "Chosen policy: " + str(self.chosen_policy) + "\n"
        if (re.match('<exploChallenge\.policies\.NaiveBayes',self.chosen_policy)):
            #print "Choice is NaiveBayes"
            return self.policy_one.getActionToPerform(visitor, possibleActions)
        elif (re.match('<exploChallenge\.policies\.LinUCB',self.chosen_policy)):
            #print "Choice is LinUCB"
            return self.policy_two.getActionToPerform(visitor, possibleActions)
        elif (re.match('<exploChallenge\.policies\.eGreedyContextual',self.chosen_policy)):
            #print "Choice is eGreedyContextual"
            return self.policy_three.getActionToPerform(visitor, possibleActions)

        else:
            print "Error with getActionToPerform in EnsembleEAnnealingUpdateAll!"
        return


    #@Override
    def updatePolicy(self, content, chosen_arm, reward, *possibleActions):

        #print self.policy_scores
        try:
            self.policy_one.updatePolicy(content, chosen_arm, reward)
        except:
            pass
        try:
            self.policy_two.updatePolicy(content, chosen_arm, reward)
        except:
            pass
        try:
            self.policy_three.updatePolicy(content, chosen_arm, reward)
        except:
            pass

        self.policy_counts[str(self.chosen_policy)] += 1

        new_value = ((self.policy_counts[str(self.chosen_policy)] - 1) / float(self.policy_counts[str(self.chosen_policy)])) * \
                    self.policy_scores[str(self.chosen_policy)] + reward * (1 / float(self.policy_counts[str(self.chosen_policy)]))
        self.policy_scores[str(self.chosen_policy)] = new_value
        print "Scores are: " + str(self.policy_scores)
        print "Counts are: " + str(self.policy_counts)

        return