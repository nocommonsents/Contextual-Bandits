__author__ = 'bixlermike'

import math
import numpy as np
import random as rn
import re

from exploChallenge.policies.ContextualBanditPolicy import ContextualBanditPolicy
from exploChallenge.policies.eAnnealing import eAnnealing
from exploChallenge.policies.MostCTR import MostCTR
from exploChallenge.policies.NaiveBayesContextual import NaiveBayesContextual

def rargmax(x):
    m = np.amax(x)
    indices = np.nonzero(x == m)[0]
    return rn.choice(indices)

#idx_max = max(enumerate(x), key=lambda x:x[1])[0]

class EnsembleEAnnealingUpdateAllModel(ContextualBanditPolicy):


    def __init__(self):
        # Create an object from each class to use for ensemble model
        self.policy_one = eAnnealing()
        self.policy_two = MostCTR()
        self.policy_three = NaiveBayesContextual()
        self.policies = [self.policy_one, self.policy_two, self.policy_three]
        # self.policy_one_score = 0.01
        # self.policy_two_score = 0.01
        # self.policy_three_score = 0.01
        # self.policy_scores = []
        self.policy_scores = {}
        # self.policy_one_count = 0.001
        # self.policy_two_count = 0.001
        # self.policy_three_count = 0.001
        # self.policy_counts = []
        self.policy_counts = {}
        self.chosen_policy = None
        self.trials = 1

    #@Override
    def getActionToPerform(self, visitor, possibleActions):
        arm_values = {}
        random_number = rn.random()
        self.epsilon = 1 / math.log(self.trials + 0.0000001)
        self.trials += 1
        #print "Epsilon is: " + str(self.epsilon)
        for i in self.policies:
            if str(i) not in self.policy_counts:
                self.policy_counts[str(i)] = 0.01
                self.policy_scores[str(i)] = 0.01


        if random_number > self.epsilon:
           # print "Exploiting!"
            policy_values = [self.policy_scores[a] for a in self.policy_scores]
            self.chosen_policy = str(self.policies[rargmax(policy_values)])
            #print self.policy_scores
        else:
            #print "Exploring"
            self.chosen_policy =  str(rn.choice(self.policies))
            #print self.policy_scores

        print "Chosen policy: " + str(self.chosen_policy)
        if (re.match('<exploChallenge\.policies\.eAnnealing',self.chosen_policy)):
            #print "Choice is EAnnealing"
            return self.policy_one.getActionToPerform(visitor, possibleActions)
        elif (re.match('<exploChallenge\.policies\.MostCTR',self.chosen_policy)):
            #print "Choice is MostCTR"
            return self.policy_two.getActionToPerform(visitor, possibleActions)
        elif (re.match('<exploChallenge\.policies\.NaiveBayes',self.chosen_policy)):
            #print "Choice is NaiveBayes"
            return self.policy_three.getActionToPerform(visitor, possibleActions)
        else:
            print "Error with getActionToPerform in EnsembleEAnnealingUpdateAll!"
        return


    #@Override
    def updatePolicy(self, content, chosen_arm, reward, *possibleActions):

        print self.policy_scores

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

        if reward is True:
            self.policy_scores[str(self.chosen_policy)] = ((self.policy_counts[str(self.chosen_policy)] - 1) / float(self.policy_counts[str(self.chosen_policy)])) * \
                                                          self.policy_scores[str(self.chosen_policy)] + (1 / float(self.policy_counts[str(self.chosen_policy)]))
            print "Scores are: " + str(self.policy_scores)
            print "Counts are: " + str(self.policy_counts)
        # if (re.match('<exploChallenge\.policies\.eAnnealing',self.chosen_policy)):
        #     self.policy_one_count +=1
        #     if reward is True:
        #         self.policy_one_score = ((self.policy_one_count - 1) / float(self.policy_one_count)) * self.policy_one_score + (1 / float(self.policy_one_count))
        # elif (re.match('<exploChallenge\.policies\.MostCTR',self.chosen_policy)):
        #     self.policy_two_count +=1
        #     if reward is True:
        #         self.policy_two_score = ((self.policy_two_count - 1) / float(self.policy_two_count)) * self.policy_two_score + (1 / float(self.policy_two_count))
        # elif (re.match('<exploChallenge\.policies\.NaiveBayes',self.chosen_policy)):
        #     self.policy_three_count +=1
        #     if reward is True:
        #         self.policy_three_score = ((self.policy_three_count - 1) / float(self.policy_three_count)) * self.policy_three_score + (1 / float(self.policy_three_count))
        # else:
        #     print "Error with updatePolicy in EnsembleEAnnealingUpdateAll!"

        return