import math
import numpy as np
import random
import re

from exploChallenge.policies.ContextualBanditPolicy import ContextualBanditPolicy
from exploChallenge.policies.eAnnealing import eAnnealing
from exploChallenge.policies.MostCTR import MostCTR
from exploChallenge.policies.NaiveBayesContextual import NaiveBayesContextual

def rargmax(x):
    return np.argmax(x)

class EnsembleSoftmaxUpdateAllModel(ContextualBanditPolicy):

    def __init__(self, temp):
        # Create an object from each class to use for ensemble model
        self.policy_one = eAnnealing()
        self.policy_two = MostCTR()
        self.policy_three = NaiveBayesContextual()
        self.policies = [self.policy_one, self.policy_two, self.policy_three]
        self.policy_one_score = 0.01
        self.policy_two_score = 0.01
        self.policy_three_score = 0.01
        self.policy_scores = []
        self.policy_one_count = 0
        self.policy_two_count = 0
        self.policy_three_count = 0
        self.policy_counts = []
        self.policy_proportions = []
        self.policy_cutoffs = []
        self.chosen_policy = None
        self.temperature = temp

    def getTemp(self):
        return self.temperature

    #@Override
    def getActionToPerform(self, visitor, possibleActions):
        random_number = random.random()
        self.policy_scores = [self.policy_one_score, self.policy_two_score, self.policy_three_score]
        z = (math.exp(float(self.policy_one_score)/self.temperature) + math.exp(float(self.policy_two_score)/self.temperature) +
             math.exp(float(self.policy_three_score)/self.temperature))
        #print z

        self.policy_proportions = [math.exp(float(self.policy_one_score)/self.temperature)/z, math.exp(float(self.policy_two_score)/self.temperature)/z,
                                   math.exp(float(self.policy_three_score)/self.temperature)/z]
        self.policy_cutoffs = [float(self.policy_proportions[0]),float(self.policy_proportions[0]+self.policy_proportions[1]), 1.0]
        #print "Proportions " + str(self.policy_proportions)
        #print "Cutoffs " + str(self.policy_cutoffs)


        if (random_number < self.policy_cutoffs[0]):
            self.chosen_policy =  str(self.policies[0])
        elif (random_number < self.policy_cutoffs[1]):
            self.chosen_policy = str(self.policies[1])
        else:
            self.chosen_policy = str(self.policies[2])
        #print "Policy chosen was " + str(self.chosen_policy)

        if (re.match('<exploChallenge\.policies\.eAnnealing',self.chosen_policy)):
            return self.policy_one.getActionToPerform(visitor, possibleActions)
        elif (re.match('<exploChallenge\.policies\.MostCTR',self.chosen_policy)):
            return self.policy_two.getActionToPerform(visitor, possibleActions)
        elif (re.match('<exploChallenge\.policies\.Naive',self.chosen_policy)):
            return self.policy_three.getActionToPerform(visitor, possibleActions)
        else:
            print "Error in getActionToPerform!"
            return

    #@Override
    def updatePolicy(self, content, chosen_arm, reward):
        #print "Updating policy " + str(self.chosen_policy)
        self.policy_counts = [self.policy_one_count, self.policy_two_count, self.policy_three_count]
        print "Scores are: " + str(self.policy_scores)
        try:
            self.policy_one.updatePolicy(content, chosen_arm, reward)
            self.policy_two.updatePolicy(content, chosen_arm, reward)
            self.policy_three.updatePolicy(content, chosen_arm, reward)
        except:
            pass

        if (re.match('<exploChallenge\.policies\.eAnnealing',self.chosen_policy)):
            self.policy_one_count +=1
            if reward is True:
                self.policy_one_score = ((self.policy_one_count - 1) / float(self.policy_one_count)) * self.policy_one_score + (1 / float(self.policy_one_count))
            #print "Policy one score is: " + str(self.policy_one_score)
        elif (re.match('<exploChallenge\.policies\.MostCTR',self.chosen_policy)):
            self.policy_two_count +=1
            if reward is True:
                self.policy_two_score = ((self.policy_two_count - 1) / float(self.policy_two_count)) * self.policy_two_score + (1 / float(self.policy_two_count))
            #print "Policy two score is: " + str(self.policy_two_score)
        elif (re.match('<exploChallenge\.policies\.Naive',self.chosen_policy)):
            self.policy_three_count +=1
            if reward is True:
                self.policy_three_score = ((self.policy_three_count - 1) / float(self.policy_three_count)) * self.policy_three_score + (1 / float(self.policy_three_count))
            #print "Policy three score is: " + str(self.policy_three_score)
        else:
            print "Error in updatePolicy!"
        return

