import math
import numpy as np
import random
import re

from exploChallenge.eval.MyEvaluationPolicy import MyEvaluationPolicy
from exploChallenge.policies.ContextualBanditPolicy import ContextualBanditPolicy
from exploChallenge.policies.RandomPolicy import RandomPolicy
from exploChallenge.policies.eGreedy import eGreedy
from exploChallenge.policies.eAnnealing import eAnnealing
from exploChallenge.policies.Softmax import Softmax
from exploChallenge.policies.UCB1 import UCB1
from exploChallenge.policies.EXP3 import EXP3
from exploChallenge.eval.EvaluatorEXP3 import EvaluatorEXP3
from exploChallenge.policies.NaiveBayes3Contextual import NaiveBayes3Contextual
from exploChallenge.policies.Contextualclick import Contextualclick
from exploChallenge.policies.LinearBayes import LinearBayes

def rargmax(x):
    return np.argmax(x)

class EnsembleSoftmaxModel(ContextualBanditPolicy):

    def __init__(self, temp):
        # Create an object from each class to use for ensemble model
        self.policy_one = eAnnealing()
        self.policy_two = Softmax(0.1)
        self.policy_three = UCB1()
        self.policy_four = NaiveBayes3Contextual()
        self.policies = [self.policy_one, self.policy_two, self.policy_three, self.policy_four]
        self.policy_one_score = 0
        self.policy_two_score = 0
        self.policy_three_score = 0
        self.policy_four_score = 0
        self.policy_scores = [self.policy_one_score, self.policy_two_score, self.policy_three_score,
                              self.policy_four_score]
        self.policy_one_count = 0
        self.policy_two_count = 0
        self.policy_three_count = 0
        self.policy_four_count = 0
        self.policy_counts = [self.policy_one_count, self.policy_two_count, self.policy_three_count,
                              self.policy_four_count]
        self.policy_proportions = []
        self.policy_cutoffs = []
        self.chosen_policy = None
        self.temperature = temp

    #@Override
    def getActionToPerform(self, visitor, possibleActions):

        z = (math.exp(float(self.policy_one_score)/self.temperature) + math.exp(float(self.policy_two_score)/self.temperature) +
             math.exp(float(self.policy_three_score)/self.temperature) + math.exp(float(self.policy_four_score)/self.temperature))
        #print z
        self.policy_proportions = [math.exp(float(self.policy_one_score)/self.temperature)/z, math.exp(float(self.policy_two_score)/self.temperature)/z,
                                   math.exp(float(self.policy_three_score)/self.temperature)/z, math.exp(float(self.policy_four_score)/self.temperature)/z]
        self.policy_cutoffs = [float(self.policy_proportions[0]),float(self.policy_proportions[0]+self.policy_proportions[1]),
                           float(self.policy_proportions[0]+self.policy_proportions[1]+self.policy_proportions[2]),1.0]
        #print "Proportions " + str(self.policy_proportions)
        #print "Cutoffs " + str(self.policy_cutoffs)

        random_number = random.random()

        if (random_number < self.policy_cutoffs[0]):
            self.chosen_policy =  str(self.policies[0])
        elif (random_number < self.policy_cutoffs[1]):
            self.chosen_policy = str(self.policies[1])
        elif (random_number < self.policy_cutoffs[2]):
            self.chosen_policy = str(self.policies[2])
        else:
            self.chosen_policy = str(self.policies[3])
        #print "Policy chosen was " + str(self.chosen_policy)

        if (re.match('<exploChallenge\.policies\.eAnnealing',self.chosen_policy)):
            #print "Choice is Annealing"
            return self.policy_one.getActionToPerform(visitor, possibleActions)
        elif (re.match('<exploChallenge\.policies\.Softmax',self.chosen_policy)):
            #print "Choice is Softmax"
            return self.policy_two.getActionToPerform(visitor, possibleActions)
        elif (re.match('<exploChallenge\.policies\.UCB1',self.chosen_policy)):
            #print "Choice is UCB1"
            return self.policy_three.getActionToPerform(visitor, possibleActions)
        elif (re.match('<exploChallenge\.policies\.Naive',self.chosen_policy)):
            #print "Choice is Naive3"
            return self.policy_four.getActionToPerform(visitor, possibleActions)
        else:
            print "Error in getActionToPerform!"
            return

    #@Override
    def updatePolicy(self, content, chosen_arm, reward):
        #print "Updating policy " + str(self.chosen_policy)
        if (re.match('<exploChallenge\.policies\.eAnnealing',self.chosen_policy)):
            self.policy_one.updatePolicy(content, chosen_arm, reward)
            self.policy_one_count +=1
            if reward is True:
                self.policy_one_score = ((self.policy_one_count - 1) / float(self.policy_one_count)) * self.policy_one_score + (1 / float(self.policy_one_count))
            #print "Policy one score is: " + str(self.policy_one_score)
        elif (re.match('<exploChallenge\.policies\.Softmax',self.chosen_policy)):
            self.policy_two.updatePolicy(content, chosen_arm, reward)
            self.policy_two_count +=1
            if reward is True:
                self.policy_two_score = ((self.policy_two_count - 1) / float(self.policy_two_count)) * self.policy_two_score + (1 / float(self.policy_two_count))
            #print "Policy two score is: " + str(self.policy_two_score)
        elif (re.match('<exploChallenge\.policies\.UCB1',self.chosen_policy)):
            self.policy_three.updatePolicy(content, chosen_arm, reward)
            self.policy_three_count +=1
            if reward is True:
                self.policy_three_score = ((self.policy_three_count - 1) / float(self.policy_three_count)) * self.policy_three_score + (1 / float(self.policy_three_count))
            #print "Policy three score is: " + str(self.policy_three_score)
        elif (re.match('<exploChallenge\.policies\.Naive',self.chosen_policy)):
            self.policy_four.updatePolicy(content, chosen_arm, reward)
            self.policy_four_count +=1
            if reward is True:
                self.policy_four_score = ((self.policy_four_count - 1) / float(self.policy_four_count)) * self.policy_four_score + (1 / float(self.policy_four_count))
            #print "Policy four score is: " + str(self.policy_four_score)
        else:
            print "Error in updatePolicy!"
        return

