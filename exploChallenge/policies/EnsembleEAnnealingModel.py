__author__ = 'bixlermike'

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
from exploChallenge.policies.NaiveBayesContextual import NaiveBayesContextual
from exploChallenge.policies.Contextualclick import Contextualclick
from exploChallenge.policies.LinearBayes import LinearBayes

class EnsembleEAnnealingModel(ContextualBanditPolicy):


    def __init__(self):
        # Create an object from each class to use for ensemble model
        self.policy_one = eAnnealing()
        self.policy_two = Softmax(0.1)
        self.policy_three = UCB1()
        self.policy_four = NaiveBayesContextual()
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
        self.chosen_policy = None
        self.counts = {}

    #@Override
    def getActionToPerform(self, visitor, possibleActions):
        random_number = random.random()

        t = sum(self.counts) + 1
        self.epsilon = 1 / math.log(t + 0.0000001)

        if random_number > self.epsilon:
            if (self.policy_one_score == max(self.policy_scores)):
                self.chosen_policy =  str(self.policies[0])
            elif (self.policy_two_score == max(self.policy_scores)):
                self.chosen_policy = str(self.policies[1])
            elif (self.policy_three_score == max(self.policy_scores)):
                self.chosen_policy = str(self.policies[2])
            elif (self.policy_four_score == max(self.policy_scores)):
                self.chosen_policy = str(self.policies[3])
            else:
                print "Problem with choosing policy in EnsembleEAnnealing."
        else:
            self.chosen_policy =  str(random.choice(self.policies))

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
        self.policy_one.updatePolicy(content, chosen_arm, reward)
        self.policy_two.updatePolicy(content, chosen_arm, reward)
        self.policy_three.updatePolicy(content, chosen_arm, reward)
        self.policy_four.updatePolicy(content, chosen_arm, reward)

        if (re.match('<exploChallenge\.policies\.eAnnealing',self.chosen_policy)):
            self.policy_one_count +=1
            if reward is True:
                self.policy_one_score = ((self.policy_one_count - 1) / float(self.policy_one_count)) * self.policy_one_score + (1 / float(self.policy_one_count))
                #print "Policy one score is: " + str(self.policy_one_score)
        elif (re.match('<exploChallenge\.policies\.Softmax',self.chosen_policy)):
            self.policy_two_count +=1
            if reward is True:
                self.policy_two_score = ((self.policy_two_count - 1) / float(self.policy_two_count)) * self.policy_two_score + (1 / float(self.policy_two_count))
                #print "Policy two score is: " + str(self.policy_two_score)
        elif (re.match('<exploChallenge\.policies\.UCB1',self.chosen_policy)):
            self.policy_three_count +=1
            if reward is True:
                self.policy_three_score = ((self.policy_three_count - 1) / float(self.policy_three_count)) * self.policy_three_score + (1 / float(self.policy_three_count))
                #print "Policy three score is: " + str(self.policy_three_score)
        elif (re.match('<exploChallenge\.policies\.Naive',self.chosen_policy)):
            self.policy_four_count +=1
            if reward is True:
                self.policy_four_score = ((self.policy_four_count - 1) / float(self.policy_four_count)) * self.policy_four_score + (1 / float(self.policy_four_count))
                #print "Policy four score is: " + str(self.policy_four_score)
        else:
            print "Error in updatePolicy!"
        return