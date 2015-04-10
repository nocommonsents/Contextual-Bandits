__author__ = 'bixlermike'

import math
import random
import re

from exploChallenge.policies.ContextualBanditPolicy import ContextualBanditPolicy
from exploChallenge.policies.eAnnealingContextual import eAnnealingContextual
from exploChallenge.policies.NaiveBayesContextual import NaiveBayesContextual
from exploChallenge.policies.LinUCB import LinUCB

class EnsembleTestModel(ContextualBanditPolicy):


    def __init__(self, regressor):
        # Create an object from each class to use for ensemble model
        self.regressor = regressor
        self.trials = 1
        self.policy_one = eAnnealingContextual()
        self.policy_two = NaiveBayesContextual()
        self.policy_three = LinUCB()
        self.policies = [self.policy_one, self.policy_two, self.policy_three]
        self.policy_one_count = 0
        self.policy_two_count = 0
        self.policy_three_count = 0
        self.policy_counts = [self.policy_one_count, self.policy_two_count, self.policy_three_count]
        self.policy_one_score = 0
        self.policy_two_score = 0
        self.policy_three_score = 0
        self.policy_scores = [self.policy_one_score, self.policy_two_score, self.policy_three_score]
        self.chosen_policy = None

    #@Override
    def getActionToPerform(self, visitor, possibleActions):
        # Code to choose policy here
        random_number = random.random()

        self.epsilon = 1 / math.log(self.trials + 0.0000001)
        self.trials += 1
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
        elif (re.match('<exploChallenge\.policies\.Naive',self.chosen_policy)):
            #print "Choice is Softmax"
            return self.policy_two.getActionToPerform(visitor, possibleActions)
        elif (re.match('<exploChallenge\.policies\.LinUCB',self.chosen_policy)):
            #print "Choice is UCB1"
            return self.policy_three.getActionToPerform(visitor, possibleActions)
        else:
            print "Error in getActionToPerform!"
        return

    #@Override
    def updatePolicy(self, content, chosen_arm, reward):
        #print "Updating policy " + str(self.chosen_policy)
        self.counts += 1
        self.policy_one.updatePolicy(content, chosen_arm, reward)
        self.policy_two.updatePolicy(content, chosen_arm, reward)
        self.policy_three.updatePolicy(content, chosen_arm, reward)

        if (re.match('<exploChallenge\.policies\.eAnnealing',self.chosen_policy)):
            self.policy_one_count +=1
            if reward is True:
                self.policy_one_score = ((self.policy_one_count - 1) / float(self.policy_one_count)) * self.policy_one_score + (1 / float(self.policy_one_count))
        elif (re.match('<exploChallenge\.policies\.Naive',self.chosen_policy)):
            self.policy_two_count +=1
            if reward is True:
                self.policy_two_score = ((self.policy_two_count - 1) / float(self.policy_two_count)) * self.policy_two_score + (1 / float(self.policy_two_count))
                #print "Policy two score is: " + str(self.policy_two_score)
        elif (re.match('<exploChallenge\.policies\.LinUCB',self.chosen_policy)):
            self.policy_three_count +=1
            if reward is True:
                self.policy_three_score = ((self.policy_three_count - 1) / float(self.policy_three_count)) * self.policy_three_score + (1 / float(self.policy_three_count))
                #print "Policy two score is: " + str(self.policy_two_score)
        else:
            print "Error in updatePolicy!"
        return