__author__ = 'bixlermike'

import math
import random
import re

from exploChallenge.policies.ContextualBanditPolicy import ContextualBanditPolicy
from exploChallenge.policies.eAnnealing import eAnnealing
from exploChallenge.policies.MostCTR import MostCTR
from exploChallenge.policies.NaiveBayesContextual import NaiveBayesContextual

class EnsembleEAnnealingUpdateAllModel(ContextualBanditPolicy):


    def __init__(self):
        # Create an object from each class to use for ensemble model
        self.policy_one = eAnnealing()
        self.policy_two = MostCTR()
        self.policy_three = NaiveBayesContextual()
        self.policies = [self.policy_one, self.policy_two, self.policy_three]
        self.policy_one_score = 0
        self.policy_two_score = 0.00001
        self.policy_three_score = 0
        self.policy_scores = []
        self.policy_one_count = 0
        self.policy_two_count = 0
        self.policy_three_count = 0
        self.policy_counts = []
        self.chosen_policy = None
        self.trials = 1

    #@Override
    def getActionToPerform(self, visitor, possibleActions):
        random_number = random.random()
        self.epsilon = 1 / math.log(self.trials + 0.0000001)
        self.trials += 1
        #print "Epsilon is: " + str(self.epsilon)

        if random_number > self.epsilon:
            self.policy_scores = [self.policy_one_score, self.policy_two_score, self.policy_three_score]

            #print "Exploiting - Scores are: " + str(self.policy_scores)
            if (self.policy_one_score == max(self.policy_scores)):
                self.chosen_policy =  str(self.policies[0])
            elif (self.policy_two_score == max(self.policy_scores)):
                self.chosen_policy = str(self.policies[1])
            elif (self.policy_three_score == max(self.policy_scores)):
                self.chosen_policy = str(self.policies[2])
            else:
                print "Problem with choosing policy in EnsembleEAnnealingUpdateAll."
        else:
            #print "Exploring"
            self.chosen_policy =  str(random.choice(self.policies))

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
    def updatePolicy(self, content, chosen_arm, reward):
        self.policy_one.updatePolicy(content, chosen_arm, reward)
        self.policy_two.updatePolicy(content, chosen_arm, reward)
        self.policy_three.updatePolicy(content, chosen_arm, reward)

        self.policy_counts = [self.policy_one_count, self.policy_two_count, self.policy_three_count]

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
        elif (re.match('<exploChallenge\.policies\.NaiveBayes',self.chosen_policy)):
            self.policy_three_count +=1
            if reward is True:
                self.policy_three_score = ((self.policy_three_count - 1) / float(self.policy_three_count)) * self.policy_three_score + (1 / float(self.policy_three_count))
                #print "Policy three score is: " + str(self.policy_three_score)
        else:
            print "Error with updatePolicy in EnsembleEAnnealingUpdateAll!"
        return