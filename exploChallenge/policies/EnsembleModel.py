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
from exploChallenge.policies.Mostclick import Mostclick
from exploChallenge.policies.Clickrate import Clickrate
from exploChallenge.policies.NaiveIII import Naive3
from exploChallenge.policies.Contextualclick import Contextualclick
from exploChallenge.policies.LinearBayes import LinearBayes
from exploChallenge.policies.LinearBayesFtu import LinearBayesFtu

def rargmax(x):
    return np.argmax(x)

class EnsembleModel(ContextualBanditPolicy):



    def __init__(self):
        # Create an object from each class to use for ensemble model
        self.policy_one = eAnnealing()
        self.policy_two = Softmax(0.1)
        self.policy_three = UCB1()
        self.policy_four = Naive3()
        self.policies = [self.policy_one, self.policy_two, self.policy_three, self.policy_four]
        self.policy_one_score = 0
        self.policy_two_score = 0
        self.policy_three_score = 0
        self.policy_four_score = 0
        self.policy_scores = [self.policy_one_score, self.policy_two_score, self.policy_three_score,
                              self.policy_four_score]
        self.policy_rewards = [0, 0, 0, 0]
        self.chosen_policy = None

    #@Override
    def getActionToPerform(self, visitor, possibleActions):
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
        if (re.match('<exploChallenge\.policies\.eAnnealing',self.chosen_policy)):
            self.policy_one.updatePolicy(content, chosen_arm, reward)
        elif (re.match('<exploChallenge\.policies\.Softmax',self.chosen_policy)):
            self.policy_two.updatePolicy(content, chosen_arm, reward)
        elif (re.match('<exploChallenge\.policies\.UCB1',self.chosen_policy)):
            self.policy_three.updatePolicy(content, chosen_arm, reward)
        elif (re.match('<exploChallenge\.policies\.Naive',self.chosen_policy)):
            self.policy_four.updatePolicy(content, chosen_arm, reward)
        else:
            print "Error in updatePolicy!"
        return

