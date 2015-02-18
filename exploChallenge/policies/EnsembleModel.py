import numpy as np
import random

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
        self.policy_one = RandomPolicy()
        self.policy_two = eGreedy(0.1)
        self.policy_three = eAnnealing()
        self.policy_four = Softmax(0.1)
        self.policy_five = UCB1()
        self.policy_six = EXP3(0.5)
        self.policy_seven = Mostclick()
        self.policy_eight = Naive3()
        self.policies = [self.policy_one, self.policy_two, self.policy_three, self.policy_four,
                self.policy_five, self.policy_six, self.policy_seven, self.policy_eight]
        return

    #@Override

    def getPolicyToPerform(self):
        # Add more complex logic to policy selection later...
        self.chosenPolicy =  random.choice(self.policies)
        return

    def getActionToPerform(self, ctx, possibleActions):
        self.chosenPolicy.getActionToPerform(self, ctx, possibleActions)
        return

    #@Override
    def updatePolicy(self, content, chosen_arm, reward):
        self.chosenPolicy.updatePolicy(self, content, chosen_arm, reward)
        return

