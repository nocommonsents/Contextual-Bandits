

from exploChallenge.policies.ContextualBanditPolicy import ContextualBanditPolicy
from exploChallenge.policies.RandomPolicy import RandomPolicy
from exploChallenge.policies.eGreedy import eGreedy
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

class EnsembleModel(ContextualBanditPolicy):


    def __init__(self):
        pass

    #@Override

    def getPolicyToPerform(selfs, ctx, possiblePolicies):
        pass

    def getActionToPerform(self, ctx, possibleActions):
        randomIndex = random.randint(0, len(possibleActions) - 1)
        return possibleActions[randomIndex]

    #@Override
    def updatePolicy(self, c, a, reward):
        pass
