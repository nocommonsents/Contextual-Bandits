__original_author__ = 'dai.shi'
__revised__ = 'bixlermike'

import numpy as np
import random as rn

from exploChallenge.policies.ContextualBanditPolicy import ContextualBanditPolicy

# In case of tie, picks one from the set of best articles at random
def rargmax(x):
    m = np.amax(x)
    indices = np.nonzero(x == m)[0]
    return rn.choice(indices)

class MostClicked(ContextualBanditPolicy):

    def __init__(self):
        self.clicks = {}
        return

    def getActionToPerform(self, visitor, possibleActions):
        for action in possibleActions:
            if action.getID() not in self.clicks:
                self.clicks[action.getID()] = 0

        psvalues = [ self.clicks[a.getID()] for a in possibleActions]
        action = possibleActions[rargmax(psvalues)]
        return action

    def updatePolicy(self, content, chosen_arm, reward):
        if reward is True:
            self.clicks[chosen_arm.getID()] += 1
        return


