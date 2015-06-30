__author__ = 'bixlermike'
# Derived from ideas from Dai Shi and John White

# Final verification 8 Apr 2015

import random
import math

from exploChallenge.policies.ContextualBanditPolicy import ContextualBanditPolicy

def categorical_draw(probs):
    z = random.random()
    cum_prob = 0.0
    for p in probs:
        prob = probs[p]
        cum_prob += prob
        if cum_prob > z:
            return p
    return probs.iterkeys().next()

class Softmax(ContextualBanditPolicy):


    def __init__(self, temp):
        self.temperature = temp
        self.counts = {}
        self.values = {}

    def getTemp(self):
        return self.temperature

    def getActionToPerform(self, visitor, possibleActions):
        arm_values = {}
        arm_probs = {}

        for action in possibleActions:
            if action.getID() not in self.counts:
                self.counts[action.getID()] = 0
                self.values[action.getID()] = 0

        arm_scores = [math.exp(self.values[a.getID()]/self.temperature) for a in possibleActions]
        #print "Arm scores: " + str(arm_scores)
        # Normalization factor z
        z = sum(arm_scores)
        # Calculate the probability that each arm will be selected
        for v in possibleActions:
            arm_probs[v.getID()]= math.exp(self.values[v.getID()] / self.temperature) / z
        #print "Arm probs: " + str(arm_probs)
        # Generate random number and see which bin it falls into to select arm
        id = categorical_draw(arm_probs)
        for action in possibleActions:
            if action.getID() == id:
                return action

    def updatePolicy(self, content, chosen_arm, reward):
        try:
            self.counts[chosen_arm.getID()] = self.counts[chosen_arm.getID()] + 1
            n = self.counts[chosen_arm.getID()]
            value = self.values[chosen_arm.getID()]
            new_value = ((n - 1) / float(n)) * value + reward * (1 / float(n))
            self.values[chosen_arm.getID()] = new_value
            return
        except:
            return



