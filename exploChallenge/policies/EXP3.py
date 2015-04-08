__original_author__ = 'dai.shi'
__revised__ = 'bixlermike'

# Algorithm psuedocode at http://jeremykun.com/2013/11/08/adversarial-bandits-and-the-exp3-algorithm/
# Originally defined in http://cseweb.ucsd.edu/~yfreund/papers/bandits.pdf

# Final verification 8 Apr 2015

import random
import operator
import math

from exploChallenge.policies.ContextualBanditPolicy import ContextualBanditPolicy

def categorical_draw(x):
    z = random.random()
    cum_prob = 0.0
    for p in x:
        prob = x[p]
        cum_prob += prob
        if cum_prob > z:
            return p
    return x.iterkeys().next()

class EXP3(ContextualBanditPolicy):


    def __init__(self, gamma):
        self.gamma = gamma
        self.weights = {}

    def getGamma(self):
        return self.gamma

    def getActionToPerform(self, visitor,possibleActions):
        arm_weights ={}
        arm_probs = {}
        for action in possibleActions:
            if action.getID() not in self.weights:
                self.weights[action.getID()] = 1
            arm_weights[action.getID()] = 0

        for w in arm_weights:
            arm_weights[w] = self.weights[w]

        # Normalization factor
        total_weight = sum(arm_weights[w] for w in arm_weights)

        # Set the arm probabilities
        for v in arm_weights:
            arm_probs[v]= (1 - self.gamma) * (self.weights[v] / total_weight)
            arm_probs[v] = arm_probs[v] + (self.gamma) * (1.0 / float(len(arm_weights)))

        id = categorical_draw(arm_probs)
        for action in possibleActions:
            if action.getID() == id:
                return action


    def updatePolicy(self, content, chosen_arm, reward, possibleActions):
        arm_weights ={}
        arm_probs = {}
        for action in possibleActions:
            if action.getID() not in self.weights:
                self.weights[action.getID()] = 1
            arm_weights[action.getID()] = 0

        for weight in arm_weights:
            arm_weights[weight] = self.weights[weight]

        # Normalization factor
        total_weight = sum(arm_weights[w] for w in arm_weights)

        # Update the arm probabilities
        for v in arm_weights:
            arm_probs[v]= (1 - self.gamma) * (self.weights[v] / total_weight)
            arm_probs[v] = arm_probs[v] + (self.gamma) * (1.0 / float(len(arm_weights)))

        # Define the estimated reward of chosen arm
        x = reward / arm_probs[chosen_arm.getID()]

        # Update the weight of the chosen arm
        growth_factor = math.exp((self.gamma / len(arm_probs)) * x)
        self.weights[chosen_arm.getID()] = self.weights[chosen_arm.getID()] * growth_factor

