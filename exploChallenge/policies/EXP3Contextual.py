__author__ = 'bixlermike'

# NOT FINISHED

# Ridge regression estimates the weight coefficient vector as:
# Theta = (xT * x + I)^-1 * xT * y
# x: Feature vector, xT = transpose of feature vector, I = identity matrix, y = reward vector (all 0s or 1s)
# Here we use A = (xT * x + I)^-1 and B = (xT * y), so theta is then theta = A^-1 * b
# Get predictions for each article by multiplying weight vector (theta) * feature vector (x)

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

class EXP3Contextual(ContextualBanditPolicy):


    def __init__(self, gamma):
        self.gamma = gamma
        self.weights = {}

    def getGamma(self):
        return self.gamma

    def getActionToPerform(self, visitor, possibleActions):
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

