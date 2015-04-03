__author__ = 'bixlermike'

import re
from scipy.stats import beta

from exploChallenge.eval.MyEvaluationPolicy import MyEvaluationPolicy
from exploChallenge.policies.ContextualBanditPolicy import ContextualBanditPolicy
from exploChallenge.policies.RandomPolicy import RandomPolicy
from exploChallenge.policies.eGreedy import eGreedy
from exploChallenge.policies.eAnnealing import eAnnealing
from exploChallenge.policies.Softmax import Softmax
from exploChallenge.policies.UCB1 import UCB1
from exploChallenge.policies.EXP3 import EXP3
from exploChallenge.eval.EvaluatorEXP3 import EvaluatorEXP3
from exploChallenge.policies.NaiveIII import Naive3
from exploChallenge.policies.Contextualclick import Contextualclick
from exploChallenge.policies.LinearBayes import LinearBayes

class EnsembleTestModel(ContextualBanditPolicy):


    def __init__(self, prior_alpha, prior_beta):
        # Create an object from each class to use for ensemble model
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.policy_one = eAnnealing()
        self.policy_two = Naive3()
        self.policies = [self.policy_one, self.policy_two]
        self.policy_one_count = 0
        self.policy_two_count = 0
        self.policy_counts = [self.policy_one_count, self.policy_two_count]
        self.policy_one_score = 0
        self.policy_two_score = 0
        self.policy_scores = [self.policy_one_score, self.policy_two_score]
        self.chosen_policy = None

    #@Override
    def getActionToPerform(self, visitor, possibleActions):
        sampled_theta = []
        for policy in self.policies:
            if (re.match('<exploChallenge\.policies\.eAnnealing',str(policy))):
                dist = beta(self.prior_alpha+self.policy_one_score,
                            self.prior_beta+self.policy_one_count-self.policy_one_score)
            else:
                dist = beta(self.prior_alpha+self.policy_two_score,
                            self.prior_beta+self.policy_two_count-self.policy_two_score)
            sampled_theta += [dist.rvs()]
        self.chosen_policy = str(self.policies[sampled_theta.index(max(sampled_theta))])

        if (re.match('<exploChallenge\.policies\.eAnnealing',self.chosen_policy)):
            #print "Choice is Annealing"
            return self.policy_one.getActionToPerform(visitor, possibleActions)
        else:
            return self.policy_two.getActionToPerform(visitor, possibleActions)

    #@Override
    def updatePolicy(self, content, chosen_arm, reward):
        #print "Updating policy " + str(self.chosen_policy)
        self.policy_one.updatePolicy(content, chosen_arm, reward)
        self.policy_two.updatePolicy(content, chosen_arm, reward)
        if (re.match('<exploChallenge\.policies\.eAnnealing',self.chosen_policy)):
            self.policy_one_count +=1
            if reward is True:
                self.policy_one_score = ((self.policy_one_count - 1) / float(self.policy_one_count)) * self.policy_one_score + (1 / float(self.policy_one_count))
        elif (re.match('<exploChallenge\.policies\.Naive',self.chosen_policy)):
            self.policy_two_count +=1
            if reward is True:
                self.policy_two_score = ((self.policy_two_count - 1) / float(self.policy_two_count)) * self.policy_two_score + (1 / float(self.policy_two_count))
                #print "Policy two score is: " + str(self.policy_two_score)
        else:
            print "Error in updatePolicy!"
        return