__author__ = 'bixlermike'

# Beta distribution sampling technique in getActionToPerform originally from https://www.chrisstucchio.com/blog/2013/bayesian_bandit.html

from scipy.stats import beta, stats

from exploChallenge.policies.ContextualBanditPolicy import ContextualBanditPolicy

class ThompsonSampling(ContextualBanditPolicy):

    def __init__(self, prior_alpha, prior_beta):
        ## Bayesian Bandit algorithm
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.successes = {}
        self.trials = {}

    def getPriors(self):
        return (self.prior_alpha, self.prior_beta)

    def getActionToPerform(self, visitor, possibleActions):
        sampled_theta = []
        for action in possibleActions:
            if action.getID() not in self.trials:
                self.successes[action.getID()] = 0.0
                self.trials[action.getID()] = 0.0
            #Construct beta distribution for posterior
            dist = beta(self.prior_alpha+self.successes[action.getID()],
                        self.prior_beta+self.trials[action.getID()]-self.successes[action.getID()])
            #Draw sample from beta distribution
            sampled_theta += [dist.rvs()]
        # Return the index of the sample with the largest value
        return possibleActions[sampled_theta.index(max(sampled_theta))]

    def updatePolicy(self, content, chosen_arm, reward):
        self.trials[chosen_arm.getID()] += 1
        n = self.trials[chosen_arm.getID()]
        if reward is True:
            self.successes[chosen_arm.getID()] += 1
        value = self.successes[chosen_arm.getID()]
        new_value = reward + (reward - value) / (n + 1)
        self.successes[chosen_arm.getID()] = new_value
        return


