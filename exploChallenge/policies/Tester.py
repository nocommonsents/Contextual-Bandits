__author__ = 'bixlermike'

from scipy.stats import beta

from exploChallenge.policies.ContextualBanditPolicy import ContextualBanditPolicy

class Tester(ContextualBanditPolicy):

    def __init__(self, prior_alpha, prior_beta):
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.successes = {}
        self.trials = {}

    def getPriors(self):
        return (self.prior_alpha, self.prior_beta)

    #def get_recommendation(self):
    #    sampled_theta = []
    #    for i in range(self.num_options):
    #        #Construct beta distribution for posterior
    #        dist = beta(self.prior[0]+self.successes[i],
    #                    self.prior[1]+self.trials[i]-self.successes[i])
    #        #Draw sample from beta distribution
    #        sampled_theta += [ dist.rvs() ]
    #    # Return the index of the sample with the largest value
    #    return sampled_theta.index( max(sampled_theta) )

    def getActionToPerform(self, visitor, possibleActions):
        sampled_theta = []
        for action in possibleActions:
            if action.getID() not in self.trials:
                self.successes[action.getID()] = 0.0
                self.trials[action.getID()] = 0.0
            #Construct beta distribution for posterior
            #dist = beta(self.prior[0]+self.successes[action.getID()],
            #            self.prior[1]+self.trials[action.getID()]-self.successes[action.getID()])
            dist = beta(self.prior_alpha,self.prior_beta)
            #Draw sample from beta distribution
            sampled_theta += [dist.rvs()]
        # Return the index of the sample with the largest value
        return possibleActions[sampled_theta.index(max(sampled_theta))]

    #def add_result(self, trial_id, success):
    #    self.trials[trial_id] = self.trials[trial_id] + 1
    #    if (success):
    #        self.successes[trial_id] = self.successes[trial_id] + 1

    def updatePolicy(self, content, chosen_arm, reward):
        self.trials[chosen_arm.getID()] += 1
        n = self.trials[chosen_arm.getID()]
        self.successes[chosen_arm.getID()] += 1
        value = self.successes[chosen_arm.getID()]
        new_value = reward + (reward - value) / (n + 1)
        self.successes[chosen_arm.getID()] = new_value
        return


