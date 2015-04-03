#-------------------------------------------------------------------------------
# Copyright (c) 2012 Jose Antonio Martin H..
# All rights reserved. This program and the accompanying materials
# are made available under the terms of the GNU Public License v3.0
# which accompanies this distribution, and is available at
# http://www.gnu.org/licenses/gpl.html
#
# Contributors:
#     Jose Antonio Martin H. - Translation to Python from Java
#     Jeremie Mary - very minor adaptation for the challenge
#-------------------------------------------------------------------------------
#package exploChallenge;

#import java.io.FileNotFoundException;
import os
import sys
import time
import numpy as np


from myPolicy.MyPolicy import MyPolicy
from exploChallenge.eval.EvaluationPolicy import EvaluationPolicy
from exploChallenge.eval.Evaluator import Evaluator
from exploChallenge.eval.MyEvaluationPolicy import MyEvaluationPolicy
from exploChallenge.logs.FromFileLogLineGenerator import FromFileLogLineGenerator
from exploChallenge.logs.yahoo.YahooArticle import YahooArticle
from exploChallenge.logs.yahoo.YahooLogLineReader import YahooLogLineReader
from exploChallenge.logs.yahoo.YahooVisitor import YahooVisitor
from exploChallenge.policies.ContextualBanditPolicy import ContextualBanditPolicy
from exploChallenge.policies.RandomPolicy import RandomPolicy
from exploChallenge.policies.eGreedy import eGreedy
from exploChallenge.policies.eGreedyContextual import eGreedyContextual
from exploChallenge.policies.eAnnealing import eAnnealing
from exploChallenge.policies.Softmax import Softmax
from exploChallenge.policies.UCB1 import UCB1
from exploChallenge.policies.EXP3 import EXP3
from exploChallenge.eval.EvaluatorEXP3 import EvaluatorEXP3
from exploChallenge.policies.NaiveIII import Naive3
from exploChallenge.policies.Contextualclick import Contextualclick
from exploChallenge.policies.RidgeRegressor import RidgeRegressor
from exploChallenge.policies.GMPolicy import GMPolicy
from exploChallenge.policies.LinUCB import LinUCB
from exploChallenge.policies.LinearBayes import LinearBayes
from exploChallenge.policies.EnsembleRandomModel import EnsembleRandomModel
from exploChallenge.policies.EnsembleRandomModelUpdateAll import EnsembleRandomModelUpdateAll
from exploChallenge.policies.EnsembleSoftmaxModel import EnsembleSoftmaxModel
from exploChallenge.policies.EnsembleEAnnealingModel import EnsembleEAnnealingModel
from exploChallenge.policies.EnsembleEAnnealingUpdateAllModel import EnsembleEAnnealingUpdateAllModel
from exploChallenge.policies.EnsembleTestModel import EnsembleTestModel

from exploChallenge.policies.BayesianBandit import BayesianBandit

from time import strftime







class Main:

    """
     * @param argv: a Python list containing the command line args
     * a usual, argv[0] is the full path name of this program.
     * @throws FileNotFoundException -> IOError?
    """
    def main(self, argv = sys.argv):

        currentTimeMillis = lambda:  int(round(time.time() * 1000))
        t = currentTimeMillis()
        reader = None

        ## Create file to write output to..."a+" option appends
        #outputFile = open("banditOutputsEnsembleRandomUpdateAllWithTime.txt", "a+")
        outputFile = open("testing.txt", "a+")


        try:
            inputFile = "/Users/bixlermike/Contextual-Bandits/exploChallenge/first_10000_lines.txt"
            #inputFile = "/Users/bixlermike/Contextual-Bandits/exploChallenge/ydata-fp-td-clicks-v2_0.20111002-08.txt"
            inputFileShort = "y"    # Yahoo! data = "y"
            reader = YahooLogLineReader(inputFile, 136)
            logStep = 1
        except:
            print "Problem with input file."
            logStep = 1


        generator = FromFileLogLineGenerator(reader)

        outputFile.write("Simulation started at: " + strftime("%Y-%m-%d %H:%M:%S") + "\n")
        outputFile.write("Input file: " + os.path.basename(inputFile) + "\n")

        ## Pick a single contextual bandit algorithm and corresponding output file

        #policy = BayesianBandit(1.0, 1.0)
        #policyName = "BayesianBandit" + str(policy.getPriors())
        #outputFile.write("Policy: BayesianBandit" + str(policy.getPriors()) + "\n")

        #policy = RandomPolicy()
        #policyName = "Random"
        #outputFile.write("Policy: Random\n")

        #policy = Contextualclick()
        #policyName = "ContextualClick"
        #outputFile.write("Policy: ContextualClick\n")

        #policy = GMPolicy()
        #policyName = "GMPolicy"
        #outputFile.write("Policy: GM\n")

        #policy = LinUCB()
        #policyName = "LinUCB"
        #outputFile.write("Policy: LinUCB\n")

        #policy = eGreedy(0.1)
        #policyName = "eGreedy" + str(policy.getEpsilon())
        #outputFile.write("Policy: eGreedy" + str(policy.getEpsilon()) + "\n")

        A0 = np.eye(136)
        b0 = np.zeros(136)
        policy = eGreedyContextual(0.1, RidgeRegressor(A0,b0))
        policyName = "eGreedyContextual" + str(policy.getEpsilon())
        outputFile.write("Policy: eGreedyContextual" + str(policy.getEpsilon()) + "\n")

        #policy = eAnnealing()
        #policyName = "eAnnealing"
        #utputFile.write("Policy: eAnnealing\n")

        #policy = Softmax(0.01)
        #policyName = "Softmax" + str(policy.getTemp())
        #outputFile.write("Policy: Softmax" + str(policy.getTemp()) + "\n")

        #policy = UCB1()
        #policyName = "UCB1"
        #outputFile.write("Policy: UCB1\n")

        #policy = EXP3(0.5)
        #policyName = "EXP3" + str(policy.getGamma())
        #outputFile.write("Policy: EXP3" + str(policy.getGamma()) + "\n")

        #policy = Naive3()
        #policyName = "Naive3"
        #outputFile.write("Policy: Naive3\n")

        #policy = Contextualclick()
        #policyName = "ContextualClick"
        #outputFile.write("Policy: Contextual Click\n")

        #policy = LinearBayes()
        #policyName = "LinearBayes"
        #outputFile.write("Policy: Linear Bayes\n")

        #policy = EnsembleRandomModel()
        #policyName = "EnsembleRandom"
        #outputFile.write("Policy: EnsembleRandom\n")

        #policy = EnsembleRandomModelUpdateAll()
        #policyName = "EnsembleRandomUpdateAll"
        #outputFile.write("Policy: EnsembleRandomUpdateAll\n")

        #policy = EnsembleSoftmaxModel(0.1)
        #policyName = "EnsembleSoftmax"
        #outputFile.write("Policy: EnsembleSoftmax\n")

        #policy = EnsembleEAnnealingModel()
        #policyName = "EnsembleEAnnealing"
        #outputFile.write("Policy: EnsembleEAnnealing\n")

        #policy = EnsembleEAnnealingUpdateAllModel()
        #policyName = "EnsembleEAnnealingUpdateAll"
        #outputFile.write("Policy: EnsembleEAnnealingUpdateAll\n")

        #policy = EnsembleTestModel(1.0, 1.0)
        #policyName = "EnsembleTest"
        #outputFile.write("Policy: EnsembleTest\n")

        evalPolicy = MyEvaluationPolicy(sys.stdout, logStep, 0, policyName, inputFileShort, outputFile)

        ## Only use the second one for the EXP3 algorithm
        ev = Evaluator(generator, evalPolicy, policy)
        #ev = EvaluatorEXP3(generator, evalPolicy, policy)


        value = ev.runEvaluation()

        if logStep == 1:
            print("final result => " + str(value))
            print(str(currentTimeMillis() - t) + " ms" + "\n")
            outputFile.write("Total runtime: " + str(currentTimeMillis() - t) + " ms" + "\n")




if __name__ == '__main__':

    app = Main()
    app.main(sys.argv)
