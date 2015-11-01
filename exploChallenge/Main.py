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
#     Michael Bixler - Adapted to add numerous non-contextual and contextual bandit algorithms
#                      as well as ensemble approaches and pre and post-processing analysis/plots
#-------------------------------------------------------------------------------
#package exploChallenge;

#import java.io.FileNotFoundException;
import numpy as np
import os
import sys
import time


from myPolicy.MyPolicy import MyPolicy
from exploChallenge.eval.EvaluationPolicy import EvaluationPolicy
from exploChallenge.eval.Evaluator import Evaluator
from exploChallenge.eval.EvaluatorEXP3 import EvaluatorEXP3
from exploChallenge.eval.MyEvaluationPolicy import MyEvaluationPolicy
from exploChallenge.logs.FromFileLogLineGenerator import FromFileLogLineGenerator
from exploChallenge.logs.yahoo.YahooArticle import YahooArticle
from exploChallenge.logs.yahoo.YahooLogLineReader import YahooLogLineReader
from exploChallenge.logs.yahoo.YahooVisitor import YahooVisitor
from exploChallenge.policies.RidgeRegressor import RidgeRegressor

# Non-contextual bandit policies
from exploChallenge.policies.RandomPolicy import RandomPolicy
from exploChallenge.policies.MostClicked import MostClicked
from exploChallenge.policies.MostRecent import MostRecent
from exploChallenge.policies.MostCTR import MostCTR
from exploChallenge.policies.eGreedy import eGreedy
from exploChallenge.policies.eAnnealing import eAnnealing
from exploChallenge.policies.Softmax import Softmax
from exploChallenge.policies.UCB1 import UCB1
from exploChallenge.policies.EXP3 import EXP3
from exploChallenge.policies.BinomialUCI import BinomialUCI

# Contextual bandit policies
from exploChallenge.policies.eGreedyContextual import eGreedyContextual
from exploChallenge.policies.eAnnealingContextual import eAnnealingContextual
from exploChallenge.policies.LinUCB import LinUCB
from exploChallenge.policies.NaiveBayesContextual import NaiveBayesContextual
from exploChallenge.policies.SoftmaxContextual import SoftmaxContextual

# Ensemble bandit policies
from exploChallenge.policies.EnsembleRandom import EnsembleRandom
from exploChallenge.policies.EnsembleRandomUpdateAll import EnsembleRandomUpdateAll
from exploChallenge.policies.EnsembleEAnnealing import EnsembleEAnnealing
from exploChallenge.policies.EnsembleSoftmax import EnsembleSoftmax
from exploChallenge.policies.EnsembleTS import EnsembleTS
from exploChallenge.policies.EnsembleBinomialUCI import EnsembleBinomialUCI
from exploChallenge.policies.EnsembleBinomialUCIMod1 import EnsembleBinomialUCIMod1
from exploChallenge.policies.EnsembleBinomialUCIMod2 import EnsembleBinomialUCIMod2

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
        #outputFile = open("banditOutputsLinUCBWithTime.txt", "a+")
        outputFile = open("dissertationTesting.txt", "a+")


        try:
            # First file is for testing only
            inputFile = "/Users/bixlermike/Contextual-Bandits/exploChallenge/first_lines.txt"
            #inputFile = "/Users/bixlermike/Contextual-Bandits/exploChallenge/ydata-fp-td-clicks-v2_0.20111002-08.txt"

            # Filtered subset that only contains features with > 10% support
            #inputFile = "/Users/bixlermike/Contextual-Bandits/exploChallenge/first_lines_filtered.txt"
            #inputFile = "/Users/bixlermike/Contextual-Bandits/exploChallenge/ydata-fp-td-clicks-v2_0.20111002-08-filtered10percent.txt"

            inputFileShort = "y"    # Yahoo! data = "y"
            reader = YahooLogLineReader(inputFile, 136)
            logStep = 1
        except:
            print "Problem with input file."
            logStep = 1


        generator = FromFileLogLineGenerator(reader)

        outputFile.write("Simulation started at: " + strftime("%Y-%m-%d %H:%M:%S") + "\n")
        outputFile.write("Input file: " + os.path.basename(inputFile) + "\n")

        ## Pick a single bandit algorithm and corresponding policy name / output file

        #policy = RandomPolicy()
        #policyName = "Random"
        #outputFile.write("Policy: Random\n")

        #policy = MostClicked()
        #policyName = "MostClicked"
        #outputFile.write("Policy: MostClicked\n")

        #policy = MostRecent()
        #policyName = "MostRecent"
        #outputFile.write("Policy: MostRecent\n")

        #policy = MostCTR()
        #policyName = "MostCTR"
        #outputFile.write("Policy: MostCTR\n")

        #policy = eGreedy(0.1)
        #policyName = "eGreedy" + str(policy.getEpsilon())
        #outputFile.write("Policy: eGreedy" + str(policy.getEpsilon()) + "\n")

        policy = eAnnealing()
        policyName = "eAnnealing"
        outputFile.write("Policy: eAnnealing\n")

        #policy = Softmax(0.1)
        #policyName = "Softmax" + str(policy.getTemp())
        #outputFile.write("Policy: Softmax" + str(policy.getTemp()) + "\n")

        #policy = UCB1()
        #policyName = "UCB1"
        #outputFile.write("Policy: UCB1\n")

        #policy = EXP3(0.5)
        #policyName = "EXP3" + str(policy.getGamma())
        #outputFile.write("Policy: EXP3" + str(policy.getGamma()) + "\n")

        #policy = BinomialUCI()
        #policyName = "BinomialUCI"
        #outputFile.write("Policy: BinomialUCI\n")

        #policy = eGreedyContextual(0.1, RidgeRegressor(np.eye(136), np.zeros(136)))
        #policyName = "eGreedyContextual" + str(policy.getEpsilon())
        #outputFile.write("Policy: eGreedyContextual" + str(policy.getEpsilon()) + "\n")

        #policy = eAnnealingContextual(RidgeRegressor(np.eye(136), np.zeros(136)))
        #policyName = "eAnnealingContextual"
        #outputFile.write("Policy: eAnnealingContextual" + "\n")

        #policy = LinUCB(0.1)
        #policyName = "LinUCB" + str(policy.getAlpha())
        #outputFile.write("Policy: LinUCB\n")

        #policy = LinearBayes()
        #policyName = "LinearBayes"
        #outputFile.write("Policy: LinearBayes\n")

        #policy = SoftmaxContextual(0.1, RidgeRegressor(np.eye(136), np.zeros(136)))
        #policyName = "SoftmaxContextual" + str(policy.getTemp())
        #outputFile.write("Policy: SoftmaxContextual" + str(policy.getTemp()) + "\n")

        #policy = NaiveBayesContextual()
        #policyName = "NaiveBayesContextual"
        #outputFile.write("Policy: NaiveBayesContextual\n")

        #policy = EnsembleRandom()
        #policyName = "EnsembleRandom"
        #outputFile.write("Policy: EnsembleRandom\n")

        #policy = EnsembleRandomUpdateAll()
        #policyName = "EnsembleRandomUpdateAll"
        #outputFile.write("Policy: EnsembleRandomUpdateAll\n")

        #policy = EnsembleEAnnealing()
        #policyName = "EnsembleEAnnealing"
        #outputFile.write("Policy: EnsembleEAnnealing\n")

        #policy = EnsembleSoftmax(0.01)
        #policyName = "EnsembleSoftmax" + str(policy.getTemp())
        #outputFile.write("Policy: EnsembleSoftmax" + str(policy.getTemp())

        #policy = EnsembleTS(RidgeRegressor(np.eye(136), np.zeros(136)))
        #policyName = "EnsembleTS"
        #outputFile.write("Policy: EnsembleTS\n")

        #policy = EnsembleBinomialUCI(RidgeRegressor(np.eye(136), np.zeros(136)))
        #policyName = "EnsembleBinomialUCI"
        #outputFile.write("Policy: EnsembleBinomialUCI\n")

        #policy = EnsembleBinomialUCIMod1(RidgeRegressor(np.eye(136), np.zeros(136)))
        #policyName = "EnsembleBinomialUCIMod1"
        #outputFile.write("Policy: EnsembleBinomialUCIMod1\n")

        #policy = EnsembleBinomialUCIMod2(RidgeRegressor(np.eye(136), np.zeros(136)))
        #policyName = "EnsembleBinomialUCIMod2"
        #outputFile.write("Policy: EnsembleBinomialUCIMod2\n")

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

