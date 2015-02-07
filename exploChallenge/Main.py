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
import sys
import time



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

        try:
            # YahooLogLineReader parameters: path, starting index, ending index, number of features
            # If not using a day's data with multiple files, use other version of constructor with only path and num of features
            # Sample line of data
            # 1317513291 id-560620 0 |user 1 9 11 13 23 16 18 17 19 15 43 14 39 30 66 50 27 104 20 |id-552077 |id-555224 |id-555528 |id-559744 |id-559855 |id-560290 |id-560518 |id-560620 |id-563115 |id-563582 |id-563643 |id-563787 |id-563846 |id-563938 |id-564335 |id-564418 |id-564604 |id-565364 |id-565479 |id-565515 |id-565533 |id-565561 |id-565589 |id-565648 |id-565747 |id-565822
            # File names: ydata-fp-td-clicks-v2_0.20111002-02, ydata-fp-td-clicks-v2_0.20111002-03, etc.
            #reader = YahooLogLineReader("/Users/bixlermike/ContextualBandit-master/exploChallenge/ydata-fp-td-clicks-v2_0.201110", 2, 3, 136)

            reader = YahooLogLineReader("/Users/bixlermike/Sites/RecEngine/ContextualBandit-master/exploChallenge/ydata-fp-td-clicks-v2_0.20111002-all", 136)
            #reader = YahooLogLineReader("./fakeFile.txt", 136)
            #reader = YahooLogLineReader("/Users/bixlermike/Sites/RecEngine/ContextualBandit-master/exploChallenge/ydata-fp-td-clicks-v2_0.20111002-", 2, 6, 136)
            logStep = 100000
        except:
            # Don't have a test file yet...create subset of one day's Yahoo data as a tester
            #reader = YahooLogLineReader("/Users/bixlermike/Sites/RecEngine/ContextualBandit-master/exploChallenge/sampleData.txt", 8)
            logStep = 1


        generator = FromFileLogLineGenerator(reader)
        #policy = MyPolicy()
        #policy = RandomPolicy()
        #policy = eGreedy()
        policy = Softmax()
        #policy = UCB1()
        #policy = EXP3()
        #policy = Mostclick()
        #policy = Clickrate()
        #policy = Naive3()
        #policy = Contextualclick()
        #policy = LinearBayes()
        #policy = LinearBayesFtu()

        evalPolicy = MyEvaluationPolicy(sys.stdout, logStep, 0)

        ev = Evaluator(generator, evalPolicy, policy)
        #ev = EvaluatorEXP3(generator, evalPolicy, policy)


        value = ev.runEvaluation()

        if logStep == 1:
            print("final result => " + str(value))
            print(str(currentTimeMillis() - t) + " ms" + "\n")




if __name__ == '__main__':

    app = Main()
    app.main(sys.argv)
