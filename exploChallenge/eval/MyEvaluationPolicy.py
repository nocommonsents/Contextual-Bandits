#-------------------------------------------------------------------------------
# Copyright (c) 2012 Jose Antonio Martin H..
# All rights reserved. This program and the accompanying materials
# are made available under the terms of the GNU Public License v3.0
# which accompanies this distribution, and is available at
# http://www.gnu.org/licenses/gpl.html
# 
# Contributors:
#     Jose Antonio Martin H. - Translation to Python from Java
#-------------------------------------------------------------------------------
#package exploChallenge.eval;

import io
import time
from exploChallenge.eval.EvaluationPolicy import EvaluationPolicy

class MyEvaluationPolicy(EvaluationPolicy):

    clicks = 0;
    evaluations = 0;
    lines = 0;
    logger = io.StringIO();
    logFrequency = 0;
    linesToSkip = 0;
    lastEvaluation = 0;

    def __init__(self, a, b = None, c = None, d = None, e = None, f = None):
        if type(a) == type(1):
            self.__init1__(a)
        else:
            self.__init2__(a, b, c, d, e, f)


    def __init1__(self, linesToSkip):
        self.linesToSkip = linesToSkip
        self.clicks = 0
        self.evaluations = 0
        self.lines = 0
        self.logFrequency = -1
        self.lastEvaluationNumber = 0
        currentTimeMillis = lambda:  int(round(time.time() * 1000))
        self.startTime = currentTimeMillis()


    def __init2__(self, outputStream, logFrequency, linesToSkip, policy, inputFileShortened, outputFile):
        self.clicks = 0;
        self.evaluations = 0;
        self.linesToSkip = linesToSkip;
        self.lines = 0;
        self.logFrequency = logFrequency;
        self.logger = outputStream
        self.lastEvaluationNumber = 0
        self.policyName = policy
        self.inputFileShort = inputFileShortened
        self.outputFile = outputFile
        currentTimeMillis = lambda:  int(round(time.time() * 1000))
        self.startTime = currentTimeMillis()
        self.outputFile.write("Policy,Input File,Evaluations,CTR,Cumulative Runtime (ms)\n")
        self.logger.write("Policy,Input File,Evaluations,CTR,Cumulative Runtime (ms)\n")


    #@Override
    def log(self):
        if (self.evaluations % 100 == 0 and self.evaluations != self.lastEvaluationNumber):
            currentTimeMillis = lambda:  int(round(time.time() * 1000))
            self.lastEvaluationNumber = self.evaluations
            self.logger.write(str(self.policyName) + "," + str(self.inputFileShort) + "," + str(self.evaluations) + "," + str(self.getResult()) + "," + str(currentTimeMillis() - self.startTime) + "\n")
            self.outputFile.write(str(self.policyName) + "," + str(self.inputFileShort) + "," + str(self.evaluations) + "," + str(self.getResult()) + "," + str(currentTimeMillis() - self.startTime) +"\n")
        self.logger.flush()

    #@Override
    def getResult(self):
        try:
            return float(self.clicks) / float(self.evaluations)
        except:
            return 0.0


    #@Override
    def evaluate(self, logLine, chosenAction):
        if self.linesToSkip > 0:
            self.linesToSkip -= 1
            return

        if logLine.getAction() == chosenAction:
            self.evaluations += 1
            if logLine.getReward():
                self.clicks += 1

        self.lines += 1
        if self.logFrequency != -1 and self.lines % self.logFrequency == 0:
            self.log()

