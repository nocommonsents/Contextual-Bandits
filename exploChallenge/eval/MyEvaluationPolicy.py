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
from exploChallenge.eval.EvaluationPolicy import EvaluationPolicy

class MyEvaluationPolicy(EvaluationPolicy):

    clicks = 0;
    evaluations = 0;
    lines = 0;
    logger = io.StringIO();
    logFrequency = 0;
    linesToSkip = 0;
    lastEvaluation = 0;

    def __init__(self, a, b = None, c = None, d = None):
        if type(a) == type(1):
            self.__init1__(a)
        else:
            self.__init2__(a, b, c, d)


    def __init1__(self, linesToSkip):
        self.linesToSkip = linesToSkip
        self.clicks = 0
        self.evaluations = 0
        self.lines = 0
        self.logFrequency = -1
        self.lastEvaluationNumber = 0


    def __init2__(self, outputStream, logFrequency, linesToSkip, outputFile):
        self.clicks = 0;
        self.evaluations = 0;
        self.linesToSkip = linesToSkip;
        self.lines = 0;
        self.logFrequency = logFrequency;
        self.logger = outputStream
        self.lastEvaluationNumber = 0
        self.outputFile = outputFile
        self.outputFile.write("Lines Evaluations Clicks CTR \n")
        self.logger.write("Lines Evaluations Clicks CTR \n")



    #@Override
    def log(self):
        if (self.evaluations % 10 == 0 and self.evaluations != self.lastEvaluationNumber):
            self.lastEvaluationNumber = self.evaluations
            self.logger.write(str(self.lines) + " " + str(self.evaluations) + " " + str(self.clicks) + " " + str(self.getResult()) + "\n")
            self.outputFile.write(str(self.lines) + " " + str(self.evaluations) + " " + str(self.clicks) + " " + str(self.getResult()) + "\n")
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

