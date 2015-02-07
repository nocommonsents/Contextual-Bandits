Run project with ./go.bat command (must be in exploChallenge folder)

3.5 GB Yahoo data set is on Google Drive.  Had to request access to file, so no use outside of academic purposes is allowed.

Goal is to create different simulated input files to compare performance of contextual bandit algorithms

Simulated input files will vary the number of users, number of features, probability of click, etc.

We will then test the same contextual bandit algorithms on the Yahoo data set to compare performance on real-world data

Contextual bandit policy is set in exploChallenge > Main.py in main method

To-do:
- (Done) Save results to file, including the contextual bandit used and the input parameters used (number of users, number of features, etc.)
(Not done) Could this be improved by automatically writing the different policies to different files?  For example, UCB1 might go to “UCB1_output.txt” while EXP3 might go to “EXP3_output.txt”

- When printing out the contextual bandit used to the output file, also print out the parameters (epsilon, alpha, etc.)

- In exploChallenge > eval > MyEvaluationPolicy.py, have log method print out every x evaluations, not x lines (creates duplicates when doing this…either fix this code or can create duplicate filter when plotting data)

- Think about how to create simulated data that algorithms can learn from.  I don’t think it’s accurate to set a single click-through rate for each simulation.  This is done right now in the DataGenerator.java file.
