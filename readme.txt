Run project with ./go.bat command (must be in exploChallenge folder)

3.5 GB Yahoo data set is on Google Drive.  Had to request access to file, so no use outside of academic purposes is allowed.

Goal is to create different simulated input files to compare performance of contextual bandit algorithms

Simulated input files will vary the number of users, number of features, probability of click, etc.

We will then test the same contextual bandit algorithms on the Yahoo data set to compare performance on real-world data

Contextual bandit policy is set in exploChallenge > Main.py in main method

To-do:
- (Done) Save results to file, including the contextual bandit used and the input parameters used (number of users, number of features, etc.)

- (Done) In exploChallenge > eval > MyEvaluationPolicy.py, have log method print out every x evaluations, not x lines without creating duplicate lines (prints duplicates until next evaluation using that particular contextual bandit)

- When printing out the contextual bandit used to the output file, also print out the parameters (epsilon, alpha, etc.)

- Think about how to create simulated data that algorithms can learn from.  I don’t think it’s accurate to explicitly set a single click-through rate for each simulation, though it could likely learn from that as well.  This is done right now in the DataGenerator.java file.
