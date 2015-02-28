Run project with ./go.bat command (must be in exploChallenge folder)

3.5 GB Yahoo data set is on Google Drive.  Had to request access to file, so no use outside of academic purposes is allowed.

Goal is to create different simulated input files to compare performance of contextual bandit algorithms

Simulated input files will vary the number of users, number of features, probability of click, etc.

We will then test the same contextual bandit algorithms on the Yahoo data set to compare performance on real-world data

Contextual bandit policy is set in exploChallenge > Main.py in main method

To-do:

- Think about how to create simulated data that algorithms can learn from.  I don’t think it’s accurate to explicitly set a single click-through rate for each simulation, though our bandit algorithms should be able to learn from that as well.  This is done right now in the DataGenerator.java file.  So far, the contextual bandits have not done well at all on simulated data (hardly better than random guessing).

- (Highest priority - have version working that picks random policy at each time step) Need to fix EnsemblePolicy class to allow for a combination of the other policies to be used.  For example, a simple version of this might randomly pick one of the other contextual policies for some number of steps, and after that pick the policy with the highest average reward.  Or we might have each contextual policy vote on which article to pick, and we weight their picks by their average rewards to that point to choose one article.