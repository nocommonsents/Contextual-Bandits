Run project with ./go.bat command (must be in exploChallenge folder)

3.5 GB Yahoo data set is on Google Drive.  Had to request access to file, so no use outside of academic purposes is allowed.

Goal is to create different simulated input files to compare performance of contextual bandit algorithms

Simulated input files will vary the number of users, number of features, probability of click, etc.

We will then test the same contextual bandit algorithms on the Yahoo data set to compare performance on real-world data

Contextual bandit policy is set in exploChallenge > Main.py in main method

To-do:

- Think about how to create simulated data that algorithms can learn from.  I don’t think it’s accurate to explicitly set a single click-through rate for each simulation, though our bandit algorithms should be able to learn from that as well.  This is done right now in the DataGenerator.java file.  So far, the contextual bandits have not done well at all on simulated data (hardly better than random guessing).

- Create different version of ensemble model.  Right now have:
    - Version that picks one of four policies completely at random and picks the arm chosen by the policy
    - Version that uses version of Softmax algorithm to pick an arm (Need to add exponential and temperature components)