import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.Random;


public class DataGenerator {

	/**
	 * @param args
	 * @throws UnsupportedEncodingException 
	 * @throws FileNotFoundException 
	 */
	public static void main(String[] args) throws FileNotFoundException, UnsupportedEncodingException {
		
		/* Probability that a user clicks on a particular article
		 * Should this be dependent on the feature vector or a constant for a given simulation
		*/
		
		final double PROB_OF_CLICK = 0.1;	
		final int NUMBER_OF_ARTICLES = 10000;
		final int NUMBER_OF_CANDIDATE_ARTICLES = 50;
		final int NUMBER_OF_FEATURES = 136;
		final int NUMBER_OF_LOG_LINES = 5000000;
		final int NUMBER_OF_USERS = 500;


		int currentArticleId;
		int currentValue;
		int featureNumber = 2; // start at 2 since feature 1 is always present
		int index;
		int reward;
		int timestamp = 100000;
		String allArticlesString = "";
		String featureString = "";

		PrintWriter writer = new PrintWriter("sampleData.txt", "UTF-8");
		Random rnd = new Random();
		ArrayList<Integer> possibleArticles = new ArrayList<Integer>();
		int[] userList = new int[NUMBER_OF_USERS-1];

		// Generate id numbers for fake articles
		for (int number = 0; number < NUMBER_OF_ARTICLES; number++){
			int n = 100000 + rnd.nextInt(900000);
			// If article id has already been generated, don't count this iteration of the loop
			if (possibleArticles.contains(n)){
				number--;
				//System.out.println("Duplicate article ID generated.  Creating another..." + n);
			}
			else {
				possibleArticles.add(n);
			}
			//System.out.println(n);
		}
		
		// Generate fake users
		
		/* Create a string of features that describe each user-article interaction
		 * Example:  1 4 24 67 132 for 136 possible feature vector values
		 */
		for (int number = 0; number < NUMBER_OF_USERS; number++) {
			featureString = "1";
			featureNumber = 2;
			while (featureNumber <= NUMBER_OF_FEATURES){
				// Generate random features from a list of NUMBER_OF_FEATURES potentials
				featureNumber += rnd.nextInt(NUMBER_OF_FEATURES) + 1;
				if (featureNumber <= NUMBER_OF_FEATURES){
					featureString += " " + featureNumber;
				}
			}
		}

		// Generate log lines
		for (int number = 0; number < NUMBER_OF_LOG_LINES; number++){
			// Generate a list of NUMBER_CANDIDATE_ARTICLES candidate articles for the recommendation
			ArrayList<Integer> copyPossibleArticles = new ArrayList<Integer>();
			for (int articlesSoFar = 0; articlesSoFar < NUMBER_OF_CANDIDATE_ARTICLES; articlesSoFar++){
				currentArticleId = rnd.nextInt(possibleArticles.size());
				if (copyPossibleArticles.contains(currentArticleId)){
					articlesSoFar--;
					//System.out.println("Duplicate candidate article generated.  Creating another..." + currentArticleId);
				}
				else {
					copyPossibleArticles.add(currentArticleId);
				}
			}
			if (number == 0){
				timestamp = 100000;
			}
			else {
				timestamp += rnd.nextInt(10);
			}
			index = rnd.nextInt(copyPossibleArticles.size());
			currentValue = copyPossibleArticles.get(index);
			if (new Random().nextDouble() <= PROB_OF_CLICK){
				reward = 1;
			}
			else {
				reward = 0;
			}

			while (featureNumber <= NUMBER_OF_FEATURES){
				// Generate random features from a list of NUMBER_OF_FEATURES potentials
				featureNumber += rnd.nextInt(NUMBER_OF_FEATURES) + 1;
				if (featureNumber <= NUMBER_OF_FEATURES){
					featureString += " " + featureNumber;
				}
			}

			for (int value : copyPossibleArticles) {
				allArticlesString += "|id-" + value + " ";
			}

			String line = timestamp + " " + "id-" + currentValue + " " + reward + 
			" " + "|user 1" + featureString + " " + allArticlesString;
			if (number % 10000 == 0){
				System.out.println("Done with log line number " + number + ".");
			}
			writer.println(line);

			featureNumber = 2;
			featureString = "";
			allArticlesString = "";
			//			 1317513291 id-560620 0 |user 1 9 11 13 23 16 18 17 19 15 43 14 39 30 66 50 27
			//			104 20 |id-552077 |id-555224 |id-555528 |id-559744 |id-559855 |id-560290
			//			|id-560518 |id-560620 |id-563115 |id-563582 |id-563643 |id-563787 |id-563846
			//			|id-563938 |id-564335 |id-564418 |id-564604 |id-565364 |id-565479 |id-565515
			//			|id-565533 |id-565561 |id-565589 |id-565648 |id-565747 |id-565822
		}
		System.out.println("Data generator complete.");
		writer.close();

	}

}
