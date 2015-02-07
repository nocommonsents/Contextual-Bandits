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
		final double PROB_OF_CLICK = 0.25;
		final int NUMBER_OF_ARTICLES = 25;
		final int NUMBER_OF_FEATURES = 136;
		final int NUMBER_OF_LOG_LINES = 10000;

		int binaryResult;
		int currentValue;
		int featureNumber = 2; // start at 2 since feature 1 is always present
		int index;
		int timestamp = 100000;
		String allArticlesString = "";
		String featureString = "";

		PrintWriter writer = new PrintWriter("sampleData.txt", "UTF-8");
		Random rnd = new Random();
		ArrayList<Integer> possibleArticles = new ArrayList<Integer>();

		// Generate id numbers for fake articles
		for (int number = 0; number < NUMBER_OF_ARTICLES; number++){
			int n = 100000 + rnd.nextInt(900000);
			// If article id has already been generated, add one more iteration to loop
			if (possibleArticles.contains(n)){
				number--;
				System.out.println("Duplicate article ID generated.  Creating another..." + n);
			}
			else {
				possibleArticles.add(n);
			}
			//System.out.println(n);
		}

		// Generate log lines
		for (int number = 0; number < NUMBER_OF_LOG_LINES; number++){
			// Copy article IDs so that we can make changes to list without permanently altering list of articles
			ArrayList<Integer> copyPossibleArticles = new ArrayList<Integer>();

			copyPossibleArticles.addAll(possibleArticles);
			if (number == 0){
				timestamp = 100000;
			}
			else {
				timestamp += rnd.nextInt(10);
			}
			index = rnd.nextInt(possibleArticles.size());
			currentValue = possibleArticles.get(index);
			if (new Random().nextDouble() <= PROB_OF_CLICK){
				binaryResult = 1;
			}
			else {
				binaryResult = 0;
			}

			while (featureNumber <= NUMBER_OF_FEATURES){
				featureNumber += rnd.nextInt(NUMBER_OF_FEATURES);
				if (featureNumber <= NUMBER_OF_FEATURES){
					featureString += " " + featureNumber;
				}
			}

			for (int value : copyPossibleArticles) {
				allArticlesString += "|id-" + value + " ";
			}

			String line = timestamp + " " + "id-" + currentValue + " " + binaryResult + 
			" " + "|user 1" + featureString + " " + allArticlesString;
			System.out.println(number);
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
		writer.close();

	}

}
