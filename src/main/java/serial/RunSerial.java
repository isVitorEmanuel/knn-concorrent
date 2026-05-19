package serial;

import classes.Neighbor;
import generator.DataSetGenerator;

import java.util.ArrayList;

/**
 * @class RunSerial
 * @brief Entry point for the serial execution of the KNN algorithm.
 * This class orchestrates the process of creating a target point,
 * initializing the KNN engine, and triggering the stream-based prediction
 * to handle large datasets (e.g., 1GB) without exceeding memory limits.
 */
public class RunSerial {

    /**
     * @brief Main execution method.
     * This method defines the dataset path, creates the target point to be classified,
     * and measures the flow of the serial KNN prediction.
     * @param args Command line arguments (not used).
     */
    public static void main(String[] args) {
        String path = "dataset_high_dim.csv";

        int numFeatures = DataSetGenerator.NUM_FEATURES;
        ArrayList<Double> targetValues = new ArrayList<>();
        for (int i = 0; i < numFeatures; i++) {
            targetValues.add(500.0);
        }
        Neighbor target = new Neighbor(targetValues, "Unknown");

        KNNSerial knn = new KNNSerial();
        int k = 21;

        System.out.println(">>> starting prediction...");
        long startTime = System.currentTimeMillis();

        String result = knn.predictStream(path, target, k);

        long elapsed = System.currentTimeMillis() - startTime;
        System.out.println(">>> predicted class: " + result);
        System.out.printf(">>> time elapsed: %.2f seconds%n", elapsed / 1000.0);
    }
}
