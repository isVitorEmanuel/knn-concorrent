package hibrid;

import classes.Neighbor;
import generator.DataSetGenerator;

import java.util.ArrayList;

public class RunHibrid {

    public static void main(String[] args) {
        String path = "dataset_high_dim.csv";

        int numFeatures = DataSetGenerator.NUM_FEATURES;
        ArrayList<Double> targetValues = new ArrayList<>();
        for (int i = 0; i < numFeatures; i++) {
            targetValues.add(500.0);
        }
        Neighbor target = new Neighbor(targetValues, "Unknown");

        KNNHibrid knn = new KNNHibrid();
        int k = 21;

        System.out.println(">>> starting prediction...");
        long startTime = System.currentTimeMillis();

        String result = knn.predictStream(path, target, k);

        long elapsed = System.currentTimeMillis() - startTime;
        System.out.println(">>> predicted class: " + result);
        System.out.printf(">>> time elapsed: %.2f seconds%n", elapsed / 1000.0);
    }
}