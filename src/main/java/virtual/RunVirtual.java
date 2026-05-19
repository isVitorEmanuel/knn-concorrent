package virtual;

import classes.Neighbor;
import generator.DataSetGenerator;

import java.util.ArrayList;

public class RunVirtual {

    public static void main(String[] args) {
        String path = "dataset_high_dim.csv";
        int k = 3;

        int numFeatures = DataSetGenerator.NUM_FEATURES;
        ArrayList<Double> targetValues = new ArrayList<>();
        for (int i = 0; i < numFeatures; i++) targetValues.add(500.0);
        Neighbor target = new Neighbor(targetValues, "Unknown");

        KNNVirtual knn = new KNNVirtual();

        System.out.println("> starting prediction with virtual threads");

        String resultVirtual = knn.predictStream(path, target, k);

        System.out.println("> predicted class : " + resultVirtual);
    }
}