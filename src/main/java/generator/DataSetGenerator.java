package generator;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;
import java.util.Locale;

/**
 * @class generator.DataSetGenerator
 * @brief Scalable utility class to generate high-dimensional datasets for KNN.
 * Generates a CSV file with double-valued features and a label column.
 * The number of features here must match NUM_FEATURES in RunSerial.java.
 */
public class DataSetGenerator {

    /** @brief Number of feature columns per data point. Must match RunSerial.NUM_FEATURES. */
    public static final int NUM_FEATURES = 5;

    public static void main(String[] args) {
        Locale.setDefault(Locale.US);

        String filename = "dataset_high_dim.csv";
        long targetSizeBytes = 1024L * 1024L * 1024L;
        String[] labels = {"Class_A", "Class_B", "Class_C", "Class_D", "Class_E"};
        Random random = new Random();

        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filename))) {
            File file = new File(filename);
            long currentSize = 0;
            long rowCount = 0;

            System.out.println("> starting dataset generation with " + NUM_FEATURES + " features...");

            StringBuilder header = new StringBuilder();
            for (int i = 1; i <= NUM_FEATURES; i++) {
                header.append("feature_").append(i).append(",");
            }
            header.append("label\n");
            writer.write(header.toString());

            while (currentSize < targetSizeBytes) {
                StringBuilder line = new StringBuilder();
                for (int i = 0; i < NUM_FEATURES; i++) {
                    double featureValue = random.nextDouble() * 1000;
                    line.append(String.format("%.2f", featureValue)).append(",");
                }
                line.append(labels[random.nextInt(labels.length)]).append("\n");
                writer.write(line.toString());
                rowCount++;

                if (rowCount % 100000 == 0) {
                    writer.flush();
                    currentSize = file.length();
                }
            }

            System.out.println("\n\n> dataset generated successfully.");
            System.out.printf("> final size: %.2f mb%n", file.length() / (1024.0 * 1024.0));
            System.out.println("> total rows: " + rowCount);
        } catch (IOException e) {
            System.err.println("error generating file: " + e.getMessage());
        }
    }
}