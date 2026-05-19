package virtual;

import classes.Neighbor;

import java.util.ArrayList;

public class RunParallel {

    public static void main(String[] args) {
        String path = "dataset_high_dim.csv";
        int k = 3;

        // ── Diagnóstico de ambiente ───────────────────────────────────────
        int cores = Runtime.getRuntime().availableProcessors();
        System.out.println("=".repeat(50));
        System.out.println("[DIAGNOSTIC] Logical CPU cores visible to JVM: " + cores);
        if (cores == 1) {
            System.out.println("[DIAGNOSTIC] WARNING: Only 1 core detected.");
            System.out.println("[DIAGNOSTIC] This may indicate a Docker container or IDE CPU restriction.");
            System.out.println("[DIAGNOSTIC] Parallel version will behave similarly to serial.");
            System.out.println("[DIAGNOSTIC] Expected time for 1GB with 100 features: ~5-10 minutes.");
        } else {
            System.out.println("[DIAGNOSTIC] Expected speedup vs serial: ~" + cores + "x");
            System.out.println("[DIAGNOSTIC] Expected time for 1GB: ~" + (20 / cores) + "-" + (30 / cores) + "s");
        }
        System.out.println("=".repeat(50));

        int numFeatures = DataSetGenerator.NUM_FEATURES;
        ArrayList<Double> targetValues = new ArrayList<>();
        for (int i = 0; i < numFeatures; i++) targetValues.add(500.0);
        Neighbor target = new Neighbor(targetValues, "Unknown");

        KNNParallel knn = new KNNParallel();

        // ── Platform Threads ──────────────────────────────────────────────
        System.out.println(">>> Starting prediction with Platform Threads...");
        long t0 = System.currentTimeMillis();

        String resultPlatform = knn.predictStream(path, target, k);

        long t1 = System.currentTimeMillis();
        System.out.println(">>> Predicted class : " + resultPlatform);
        System.out.printf (">>> Time elapsed    : %.2f seconds%n", (t1 - t0) / 1000.0);
    }
}