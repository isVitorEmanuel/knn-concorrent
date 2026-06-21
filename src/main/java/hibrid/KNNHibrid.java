package hibrid;

import classes.Neighbor;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.concurrent.*;

/**
 * @class KNNHibrid
 * @brief Implements the K-Nearest Neighbors algorithm using a hybrid producer-consumer architecture.
 * A single virtual thread reads the file and dispatches line batches into a shared queue,
 * while multiple platform threads consume and process those batches concurrently.
 */
public class KNNHibrid {

    private static final int NUM_PLATFORM_THREADS =
            Math.max(2, Runtime.getRuntime().availableProcessors());

    /** Number of lines grouped into a single batch to reduce synchronization overhead. */
    private static final int BATCH_SIZE = 5_000;

    /** Maximum number of batches held in the blocking queue at any time. */
    private static final int QUEUE_CAPACITY = 200;

    /**
     * @record DistanceRecord
     * @brief Holds a neighbor data point and its calculated distance to the target.
     * Implements Comparable to allow sorting and ordering within priority queues.
     */
    private record DistanceRecord(Neighbor neighbor, double distance) implements Comparable<DistanceRecord> {
        @Override
        public int compareTo(DistanceRecord other) {
            return Double.compare(this.distance, other.distance);
        }
    }

    /**
     * @method predictStream
     * @brief Entry point for the hybrid parallel classification.
     * Orchestrates the execution by invoking the hybrid engine with one virtual producer
     * and multiple platform consumers.
     * @param filePath Path to the CSV dataset file.
     * @param target The target instance to be classified.
     * @param k The number of nearest neighbors to consider.
     * @return The predicted class label string.
     */
    public String predictStream(String filePath, Neighbor target, int k) {
        System.out.printf("[Hybrid Batched] 1 Virtual (Producer) | %d Platform (Consumers)%n", NUM_PLATFORM_THREADS);
        return runHybrid(filePath, target, k);
    }

    /**
     * @method runHybrid
     * @brief Manages the full lifecycle of the hybrid producer-consumer pipeline.
     * Initializes the shared batch queue, spawns platform consumer threads and a single
     * virtual producer thread, waits for completion, merges local results, and performs
     * the final majority vote.
     * @param filePath Path to the CSV dataset file.
     * @param target The target instance to be classified.
     * @param k The number of nearest neighbors to consider.
     * @return The final predicted label or "Unknown" in case of failures.
     */
    private String runHybrid(String filePath, Neighbor target, int k) {
        BlockingQueue<List<String>> queue = new ArrayBlockingQueue<>(QUEUE_CAPACITY);

        @SuppressWarnings("unchecked")
        PriorityQueue<DistanceRecord>[] results = new PriorityQueue[NUM_PLATFORM_THREADS];
        Thread[] consumers = new Thread[NUM_PLATFORM_THREADS];

        Thread.Builder platformBuilder = Thread.ofPlatform().name("knn-consumer-", 0);

        for (int i = 0; i < NUM_PLATFORM_THREADS; i++) {
            final int index = i;
            consumers[i] = platformBuilder.unstarted(() -> {
                PriorityQueue<DistanceRecord> localTopK = new PriorityQueue<>(k, Collections.reverseOrder());

                try {
                    while (true) {
                        List<String> batch = queue.take();

                        if (batch.isEmpty()) break;

                        for (String line : batch) {
                            if (line.isBlank()) continue;

                            Neighbor current = parseLineToNeighbor(line);
                            if (current == null) continue;

                            if (current.getValues().size() != target.getValues().size()) continue;

                            double dist = calculateEuclideanDistance(target, current);

                            if (localTopK.size() < k) {
                                localTopK.add(new DistanceRecord(current, dist));
                            } else if (dist < localTopK.peek().distance) {
                                localTopK.poll();
                                localTopK.add(new DistanceRecord(current, dist));
                            }
                        }
                    }
                    results[index] = localTopK;
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            });
            consumers[i].start();
        }

        Thread producer = Thread.ofVirtual().name("knn-producer").start(() -> {
            try (BufferedReader reader = new BufferedReader(
                    new InputStreamReader(new FileInputStream(filePath), StandardCharsets.UTF_8))) {

                String line = reader.readLine();
                if (line == null) return;

                List<String> currentBatch = new ArrayList<>(BATCH_SIZE);

                while ((line = reader.readLine()) != null) {
                    currentBatch.add(line);

                    if (currentBatch.size() == BATCH_SIZE) {
                        queue.put(currentBatch);
                        currentBatch = new ArrayList<>(BATCH_SIZE);
                    }
                }

                if (!currentBatch.isEmpty()) {
                    queue.put(currentBatch);
                }

            } catch (IOException e) {
                System.err.println("Error reading file: " + e.getMessage());
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            } finally {
                for (int i = 0; i < NUM_PLATFORM_THREADS; i++) {
                    try {
                        queue.put(Collections.emptyList());
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                    }
                }
            }
        });

        try {
            producer.join();
            for (Thread t : consumers) t.join();
        } catch (InterruptedException e) {
            System.err.println("Main thread interrupted. Cleaning up...");
            producer.interrupt();
            for (Thread t : consumers) if (t != null) t.interrupt();
            Thread.currentThread().interrupt();
            return "Unknown";
        }

        PriorityQueue<DistanceRecord> globalTopK = new PriorityQueue<>(k, Collections.reverseOrder());
        for (PriorityQueue<DistanceRecord> localTopK : results) {
            if (localTopK == null) continue;
            for (DistanceRecord record : localTopK) {
                if (globalTopK.size() < k) {
                    globalTopK.add(record);
                } else if (record.distance < globalTopK.peek().distance) {
                    globalTopK.poll();
                    globalTopK.add(record);
                }
            }
        }

        if (globalTopK.isEmpty()) {
            System.err.println("Error: no valid neighbors found.");
            return "Unknown";
        }

        return majorityVote(globalTopK);
    }

    /**
     * @method calculateEuclideanDistance
     * @brief Computes the Euclidean distance between two Neighbor multidimensional vectors.
     * Loops through numerical features sequentially to perform geometric distance calculation.
     * @param target The reference entity.
     * @param dataPoint The dataset record entity.
     * @return Geometric Euclidean distance as a double value.
     */
    private double calculateEuclideanDistance(Neighbor target, Neighbor dataPoint) {
        double sum = 0.0;
        ArrayList<Double> tv = target.getValues();
        ArrayList<Double> dv = dataPoint.getValues();
        for (int i = 0; i < tv.size(); i++) {
            double diff = tv.get(i) - dv.get(i);
            sum += diff * diff;
        }
        return Math.sqrt(sum);
    }

    /**
     * @method parseLineToNeighbor
     * @brief Converts a comma-separated text line into a typed Neighbor domain model.
     * Extracts numerical properties from previous columns and matches the final column
     * to the category/classification label string.
     * @param line Raw line string extracted from the text file.
     * @return A validated Neighbor object instance or null if parsing fails.
     */
    private Neighbor parseLineToNeighbor(String line) {
        String[] parts = line.split(",");
        if (parts.length < 2) return null;
        ArrayList<Double> values = new ArrayList<>();
        try {
            for (int i = 0; i < parts.length - 1; i++)
                values.add(Double.parseDouble(parts[i].trim()));
            return new Neighbor(values, parts[parts.length - 1].trim());
        } catch (NumberFormatException e) {
            return null;
        }
    }

    /**
     * @method majorityVote
     * @brief Resolves class labels by frequency count over the consolidated nearest neighbors.
     * Iterates over elements inside the queue, maps occurrence scores, and determines the modes.
     * @param topK Priority queue containing the global nearest dataset entries.
     * @return String holding the winner classification label name.
     */
    private String majorityVote(PriorityQueue<DistanceRecord> topK) {
        Map<String, Integer> freq = new HashMap<>();
        for (DistanceRecord r : topK)
            freq.merge(r.neighbor.getLabel(), 1, Integer::sum);
        return Collections.max(freq.entrySet(), Map.Entry.comparingByValue()).getKey();
    }
}