package executor;

import classes.Neighbor;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.concurrent.StructuredTaskScope;

/**
 * @class KNNStructuredConcurrency
 * @brief Implements the K-Nearest Neighbors algorithm using Structured Concurrency
 * ({@link StructuredTaskScope}) (Version 4). Mirrors the exact same fixed-partition
 * strategy used by {@code KNNPlatform}: the file is split into {@code NUM_THREADS}
 * line-aligned byte chunks via the identical {@code computeChunks}/{@code processChunk}
 * logic, and each chunk becomes one subtask forked into a single {@link StructuredTaskScope}.
 * <p>
 * Where {@code KNNPlatform} manually manages an array of {@link Thread} objects and joins
 * them one by one, and {@code KNNForkJoin} uses {@code RecursiveTask}/{@code invokeAll},
 * this version replaces both with the structured-concurrency idiom: one {@code fork()} call
 * per chunk, a single {@code join()} that waits for every subtask as a unit, and
 * {@code Subtask.get()} to retrieve each local Top-K — all inside one try-with-resources
 * block, so the whole chunk-processing "family" of subtasks has one lifetime, one error
 * path, and cannot outlive the scope. There is still no shared mutable state and no
 * {@code synchronized} block anywhere in the parallel phase: exactly like the other two
 * versions, each subtask returns its own local Top-K {@link PriorityQueue}, merged only
 * after every subtask has completed.
 * <p>
 * Note: {@link StructuredTaskScope} is a preview API (JEP 480, JDK 21+). Compiling and
 * running this class requires {@code --enable-preview} on both {@code javac} and {@code java}.
 */
public class KNNStructuredConcurrency {

    private static final int NUM_THREADS = Math.max(2, Runtime.getRuntime().availableProcessors());

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
     * @record ChunkBounds
     * @brief Represents the byte boundaries (start and end) of a file segment.
     */
    private record ChunkBounds(long startByte, long endByte) {}

    /**
     * @method predictStream
     * @brief Entry point for the stream-based parallel classification.
     * Orchestrates execution by opening a single {@link StructuredTaskScope} sized
     * implicitly to the number of chunks, just like the fixed thread count used by
     * {@code KNNPlatform}.
     * @param filePath Path to the CSV dataset file.
     * @param target The target instance to be classified.
     * @param k The number of nearest neighbors to consider.
     * @return The predicted class label string.
     */
    public String predictStream(String filePath, Neighbor target, int k) {
        System.out.printf("[StructuredConcurrency] Using %d chunks in one StructuredTaskScope%n", NUM_THREADS);
        return runParallel(filePath, target, k, NUM_THREADS);
    }

    /**
     * @method runParallel
     * @brief Manages the execution lifecycle of the structured-concurrency workers.
     * Computes file positions exactly like {@code KNNPlatform}, opens one
     * {@link StructuredTaskScope.ShutdownOnFailure}, forks one subtask per chunk, joins
     * the whole scope as a unit, merges every local Top-K into a single global one, and
     * triggers the final majority vote.
     * @param filePath Path to the CSV dataset file.
     * @param target The target instance to be classified.
     * @param k The number of nearest neighbors to consider.
     * @param numChunks The number of line-aligned chunks to split the file into.
     * @return The final predicted label or "Unknown" in case of failures.
     */
    private String runParallel(String filePath, Neighbor target, int k, int numChunks) {
        List<ChunkBounds> chunks;
        try {
            chunks = computeChunks(filePath, numChunks);
            System.out.printf("[runParallel] %d chunks | target dim=%d%n",
                    chunks.size(), target.getValues().size());
        } catch (IOException e) {
            System.err.println("Error computing file chunks: " + e.getMessage());
            return "Unknown";
        }

        PriorityQueue<DistanceRecord> globalTopK = new PriorityQueue<>(k, Collections.reverseOrder());

        // Structured concurrency: every subtask forked here shares one lifetime with the
        // scope. join() blocks until all of them finish (success or failure); if any
        // fails, throwIfFailed() propagates it and the scope makes sure the siblings are
        // wound down before the try-with-resources block exits.
        try (var scope = new StructuredTaskScope.ShutdownOnFailure()) {

            List<StructuredTaskScope.Subtask<PriorityQueue<DistanceRecord>>> subtasks =
                    new ArrayList<>(chunks.size());

            for (ChunkBounds chunk : chunks) {
                subtasks.add(scope.fork(() -> processChunk(filePath, chunk, target, k)));
            }

            scope.join();
            scope.throwIfFailed(e -> new RuntimeException("Error processing chunk", e));

            for (var subtask : subtasks) {
                PriorityQueue<DistanceRecord> localTopK = subtask.get();
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

        } catch (InterruptedException e) {
            System.err.println("Main thread interrupted while joining the scope.");
            Thread.currentThread().interrupt();
            return "Unknown";
        } catch (RuntimeException e) {
            System.err.println("Error processing chunks: " + e.getMessage());
            return "Unknown";
        }

        if (globalTopK.isEmpty()) {
            System.err.println("Error: no valid neighbors found.");
            return "Unknown";
        }

        return majorityVote(globalTopK);
    }

    /**
     * @method computeChunks
     * @brief Calculates balanced byte chunks across the file, ensuring line alignment.
     * Dynamically seeks line breaks ('\n') to ensure that lines are not chopped mid-text
     * when dividing the file among worker subtasks. Identical to {@code KNNPlatform.computeChunks}.
     * @param filePath Path to the file.
     * @param numChunks Desired number of partitions.
     * @return A list containing the byte boundaries for each chunk.
     * @throws IOException If file access errors occur.
     */
    private List<ChunkBounds> computeChunks(String filePath, int numChunks) throws IOException {
        List<ChunkBounds> chunks = new ArrayList<>();

        try (RandomAccessFile raf = new RandomAccessFile(filePath, "r")) {
            long fileSize = raf.length();
            if (fileSize == 0) throw new IOException("File is empty: " + filePath);

            raf.seek(0);
            int b;
            while ((b = raf.read()) != -1 && b != '\n') {}

            if (raf.getFilePointer() >= fileSize)
                throw new IOException("File contains only the header — no data.");

            long dataStart    = raf.getFilePointer();
            long dataSize     = fileSize - dataStart;
            long rawChunkSize = dataSize / numChunks;
            long chunkStart   = dataStart;

            for (int i = 0; i < numChunks; i++) {
                long chunkEnd;

                if (i == numChunks - 1) {
                    chunkEnd = fileSize;
                } else {
                    long rawEnd = dataStart + (long) (i + 1) * rawChunkSize;
                    raf.seek(rawEnd);
                    while ((b = raf.read()) != -1 && b != '\n') {}
                    chunkEnd = (b == -1) ? fileSize : raf.getFilePointer();
                }

                if (chunkStart < chunkEnd)
                    chunks.add(new ChunkBounds(chunkStart, chunkEnd));

                chunkStart = chunkEnd;
                if (chunkStart >= fileSize) break;
            }
        }

        return chunks;
    }

    /**
     * @method processChunk
     * @brief Processes a specific chunk of the file assigned to a single subtask.
     * Positions a stream at the starting byte, reads limited records up to the ending byte,
     * parses the instances, and tracks local Top-K nearest neighbors. Identical to
     * {@code KNNPlatform.processChunk}.
     * @param filePath Path to the file.
     * @param chunk The pre-calculated boundary limits for this subtask.
     * @param target The target instance being evaluated.
     * @param k The number of closest neighbors to filter.
     * @return A max-heap PriorityQueue containing the closest local records.
     */
    private static PriorityQueue<DistanceRecord> processChunk(String filePath, ChunkBounds chunk, Neighbor target, int k) {
        PriorityQueue<DistanceRecord> localTopK = new PriorityQueue<>(k, Collections.reverseOrder());
        long chunkSize = chunk.endByte() - chunk.startByte();

        try (FileInputStream fis = new FileInputStream(filePath)) {
            fis.getChannel().position(chunk.startByte());

            InputStream boundedIn = new InputStream() {
                long remaining = chunkSize;

                @Override
                public int read() throws IOException {
                    if (remaining <= 0) return -1;
                    int b = fis.read();
                    if (b != -1) remaining--;
                    return b;
                }

                @Override
                public int read(byte[] b, int off, int len) throws IOException {
                    if (remaining <= 0) return -1;
                    int bytesToRead = (int) Math.min(len, remaining);
                    int bytesRead = fis.read(b, off, bytesToRead);
                    if (bytesRead != -1) remaining -= bytesRead;
                    return bytesRead;
                }
            };

            try (BufferedReader reader = new BufferedReader(new InputStreamReader(boundedIn, StandardCharsets.UTF_8))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    if (line.isBlank()) continue;

                    Neighbor current = parseLineToNeighbor(line);
                    if (current == null) continue;

                    if (current.getValues().size() != target.getValues().size()) {
                        continue;
                    }

                    double dist = calculateEuclideanDistance(target, current);

                    if (localTopK.size() < k) {
                        localTopK.add(new DistanceRecord(current, dist));
                    } else if (dist < localTopK.peek().distance) {
                        localTopK.poll();
                        localTopK.add(new DistanceRecord(current, dist));
                    }
                }
            }
        } catch (IOException e) {
            System.err.println("Error processing chunk: " + e.getMessage());
        }

        return localTopK;
    }

    /**
     * @method calculateEuclideanDistance
     * @brief Computes the Euclidean distance between two Neighbor multidimensional vectors.
     * Loops through numerical features sequentially to perform geometric distance calculation.
     * @param target The reference entity.
     * @param dataPoint The dataset record entity.
     * @return Geometric Euclidean distance as a double value.
     */
    private static double calculateEuclideanDistance(Neighbor target, Neighbor dataPoint) {
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
    private static Neighbor parseLineToNeighbor(String line) {
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
