package executor;

import classes.Neighbor;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;

/**
 * @class KNNParallelStream
 * @brief Implements the K-Nearest Neighbors algorithm using Java 8 Parallel Streams (Version 4).
 * Mirrors the exact same fixed-partition strategy used by {@code KNNPlatform}: the file is
 * split into {@code NUM_THREADS} line-aligned byte chunks. Instead of spawning platform
 * {@link Thread}s manually and joining them, each chunk is processed through a single
 * {@code parallelStream().map(...).reduce(...)} pipeline. The JDK's common {@code ForkJoinPool}
 * (the same engine behind {@code KNNForkJoin}) transparently distributes the chunks across
 * worker threads; no {@code Thread}, {@code ExecutorService}, or {@code ForkJoinTask} is ever
 * referenced explicitly, and no {@code synchronized} block is needed because every chunk
 * produces its own local Top-K {@link PriorityQueue}, combined only through the associative
 * {@code reduce()} merge step.
 */
public class KNNParallelStream {

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
     * @brief Entry point for the Parallel Stream-based classification.
     * Orchestrates execution over the same number of chunks {@code KNNPlatform} would use,
     * letting the JDK's common ForkJoinPool decide how many worker threads actually run them.
     * @param filePath Path to the CSV dataset file.
     * @param target The target instance to be classified.
     * @param k The number of nearest neighbors to consider.
     * @return The predicted class label string.
     */
    public String predictStream(String filePath, Neighbor target, int k) {
        System.out.printf("[ParallelStream] Using parallelStream() over %d chunks (common pool parallelism=%d)%n",
                NUM_THREADS, ForkJoinPoolParallelismProbe.parallelism());
        return runParallel(filePath, target, k, NUM_THREADS);
    }

    /**
     * @method runParallel
     * @brief Manages the execution lifecycle of the Parallel Stream-based computation.
     * Computes file positions exactly like {@code KNNPlatform}, then replaces the manual
     * {@code Thread[]} + {@code join()} loop with a single {@code parallelStream()} pipeline:
     * {@code map()} turns every chunk into its own local Top-K, and {@code reduce()} merges
     * all local Top-Ks into one global Top-K using an associative combiner. Finally triggers
     * the majority vote.
     * @param filePath Path to the CSV dataset file.
     * @param target The target instance to be classified.
     * @param k The number of nearest neighbors to consider.
     * @param numThreads The number of chunks to partition the file into.
     * @return The final predicted label or "Unknown" in case of failures.
     */
    private String runParallel(String filePath, Neighbor target, int k, int numThreads) {
        List<ChunkBounds> chunks;
        try {
            chunks = computeChunks(filePath, numThreads);
            System.out.printf("[runParallel] %d chunks | target dim=%d%n",
                    chunks.size(), target.getValues().size());
        } catch (IOException e) {
            System.err.println("Error computing file chunks: " + e.getMessage());
            return "Unknown";
        }

        PriorityQueue<DistanceRecord> emptyTopK = new PriorityQueue<>(k, Collections.reverseOrder());

        PriorityQueue<DistanceRecord> globalTopK = chunks.parallelStream()
                .map(chunk -> processChunk(filePath, chunk, target, k))
                .reduce(emptyTopK, (a, b) -> mergeTopK(a, b, k));

        if (globalTopK.isEmpty()) {
            System.err.println("Error: no valid neighbors found.");
            return "Unknown";
        }

        return majorityVote(globalTopK);
    }

    /**
     * @method mergeTopK
     * @brief Combines two local Top-K priority queues into a single Top-K queue of size k.
     * Used as the associative combiner passed to {@code reduce()}: every pairwise merge is
     * independent of merge order, which is exactly what makes it safe for the parallel
     * stream's divide-and-combine execution.
     * @param a First local Top-K queue.
     * @param b Second local Top-K queue.
     * @param k The number of nearest neighbors to keep.
     * @return A single merged max-heap PriorityQueue containing the closest k records.
     */
    private static PriorityQueue<DistanceRecord> mergeTopK(PriorityQueue<DistanceRecord> a,
                                                            PriorityQueue<DistanceRecord> b,
                                                            int k) {
        PriorityQueue<DistanceRecord> merged = new PriorityQueue<>(k, Collections.reverseOrder());

        for (PriorityQueue<DistanceRecord> source : List.of(a, b)) {
            if (source == null) continue;
            for (DistanceRecord record : source) {
                if (merged.size() < k) {
                    merged.add(record);
                } else if (record.distance < merged.peek().distance) {
                    merged.poll();
                    merged.add(record);
                }
            }
        }

        return merged;
    }

    /**
     * @method computeChunks
     * @brief Calculates balanced byte chunks across the file, ensuring line alignment.
     * Dynamically seeks line breaks ('\n') to ensure that lines are not chopped mid-text
     * when dividing the file among the chunks the parallel stream will later process.
     * Identical to {@code KNNPlatform.computeChunks}.
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
     * @brief Processes a specific chunk of the file, invoked once per chunk from within
     * the parallel stream's {@code map()} stage.
     * Positions a stream at the starting byte, reads limited records up to the ending byte,
     * parses the instances, and tracks local Top-K nearest neighbors. Identical to
     * {@code KNNPlatform.processChunk}.
     * @param filePath Path to the file.
     * @param chunk The pre-calculated boundary limits for this chunk.
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

    /**
     * @class ForkJoinPoolParallelismProbe
     * @brief Tiny helper used only for the log line, so it's obvious at runtime how many
     * worker threads the common {@code ForkJoinPool} (the engine behind every parallel
     * stream) actually has available — this is controlled by the JVM/system property
     * mentioned in the slides ({@code java.util.concurrent.ForkJoinPool.common.parallelism}),
     * not by {@code NUM_THREADS}.
     */
    private static final class ForkJoinPoolParallelismProbe {
        static int parallelism() {
            return java.util.concurrent.ForkJoinPool.getCommonPoolParallelism();
        }
    }
}
