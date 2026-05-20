package lock;

import classes.Neighbor;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.concurrent.locks.ReentrantLock;

/**
 * @class KNNLock
 * @brief Implements the K-Nearest Neighbors algorithm using parallel Virtual Threads
 * and a ReentrantLock to safely coordinate concurrent updates to a shared global Top-K queue.
 */
public class KNNLock {

    private static final int NUM_PLATFORM_THREADS = Math.max(2, Runtime.getRuntime().availableProcessors());

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
     * @brief Entry point for the lock-based parallel classification.
     * @param filePath Path to the CSV dataset file.
     * @param target The target instance to be classified.
     * @param k The number of nearest neighbors to consider.
     * @return The predicted class label string.
     */
    public String predictStream(String filePath, Neighbor target, int k) {
        System.out.printf("[ReentrantLock] Using %d virtual threads with shared global queue synchronization%n", NUM_PLATFORM_THREADS);
        return runParallel(filePath, target, k, NUM_PLATFORM_THREADS);
    }

    /**
     * @method runParallel
     * @brief Manages the execution lifecycle of the virtual worker threads.
     * Splices the source file, creates a single shared priority queue, protects it via ReentrantLock,
     * and joins all virtual tasks before computing the final majority vote.
     */
    private String runParallel(String filePath, Neighbor target, int k, int numThreads) {
        List<ChunkBounds> chunks;
        try {
            chunks = computeChunks(filePath, numThreads);
        } catch (IOException e) {
            System.err.println("Error computing file chunks: " + e.getMessage());
            return "Unknown";
        }

        PriorityQueue<DistanceRecord> globalTopK = new PriorityQueue<>(k, Collections.reverseOrder());

        ReentrantLock lock = new ReentrantLock();

        Thread.Builder builder = Thread.ofPlatform().name("knn-lock-", 0);
        int actualChunks = chunks.size();
        Thread[] threads = new Thread[actualChunks];

        for (int i = 0; i < actualChunks; i++) {
            ChunkBounds chunk = chunks.get(i);
            threads[i] = builder.start(() -> {
                processChunk(filePath, chunk, target, k, globalTopK, lock);
            });
        }

        for (Thread t : threads) {
            try {
                t.join();
            } catch (InterruptedException e) {
                System.err.println("Main thread interrupted while awaiting workers.");
                Thread.currentThread().interrupt();
                return "Unknown";
            }
        }

        if (globalTopK.isEmpty()) {
            return "Unknown";
        }

        return majorityVote(globalTopK);
    }

    /**
     * @method processChunk
     * @brief Processes a specific chunk of the file and safely updates the global queue using the ReentrantLock.
     */
    private void processChunk(String filePath, ChunkBounds chunk, Neighbor target, int k,
                              PriorityQueue<DistanceRecord> globalTopK, ReentrantLock lock) {
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
                    if (current.getValues().size() != target.getValues().size()) continue;

                    double dist = calculateEuclideanDistance(target, current);

                    lock.lock();
                    try {
                        if (globalTopK.size() < k) {
                            globalTopK.add(new DistanceRecord(current, dist));
                        } else if (dist < globalTopK.peek().distance) {
                            globalTopK.poll();
                            globalTopK.add(new DistanceRecord(current, dist));
                        }
                    } finally {
                        lock.unlock();
                    }
                }
            }
        } catch (IOException e) {
            System.err.println("Error processing chunk: " + e.getMessage());
        }
    }

    private List<ChunkBounds> computeChunks(String filePath, int numChunks) throws IOException {
        List<ChunkBounds> chunks = new ArrayList<>();
        try (RandomAccessFile raf = new RandomAccessFile(filePath, "r")) {
            long fileSize = raf.length();
            if (fileSize == 0) throw new IOException("File is empty.");

            raf.seek(0);
            int b;
            while ((b = raf.read()) != -1 && b != '\n') { /* pula header */ }

            long dataStart = raf.getFilePointer();
            long dataSize = fileSize - dataStart;
            long rawChunkSize = dataSize / numChunks;
            long chunkStart = dataStart;

            for (int i = 0; i < numChunks; i++) {
                long chunkEnd;
                if (i == numChunks - 1) {
                    chunkEnd = fileSize;
                } else {
                    long rawEnd = dataStart + (long) (i + 1) * rawChunkSize;
                    raf.seek(rawEnd);
                    while ((b = raf.read()) != -1 && b != '\n') { /* alinha */ }
                    chunkEnd = (b == -1) ? fileSize : raf.getFilePointer();
                }
                if (chunkStart < chunkEnd) {
                    chunks.add(new ChunkBounds(chunkStart, chunkEnd));
                }
                chunkStart = chunkEnd;
                if (chunkStart >= fileSize) break;
            }
        }
        return chunks;
    }

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

    private String majorityVote(PriorityQueue<DistanceRecord> topK) {
        Map<String, Integer> freq = new HashMap<>();
        for (DistanceRecord r : topK)
            freq.merge(r.neighbor.getLabel(), 1, Integer::sum);
        return Collections.max(freq.entrySet(), Map.Entry.comparingByValue()).getKey();
    }
}