package atomic;

import classes.Neighbor;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.concurrent.atomic.AtomicReference;

/**
 * @class KNNAtomic
 * @brief Implements the K-Nearest Neighbors algorithm using parallel platform threads
 * and a lock-free shared global state managed by an AtomicReference.
 */
public class KNNAtomic {

    private static final int NUM_PLATFORM_THREADS = Math.max(2, Runtime.getRuntime().availableProcessors());

    /**
     * @record DistanceRecord
     * @brief Holds a neighbor data point and its calculated distance to the target.
     * Implements Comparable to allow natural ascending sorting by distance.
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
     * @class ImmutableTopK
     * @brief An immutable structure that holds a sorted list of the current best neighbors.
     * Facilitates safe lock-free state transitions inside atomic references.
     */
    private static class ImmutableTopK {
        final List<DistanceRecord> list;
        final int k;

        /**
         * @method ImmutableTopK
         * @brief Constructor for the initial empty state.
         * @param k The maximum allowed capacity of neighbors.
         */
        ImmutableTopK(int k) {
            this.k = k;
            this.list = Collections.emptyList();
        }

        /**
         * @method ImmutableTopK
         * @brief Internal constructor to map a new immutable list snapshot.
         * @param list The sorted list of records.
         * @param k The maximum capacity.
         */
        ImmutableTopK(List<DistanceRecord> list, int k) {
            this.k = k;
            this.list = list;
        }

        /**
         * @method tryUpdate
         * @brief Evaluates a new record and returns a new immutable snapshot if it qualifies.
         * Since the list is sorted in ascending order, the last element (size - 1) is always the worst.
         * @param record The new distance record to evaluate.
         * @return A new ImmutableTopK instance if updated, or null if the record does not qualify.
         */
        ImmutableTopK tryUpdate(DistanceRecord record) {
            if (list.size() < k) {
                List<DistanceRecord> newList = new ArrayList<>(list);
                newList.add(record);
                Collections.sort(newList);
                return new ImmutableTopK(Collections.unmodifiableList(newList), k);
            }

            DistanceRecord worst = list.get(list.size() - 1);
            if (record.distance < worst.distance) {
                List<DistanceRecord> newList = new ArrayList<>(list);
                newList.remove(list.size() - 1); // Remove o pior elemento (maior distância)
                newList.add(record);
                Collections.sort(newList); // Reordena
                return new ImmutableTopK(Collections.unmodifiableList(newList), k);
            }
            return null;
        }
    }

    /**
     * @method predictStream
     * @brief Entry point for the stream-based parallel classification.
     * Orchestrates the execution by invoking the parallel engine with the machine's allocated platform threads.
     * @param filePath Path to the CSV dataset file.
     * @param target The target instance to be classified.
     * @param k The number of nearest neighbors to consider.
     * @return The predicted class label string.
     */
    public String predictStream(String filePath, Neighbor target, int k) {
        System.out.printf("[Atomic] Using %d platform threads with lock-free AtomicReference%n", NUM_PLATFORM_THREADS);
        String predict = runParallel(filePath, target, k, NUM_PLATFORM_THREADS);
        System.out.println(predict);
        return predict;
    }

    /**
     * @method runParallel
     * @brief Manages the execution lifecycle of the parallel platform workers.
     * Computes file positions, initializes the AtomicReference container, spawns threads,
     * and triggers the final majority vote once all threads finish.
     * @param filePath Path to the CSV dataset file.
     * @param target The target instance to be classified.
     * @param k The number of nearest neighbors to consider.
     * @param numThreads The total number of platform threads to deploy.
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

        Thread.Builder builder = Thread.ofPlatform().name("knn-atomic-", 0);

        int numChunks = chunks.size();
        Thread[] threads = new Thread[numChunks];

        AtomicReference<ImmutableTopK> globalTopK = new AtomicReference<>(new ImmutableTopK(k));

        for (int i = 0; i < numChunks; i++) {
            ChunkBounds chunk = chunks.get(i);

            threads[i] = builder.unstarted(() -> {
                processChunk(filePath, chunk, target, k, globalTopK);
            });
            threads[i].start();
        }

        for (Thread t : threads) {
            try {
                t.join();
            } catch (InterruptedException e) {
                System.err.println("Main thread interrupted. Stopping workers...");
                for (Thread worker : threads) if (worker != null) worker.interrupt();
                Thread.currentThread().interrupt();
                return "Unknown";
            }
        }

        ImmutableTopK finalResult = globalTopK.get();
        if (finalResult.list.isEmpty()) {
            System.err.println("Error: no valid neighbors found.");
            return "Unknown";
        }

        return majorityVote(finalResult.list);
    }

    /**
     * @method computeChunks
     * @brief Calculates balanced byte chunks across the file, ensuring line alignment.
     * Dynamically seeks line breaks ('\n') to ensure that lines are not chopped mid-text
     * when dividing the file among worker threads.
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
            while ((b = raf.read()) != -1 && b != '\n') { /* pula header */ }

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
                    while ((b = raf.read()) != -1 && b != '\n') { /* alinha */ }
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
     * @brief Processes a specific chunk of the file assigned to a single thread.
     * Evaluates metrics and updates the shared atomic container via lock-free CAS loops,
     * applying a fast-path read bypass to minimize contention.
     * @param filePath Path to the file.
     * @param chunk The pre-calculated boundary limits for this thread.
     * @param target The target instance being evaluated.
     * @param k The number of closest neighbors to filter.
     * @param globalTopK The shared AtomicReference tracking the closest entries globally.
     */
    private void processChunk(String filePath, ChunkBounds chunk, Neighbor target, int k,
                              AtomicReference<ImmutableTopK> globalTopK) {
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

                    // 1. FAST-PATH (Leitura Volátil Pura): Evita disputa de escrita e alocações desnecessárias
                    ImmutableTopK currentTopK = globalTopK.get();
                    if (currentTopK.list.size() == k && dist >= currentTopK.list.get(k - 1).distance) {
                        continue; // Elemento descartado imediatamente sem tocar no laço CAS
                    }

                    // 2. SLOW-PATH: Elemento qualificado. Entra no laço otimista de atualização atômica (CAS)
                    DistanceRecord record = new DistanceRecord(current, dist);
                    while (true) {
                        currentTopK = globalTopK.get();
                        ImmutableTopK nextTopK = currentTopK.tryUpdate(record);

                        if (nextTopK == null) {
                            break; // Outra thread mudou o estado e este registro não serve mais
                        }

                        // Executa a troca atômica se o estado não mudou no meio do caminho
                        if (globalTopK.compareAndSet(currentTopK, nextTopK)) {
                            break;
                        }
                    }
                }
            }
        } catch (IOException e) {
            System.err.println("Error processing chunk: " + e.getMessage());
        }
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
     * Iterates over elements inside the list, maps occurrence scores, and determines the modes.
     * @param topK Immutable list containing the global nearest dataset entries.
     * @return String holding the winner classification label name.
     */
    private String majorityVote(List<DistanceRecord> topK) {
        Map<String, Integer> freq = new HashMap<>();
        for (DistanceRecord r : topK)
            freq.merge(r.neighbor.getLabel(), 1, Integer::sum);
        return Collections.max(freq.entrySet(), Map.Entry.comparingByValue()).getKey();
    }
}