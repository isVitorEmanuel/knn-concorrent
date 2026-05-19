package virtual;

import classes.Neighbor;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.concurrent.atomic.AtomicLong;

public class KNNVirtual {

    private static final int NUM_VIRTUAL_THREADS =
            Math.max(2, Runtime.getRuntime().availableProcessors() * 3);

    private record DistanceRecord(Neighbor neighbor, double distance) implements Comparable<DistanceRecord> {
        @Override
        public int compareTo(DistanceRecord other) {
            return Double.compare(this.distance, other.distance);
        }
    }

    private record ChunkBounds(long startByte, long endByte) {}

    public String predictStream(String filePath, Neighbor target, int k) {
        AtomicLong processedLines = new AtomicLong(0);
        String result = runParallel(filePath, target, k, NUM_VIRTUAL_THREADS, processedLines);

        System.out.printf(">>> Total lines processed (Platform): %d%n", processedLines.get());
        return result;
    }

    private String runParallel(String filePath, Neighbor target, int k,
                               int numThreads, AtomicLong processedLines) {

        List<ChunkBounds> chunks;
        try {
            chunks = computeChunks(filePath, numThreads);
            System.out.printf("[runParallel] %d chunks | target dim=%d%n",
                    chunks.size(), target.getValues().size());
        } catch (IOException e) {
            System.err.println("Error computing file chunks: " + e.getMessage());
            return "Unknown";
        }

        Thread.Builder builder = Thread.ofVirtual().name("knn-virtual-", 0);

        AtomicLong discardedLines = new AtomicLong(0);

        int numChunks = chunks.size();
        Thread[] threads = new Thread[numChunks];

        @SuppressWarnings("unchecked")
        PriorityQueue<DistanceRecord>[] results = new PriorityQueue[numChunks];

        for (int i = 0; i < numChunks; i++) {
            final int index = i;
            ChunkBounds chunk = chunks.get(i);

            threads[i] = builder.unstarted(() -> {
                results[index] = processChunk(filePath, chunk, target, k, discardedLines, processedLines);
            });
            threads[i].start();
        }

        for (Thread t : threads) {
            try {
                t.join();
            } catch (InterruptedException e) {
                System.err.println("Main thread interrupted. Stopping workers...");
                for (Thread worker : threads) {
                    if (worker != null) worker.interrupt();
                }
                Thread.currentThread().interrupt();
                return "Unknown";
            }
        }

        long discarded = discardedLines.get();
        if (discarded > 0) {
            System.err.printf("[WARNING] %d lines discarded — dimension mismatch.%n", discarded);
        }

        System.out.printf("[runParallel] finished — partial processed=%d discarded=%d%n",
                processedLines.get(), discarded);

        PriorityQueue<DistanceRecord> globalTopK =
                new PriorityQueue<>(k, Collections.reverseOrder());

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

    private PriorityQueue<DistanceRecord> processChunk(String filePath,
                                                       ChunkBounds chunk,
                                                       Neighbor target,
                                                       int k,
                                                       AtomicLong discardedLines,
                                                       AtomicLong processedLines) {

        PriorityQueue<DistanceRecord> localTopK =
                new PriorityQueue<>(k, Collections.reverseOrder());

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
                        discardedLines.incrementAndGet();
                        continue;
                    }

                    processedLines.incrementAndGet();

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