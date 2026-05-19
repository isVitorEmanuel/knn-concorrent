package hibrid;

import classes.Neighbor;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicLong;

public class KNNHibrid {

    private static final int NUM_PLATFORM_THREADS =
            Math.max(2, Runtime.getRuntime().availableProcessors());

    // Tamanho do lote de linhas enviado para reduzir o overhead de sincronização
    private static final int BATCH_SIZE = 5_000;
    private static final int QUEUE_CAPACITY = 200;

    private record DistanceRecord(Neighbor neighbor, double distance) implements Comparable<DistanceRecord> {
        @Override
        public int compareTo(DistanceRecord other) {
            return Double.compare(this.distance, other.distance);
        }
    }

    public String predictStream(String filePath, Neighbor target, int k) {
        System.out.printf("[Hybrid Batched] 1 Virtual (Producer) | %d Platform (Consumers)%n", NUM_PLATFORM_THREADS);

        AtomicLong processedLines = new AtomicLong(0);
        String result = runHybrid(filePath, target, k, processedLines);

        System.out.printf(">>> Total lines processed (Hybrid): %d%n", processedLines.get());
        return result;
    }

    private String runHybrid(String filePath, Neighbor target, int k, AtomicLong processedLines) {
        // Agora a fila trafega LISTAS de Strings (Lotes) em vez de linhas individuais
        BlockingQueue<List<String>> queue = new ArrayBlockingQueue<>(QUEUE_CAPACITY);
        AtomicLong discardedLines = new AtomicLong(0);

        @SuppressWarnings("unchecked")
        PriorityQueue<DistanceRecord>[] results = new PriorityQueue[NUM_PLATFORM_THREADS];
        Thread[] consumers = new Thread[NUM_PLATFORM_THREADS];

        // 1. INICIALIZA AS CONSUMIDORAS (PLATFORM THREADS)
        Thread.Builder platformBuilder = Thread.ofPlatform().name("knn-consumer-", 0);

        for (int i = 0; i < NUM_PLATFORM_THREADS; i++) {
            final int index = i;
            consumers[i] = platformBuilder.unstarted(() -> {
                PriorityQueue<DistanceRecord> localTopK = new PriorityQueue<>(k, Collections.reverseOrder());
                long localProcessed = 0;
                long localDiscarded = 0;

                try {
                    while (true) {
                        List<String> batch = queue.take(); // Retira um lote inteiro da fila

                        // Usamos uma lista vazia como Poison Pill (sinal de término)
                        if (batch.isEmpty()) {
                            break;
                        }

                        // Processa o lote localmente sem nenhuma concorrência ou lock
                        for (String line : batch) {
                            if (line.isBlank()) continue;

                            Neighbor current = parseLineToNeighbor(line);
                            if (current == null) continue;

                            if (current.getValues().size() != target.getValues().size()) {
                                localDiscarded++;
                                continue;
                            }

                            localProcessed++;
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
                } finally {
                    // Atualiza os contadores globais apenas UMA vez por thread
                    processedLines.addAndGet(localProcessed);
                    discardedLines.addAndGet(localDiscarded);
                }
            });
            consumers[i].start();
        }

        // 2. INICIALIZA A PRODUTORA (UMA ÚNICA VIRTUAL THREAD)
        Thread producer = Thread.ofVirtual().name("knn-producer").start(() -> {
            try (BufferedReader reader = new BufferedReader(
                    new InputStreamReader(new FileInputStream(filePath), StandardCharsets.UTF_8))) {

                String line = reader.readLine(); // Pula o Header do CSV
                if (line == null) return;

                List<String> currentBatch = new ArrayList<>(BATCH_SIZE);

                while ((line = reader.readLine()) != null) {
                    currentBatch.add(line);

                    // Quando o lote enche, envia para a fila
                    if (currentBatch.size() == BATCH_SIZE) {
                        queue.put(currentBatch);
                        currentBatch = new ArrayList<>(BATCH_SIZE);
                    }
                }

                // Envia o último lote residual, se houver
                if (!currentBatch.isEmpty()) {
                    queue.put(currentBatch);
                }

            } catch (IOException e) {
                System.err.println("Error reading file: " + e.getMessage());
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            } finally {
                // Envia uma lista vazia (Poison Pill) para cada consumidor finalizar
                for (int i = 0; i < NUM_PLATFORM_THREADS; i++) {
                    try {
                        queue.put(Collections.emptyList());
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                    }
                }
            }
        });

        // 3. AGUARDA A FINALIZAÇÃO
        try {
            producer.join();
            for (Thread t : consumers) {
                t.join();
            }
        } catch (InterruptedException e) {
            System.err.println("Main thread interrupted. Cleaning up...");
            producer.interrupt();
            for (Thread t : consumers) {
                if (t != null) t.interrupt();
            }
            Thread.currentThread().interrupt();
            return "Unknown";
        }

        long discarded = discardedLines.get();
        if (discarded > 0) {
            System.err.printf("[WARNING] %d lines discarded — dimension mismatch.%n", discarded);
        }

        // 4. MERGE GLOBAL DOS TOP-K
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