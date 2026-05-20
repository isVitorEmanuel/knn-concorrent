package serial;

import classes.Neighbor;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

/**
 * @class KNNSerial
 * @brief Implements the K-Nearest Neighbors algorithm in a serial (single-threaded) manner.
 * (CPU-bound naive implementation for baseline comparison).
 */
public class KNNSerial {

    private static class DistanceRecord implements Comparable<DistanceRecord> {
        Neighbor neighbor;
        Double distance;

        public DistanceRecord(Neighbor neighbor, Double distance) {
            this.neighbor = neighbor;
            this.distance = distance;
        }

        @Override
        public int compareTo(DistanceRecord other) {
            return this.distance.compareTo(other.distance);
        }
    }

    public String predictStream(String filePath, Neighbor target, int k) {
        PriorityQueue<DistanceRecord> pq = new PriorityQueue<>(Collections.reverseOrder());

        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            boolean firstLine = true;

            while ((line = br.readLine()) != null) {
                if (firstLine) {
                    firstLine = false;
                    continue;
                }

                Neighbor current = parseLineToNeighbor(line);
                if (current == null) continue;

                if (current.getValues().size() != target.getValues().size()) {
                    continue;
                }

                Double dist = calculateEuclideanDistance(target, current);
                pq.add(new DistanceRecord(current, dist));

                if (pq.size() > k) {
                    pq.poll();
                }
            }
        } catch (IOException e) {
            System.err.println("error reading file: " + e.getMessage());
        }

        if (pq.isEmpty()) return "Unknown";

        Map<String, Integer> labelFrequencies = new HashMap<>();
        for (DistanceRecord record : pq) {
            String label = record.neighbor.getLabel();
            labelFrequencies.put(label, labelFrequencies.getOrDefault(label, 0) + 1);
        }

        return Collections.max(labelFrequencies.entrySet(), Map.Entry.comparingByValue()).getKey();
    }

    private Neighbor parseLineToNeighbor(String line) {
        String[] parts = line.split(",");
        if (parts.length < 2) return null;

        ArrayList<Double> values = new ArrayList<>();
        try {
            for (int i = 0; i < parts.length - 1; i++) {
                values.add(Double.valueOf(parts[i].replaceAll("\\s+", "")));
            }
            return new Neighbor(values, parts[parts.length - 1].trim());
        } catch (Exception e) {
            return null;
        }
    }

    private Double calculateEuclideanDistance(Neighbor target, Neighbor dataPoint) {
        Double sum = 0.0;
        ArrayList<Double> targetValues = target.getValues();
        ArrayList<Double> dataValues = dataPoint.getValues();

        for (int i = 0; i < targetValues.size(); i++) {
            Double diff = targetValues.get(i) - dataValues.get(i);
            sum += Math.pow(diff, 2.0);
        }
        return Math.sqrt(sum);
    }
}