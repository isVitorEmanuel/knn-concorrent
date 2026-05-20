package semaphore;

import classes.Neighbor;
import generator.DataSetGenerator;
import org.openjdk.jmh.annotations.*;
import java.util.ArrayList;
import java.util.concurrent.TimeUnit;

@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.SECONDS)
@State(Scope.Thread)
@Warmup(iterations = 3, time = 1, timeUnit = TimeUnit.SECONDS)
@Measurement(iterations = 5, time = 1, timeUnit = TimeUnit.SECONDS)
@Fork(0)
public class KNNBenchmark {

    private KNNSemaphore knn;
    private Neighbor target;
    private String path;
    private int k;

    @Setup(Level.Trial)
    public void setUp() {
        path = "dataset_high_dim.csv";
        k = 21;
        knn = new KNNSemaphore();

        ArrayList<Double> targetValues = new ArrayList<>();
        for (int i = 0; i < DataSetGenerator.NUM_FEATURES; i++) {
            targetValues.add(500.0);
        }
        target = new Neighbor(targetValues, "Unknown");
    }

    @Benchmark
    public String testSerialPrediction() {
        return knn.predictStream(path, target, k);
    }

    public static void main(String[] args) throws Exception {
        org.openjdk.jmh.Main.main(args);
    }
}