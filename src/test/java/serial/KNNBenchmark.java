package serial;

import classes.Neighbor;
import generator.DataSetGenerator;
import org.openjdk.jmh.annotations.*;
import java.util.ArrayList;
import java.util.concurrent.TimeUnit;

/**
 * Configurações do Benchmark:
 * - AverageTime: Mede o tempo médio de cada execução.
 * - Warmup: Roda 3 iterações "falsas" apenas para aquecer a JVM (não entram na média).
 * - Measurement: Roda 5 iterações reais que contarão para o resultado final.
 * - Fork: Roda tudo em 1 processo isolado da JVM.
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.SECONDS)
@State(Scope.Thread)
@Warmup(iterations = 3, time = 1, timeUnit = TimeUnit.SECONDS)
@Measurement(iterations = 5, time = 1, timeUnit = TimeUnit.SECONDS)
@Fork(0)
public class KNNBenchmark {

    private KNNSerial knn;
    private Neighbor target;
    private String path;
    private int k;

    // O @Setup roda UMA VEZ antes de começar as iterações do benchmark
    @Setup(Level.Trial)
    public void setUp() {
        path = "dataset_high_dim.csv";
        k = 21;
        knn = new KNNSerial();

        ArrayList<Double> targetValues = new ArrayList<>();
        for (int i = 0; i < DataSetGenerator.NUM_FEATURES; i++) {
            targetValues.add(500.0);
        }
        target = new Neighbor(targetValues, "Unknown");
    }

    // O @Benchmark é o método que será testado repetidas vezes
    @Benchmark
    public String testSerialPrediction() {
        return knn.predictStream(path, target, k);
    }

    // Método main para facilitar a execução via Maven
    public static void main(String[] args) throws Exception {
        org.openjdk.jmh.Main.main(args);
    }
}