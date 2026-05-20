package benchmark;

import classes.Neighbor;
import generator.DataSetGenerator;
import serial.KNNSerial;
import platform.KNNPlatform;
import virtual.KNNVirtual;
import hibrid.KNNHibrid;
import lock.KNNLock;
import atomic.KNNAtomic;
import barrier.KNNBarrier;

import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;

import java.util.ArrayList;
import java.util.concurrent.TimeUnit;

@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.SECONDS)
@Warmup(iterations = 2, time = 2, timeUnit = TimeUnit.SECONDS)
@Measurement(iterations = 5, time = 3, timeUnit = TimeUnit.SECONDS)
@Fork(0)
@State(Scope.Benchmark)
public class KNNBenchmark {

    @Param({"3", "11", "51"})
    private int k;

    private String filePath;
    private Neighbor targetInstance;

    private KNNSerial knnSerial;
    private KNNPlatform knnPlatform;
    private KNNVirtual knnVirtual;
    private KNNHibrid knnHibrid;
    private KNNLock knnLock;
    private KNNAtomic knnAtomic;
    private KNNBarrier knnBarrier;

    @Setup(Level.Trial)
    public void setupTrial() {
        this.filePath = "dataset_high_dim.csv";

        int numFeatures = DataSetGenerator.NUM_FEATURES;
        ArrayList<Double> targetValues = new ArrayList<>();
        for (int i = 0; i < numFeatures; i++) {
            targetValues.add(500.0);
        }
        this.targetInstance = new Neighbor(targetValues, "Unknown");

        this.knnSerial = new KNNSerial();
        this.knnPlatform = new KNNPlatform();
        this.knnVirtual = new KNNVirtual();
        this.knnHibrid = new KNNHibrid();
        this.knnLock = new KNNLock();
        this.knnAtomic = new KNNAtomic();
        this.knnBarrier = new KNNBarrier();
    }

    @Benchmark
    public void testKNNSerial(Blackhole bh) {
        String result = knnSerial.predictStream(filePath, targetInstance, k);
        bh.consume(result);
    }

    @Benchmark
    public void testKNNPlatform(Blackhole bh) {
        String result = knnPlatform.predictStream(filePath, targetInstance, k);
        bh.consume(result);
    }

    @Benchmark
    public void testKNNVirtual(Blackhole bh) {
        String result = knnVirtual.predictStream(filePath, targetInstance, k);
        bh.consume(result);
    }

    @Benchmark
    public void testKNNHybrid(Blackhole bh) {
        String result = knnHibrid.predictStream(filePath, targetInstance, k);
        bh.consume(result);
    }

    @Benchmark
    public void testKNNLock(Blackhole bh) {
        String result = knnLock.predictStream(filePath, targetInstance, k);
        bh.consume(result);
    }

    @Benchmark
    public void testKNNAtomic(Blackhole bh) {
        String result = knnAtomic.predictStream(filePath, targetInstance, k);
        bh.consume(result);
    }

    @Benchmark
    public void testKNNCyclicBarrier(Blackhole bh) {
        String result = knnBarrier.predictStream(filePath, targetInstance, k);
        bh.consume(result);
    }
}