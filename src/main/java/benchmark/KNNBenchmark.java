package benchmark;

import generator.DataSetGenerator;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;

import java.util.ArrayList;
import java.util.concurrent.TimeUnit;

import serial.KNNSerial;
import platform.KNNPlatform;
import virtual.KNNVirtual;
import sync.KNNSync;
import semaphore.KNNSemaphore;
import lock.KNNLock;
import hibrid.KNNHibrid;
import barrier.KNNBarrier;
import atomic.KNNAtomic;
import classes.Neighbor;

@BenchmarkMode({Mode.Throughput, Mode.AverageTime})
@OutputTimeUnit(TimeUnit.SECONDS)
@Warmup(iterations = 1, time = 2, timeUnit = TimeUnit.SECONDS)
@Measurement(iterations = 5, time = 3, timeUnit = TimeUnit.SECONDS)
@Fork(0)
@State(Scope.Benchmark)
public class KNNBenchmark {

    @Param({"21"})
    private int k;

    private String filePath;
    private Neighbor targetInstance;

    private KNNSerial knnSerial;
    private KNNPlatform knnPlatform;
    private KNNVirtual knnVirtual;
    private KNNSync knnSync;
    private KNNSemaphore knnSemaphore;
    private KNNLock knnLock;
    private KNNHibrid knnHibrid;
    private KNNBarrier knnBarrier;
    private KNNAtomic knnAtomic;

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
        this.knnSync = new KNNSync();
        this.knnSemaphore = new KNNSemaphore();
        this.knnLock = new KNNLock();
        this.knnHibrid = new KNNHibrid();
        this.knnBarrier = new KNNBarrier();
        this.knnAtomic = new KNNAtomic();
    }

    @Benchmark
    public void test01_Serial(Blackhole bh) {
        bh.consume(knnSerial.predictStream(filePath, targetInstance, k));
    }

    @Benchmark
    public void test02_Platform(Blackhole bh) {
        bh.consume(knnPlatform.predictStream(filePath, targetInstance, k));
    }

    @Benchmark
    public void test03_Virtual(Blackhole bh) {
        bh.consume(knnVirtual.predictStream(filePath, targetInstance, k));
    }

    @Benchmark
    public void test04_Hibrid(Blackhole bh) {
        bh.consume(knnHibrid.predictStream(filePath, targetInstance, k));
    }

    @Benchmark
    public void test05_Synchronized(Blackhole bh) {
        bh.consume(knnSync.predictStream(filePath, targetInstance, k));
    }

    @Benchmark
    public void test06_Lock(Blackhole bh) {
        bh.consume(knnLock.predictStream(filePath, targetInstance, k));
    }

    @Benchmark
    public void test07_Semaphore(Blackhole bh) {
        bh.consume(knnSemaphore.predictStream(filePath, targetInstance, k));
    }

    @Benchmark
    public void test08_Barrier(Blackhole bh) {
        bh.consume(knnBarrier.predictStream(filePath, targetInstance, k));
    }

    @Benchmark
    public void test09_Atomic(Blackhole bh) {
        bh.consume(knnAtomic.predictStream(filePath, targetInstance, k));
    }

    public static void main(String[] args) throws Exception {
        org.openjdk.jmh.Main.main(args);
    }
}