package benchmark;

import generator.DataSetGenerator;
import org.apache.spark.sql.SparkSession;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;

import java.util.ArrayList;
import java.util.concurrent.TimeUnit;

import serial.KNNSerial;
import platform.KNNPlatform;
import platform.KNNSpark;
import platform.KNNSparkCsvFixedSchema;
import spark.KNNSparkJson;
import virtual.KNNVirtual;
import sync.KNNSync;
import semaphore.KNNSemaphore;
import lock.KNNLock;
import hibrid.KNNHibrid;
import barrier.KNNBarrier;
import atomic.KNNAtomic;
import executor.KNNExecutor;
import executor.KNNCallableFuture;
import executor.KNNForkJoin;
import executor.KNNParallelStream;
import executor.KNNCompletableFuture;
import executor.KNNStructuredConcurrency;
import classes.Neighbor;

@BenchmarkMode({Mode.Throughput, Mode.AverageTime})
@OutputTimeUnit(TimeUnit.SECONDS)
@Warmup(iterations = 3, time = 2, timeUnit = TimeUnit.SECONDS)
@Measurement(iterations = 5, time = 3, timeUnit = TimeUnit.SECONDS)
@Fork(value = 1, jvmArgsAppend = {"--enable-preview"})
@State(Scope.Benchmark)
public class KNNBenchmark {

    @Param({"21"})
    private int k;

    private String csvPath;
    private String jsonPath;

    private Neighbor targetInstance;
    private double[] targetArray;

    private KNNSerial knnSerial;
    private KNNPlatform knnPlatform;
    private KNNVirtual knnVirtual;
    private KNNSync knnSync;
    private KNNSemaphore knnSemaphore;
    private KNNLock knnLock;
    private KNNHibrid knnHibrid;
    private KNNBarrier knnBarrier;
    private KNNAtomic knnAtomic;
    private KNNForkJoin knnForkJoin;
    private KNNStructuredConcurrency knnStructuredConcurrency;
    private KNNExecutor knnExecutor;
    private KNNCallableFuture knnCallableFuture;
    private KNNParallelStream knnParallelStream;
    private KNNCompletableFuture knnCompletableFuture;

    private SparkSession sparkSession;
    private KNNSpark knnSpark;
    private KNNSparkCsvFixedSchema knnSparkCsvFixedSchema;
    private KNNSparkJson knnSparkJson;

    @Setup(Level.Trial)
    public void setupTrial() {
        this.csvPath = "dataset_high_dim.csv";
        this.jsonPath = "dataset_high_dim.json";

        int numFeatures = DataSetGenerator.NUM_FEATURES;

        ArrayList<Double> targetValues = new ArrayList<>();
        this.targetArray = new double[numFeatures];
        for (int i = 0; i < numFeatures; i++) {
            targetValues.add(500.0);
            this.targetArray[i] = 500.0;
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
        this.knnForkJoin = new KNNForkJoin();
        this.knnStructuredConcurrency = new KNNStructuredConcurrency();
        this.knnExecutor = new KNNExecutor();
        this.knnCallableFuture = new KNNCallableFuture();
        this.knnParallelStream = new KNNParallelStream();
        this.knnCompletableFuture = new KNNCompletableFuture();

        this.sparkSession = SparkSession.builder()
                .appName("KNN-JMH-Benchmark")
                .master("local[*]")
                .getOrCreate();

        this.knnSpark = new KNNSpark(sparkSession);
        this.knnSparkCsvFixedSchema = new KNNSparkCsvFixedSchema(sparkSession);
        this.knnSparkJson = new KNNSparkJson(sparkSession);
    }

    @TearDown(Level.Trial)
    public void tearDownTrial() {
        if (sparkSession != null) {
            sparkSession.stop();
        }
    }

    @Benchmark
    public void test01_Serial(Blackhole bh) {
        bh.consume(knnSerial.predictStream(csvPath, targetInstance, k));
    }

    @Benchmark
    public void test02_Platform(Blackhole bh) {
        bh.consume(knnPlatform.predictStream(csvPath, targetInstance, k));
    }

    @Benchmark
    public void test03_Virtual(Blackhole bh) {
        bh.consume(knnVirtual.predictStream(csvPath, targetInstance, k));
    }

    @Benchmark
    public void test04_Hibrid(Blackhole bh) {
        bh.consume(knnHibrid.predictStream(csvPath, targetInstance, k));
    }

    @Benchmark
    public void test05_Synchronized(Blackhole bh) {
        bh.consume(knnSync.predictStream(csvPath, targetInstance, k));
    }

    @Benchmark
    public void test06_Lock(Blackhole bh) {
        bh.consume(knnLock.predictStream(csvPath, targetInstance, k));
    }

    @Benchmark
    public void test07_Semaphore(Blackhole bh) {
        bh.consume(knnSemaphore.predictStream(csvPath, targetInstance, k));
    }

    @Benchmark
    public void test08_Barrier(Blackhole bh) {
        bh.consume(knnBarrier.predictStream(csvPath, targetInstance, k));
    }

    @Benchmark
    public void test09_Atomic(Blackhole bh) {
        bh.consume(knnAtomic.predictStream(csvPath, targetInstance, k));
    }

    @Benchmark
    public void test10_ForkJoin(Blackhole bh) {
        bh.consume(knnForkJoin.predictStream(csvPath, targetInstance, k));
    }

    @Benchmark
    public void test11_StructuredConcurrency(Blackhole bh) {
        bh.consume(knnStructuredConcurrency.predictStream(csvPath, targetInstance, k));
    }

    @Benchmark
    public void test12_SparkCsvInferSchema(Blackhole bh) {
        bh.consume(knnSpark.predict(csvPath, targetArray, k));
    }

    @Benchmark
    public void test13_SparkCsvFixedSchema(Blackhole bh) {
        bh.consume(knnSparkCsvFixedSchema.predict(csvPath, targetArray, k));
    }

    @Benchmark
    public void test14_SparkJson(Blackhole bh) {
        bh.consume(knnSparkJson.predict(jsonPath, targetArray, k));
    }

    @Benchmark
    public void test15_Executor(Blackhole bh) {
        bh.consume(knnExecutor.predictStream(csvPath, targetInstance, k));
    }

    @Benchmark
    public void test16_CallableFuture(Blackhole bh) {
        bh.consume(knnCallableFuture.predictStream(csvPath, targetInstance, k));
    }

    @Benchmark
    public void test17_ParallelStream(Blackhole bh) {
        bh.consume(knnParallelStream.predictStream(csvPath, targetInstance, k));
    }

    @Benchmark
    public void test18_CompletableFuture(Blackhole bh) {
        bh.consume(knnCompletableFuture.predictStream(csvPath, targetInstance, k));
    }

    public static void main(String[] args) throws Exception {
        org.openjdk.jmh.Main.main(args);
    }
}
