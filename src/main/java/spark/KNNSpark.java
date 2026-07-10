package platform;

import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.sql.Column;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.api.java.UDF1;
import org.apache.spark.sql.types.DataTypes;

import java.util.Arrays;

import static org.apache.spark.sql.functions.callUDF;
import static org.apache.spark.sql.functions.col;
import static org.apache.spark.sql.functions.desc;
import static org.apache.spark.sql.functions.struct;

/**
 * @class KNNSpark
 * @brief Implements the K-Nearest Neighbors algorithm using Apache Spark's DataFrame API.
 * This is the Spark counterpart of {@code platform.KNNPlatform}: instead of manually
 * splitting the file into byte chunks and coordinating platform threads, the ingestion,
 * distance calculation, ranking and majority vote are all expressed as Spark DataFrame
 * transformations/actions and executed by the Spark engine (locally, in this course's
 * setup, but the same code runs unmodified on a real cluster).
 * <p>
 * Pipeline:
 * <ol>
 *     <li>Ingestion: {@code spark.read().format("csv")...} loads the dataset directly
 *         into a {@code Dataset<Row>}, with schema inference, instead of manual
 *         RandomAccessFile/BufferedReader chunking.</li>
 *     <li>Transformation: a UDF computes the Euclidean distance of every row to the
 *         broadcast target vector; {@code withColumn} adds it as a new column.</li>
 *     <li>Transformation: {@code orderBy(distance).limit(k)} lets Spark's distributed
 *         sort pick the k nearest neighbors, replacing the manual PriorityQueue merge.</li>
 *     <li>Action: {@code groupBy(label).count()} + {@code orderBy(desc(count))} performs
 *         the majority vote, replacing the manual HashMap frequency count.</li>
 * </ol>
 */
public class KNNSpark {

    private final SparkSession spark;
    private final boolean ownsSession;

    /**
     * @brief Creates a KNNSpark instance backed by a new local SparkSession.
     * Useful for standalone runs / benchmarks. Call {@link #close()} when done.
     */
    public KNNSpark() {
        this(SparkSession.builder()
                .appName("KNN-Spark-DataFrame")
                .master("local[*]")
                .getOrCreate(), true);
    }

    /**
     * @brief Creates a KNNSpark instance reusing an existing SparkSession.
     * Preferred when the caller already manages the Spark lifecycle (e.g. inside
     * a larger benchmark that also runs other Spark-based variants).
     * @param spark An already-built SparkSession.
     */
    public KNNSpark(SparkSession spark) {
        this(spark, false);
    }

    private KNNSpark(SparkSession spark, boolean ownsSession) {
        this.spark = spark;
        this.ownsSession = ownsSession;
    }

    /**
     * @method predict
     * @brief Entry point mirroring {@code KNNPlatform.predictStream}: loads the CSV file
     * and classifies the target using the Spark DataFrame pipeline.
     * @param filePath Path to the CSV dataset file (header row + numeric feature columns
     *                 followed by a single trailing label column).
     * @param target   Feature vector to classify.
     * @param k        Number of nearest neighbors to consider.
     * @return The predicted class label, or "Unknown" if the dataset yields no valid rows.
     */
    public String predict(String filePath, double[] target, int k) {
        Dataset<Row> raw = loadDataset(filePath);
        return predict(raw, target, k);
    }

    /**
     * @method predict
     * @brief Same as {@link #predict(String, double[], int)} but accepts an already
     * loaded/cached DataFrame, so repeated queries against the same dataset don't pay
     * the ingestion cost again.
     * @param raw    The dataset as a DataFrame; last column is treated as the label.
     * @param target Feature vector to classify.
     * @param k      Number of nearest neighbors to consider.
     * @return The predicted class label, or "Unknown" if the dataset yields no valid rows.
     */
    public String predict(Dataset<Row> raw, double[] target, int k) {
        String[] columns = raw.columns();
        if (columns.length < 2) {
            System.err.println("Error: dataset must have at least one feature column and one label column.");
            return "Unknown";
        }

        String labelCol = columns[columns.length - 1];
        String[] featureCols = Arrays.copyOfRange(columns, 0, columns.length - 1);

        if (target.length != featureCols.length) {
            System.err.printf("Error: target dimension (%d) does not match dataset feature count (%d)%n",
                    target.length, featureCols.length);
            return "Unknown";
        }

        Broadcast<double[]> broadcastTarget =
                JavaSparkContext.fromSparkContext(spark.sparkContext()).broadcast(target);

        UDF1<Row, Double> euclideanDistance = (Row featureStruct) -> {
            double[] t = broadcastTarget.value();
            double sum = 0.0;
            for (int i = 0; i < t.length; i++) {
                Object raw0 = featureStruct.get(i);
                double v = ((Number) raw0).doubleValue();
                double diff = v - t[i];
                sum += diff * diff;
            }
            return Math.sqrt(sum);
        };
        spark.udf().register("euclideanDistance", euclideanDistance, DataTypes.DoubleType);

        Column[] featureColumns = new Column[featureCols.length];
        for (int i = 0; i < featureCols.length; i++) {
            featureColumns[i] = col(featureCols[i]);
        }

        Dataset<Row> withDistance = raw.withColumn("distance", callUDF("euclideanDistance", struct(featureColumns)));

        Dataset<Row> topK = withDistance.orderBy(col("distance").asc()).limit(k);

        Row winner = topK.groupBy(col(labelCol).alias("label"))
                .count()
                .orderBy(desc("count"))
                .first();

        if (winner == null) {
            System.err.println("Error: no valid neighbors found.");
            return "Unknown";
        }

        Object labelValue = winner.get(0);
        return labelValue == null ? "Unknown" : labelValue.toString();
    }

    /**
     * @method loadDataset
     * @brief Ingests the CSV file into a DataFrame, letting Spark handle partitioning and
     * schema inference instead of the manual byte-boundary chunking used by KNNPlatform.
     * @param filePath Path to the CSV file.
     * @return The ingested DataFrame.
     */
    private Dataset<Row> loadDataset(String filePath) {
        return spark.read()
                .format("csv")
                .option("header", "true")
                .option("inferSchema", "true")
                .load(filePath);
    }

    /**
     * @method close
     * @brief Stops the underlying SparkSession, if this instance created it.
     */
    public void close() {
        if (ownsSession) {
            spark.stop();
        }
    }
}
