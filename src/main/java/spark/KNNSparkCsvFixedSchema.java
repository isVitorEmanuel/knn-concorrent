package platform;

import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.sql.Column;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.api.java.UDF1;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import java.util.Arrays;

import static org.apache.spark.sql.functions.callUDF;
import static org.apache.spark.sql.functions.col;
import static org.apache.spark.sql.functions.desc;
import static org.apache.spark.sql.functions.struct;

/**
 * @class KNNSparkCsvFixedSchema
 * @brief KNN variant in Spark that ingests the dataset in CSV format, but without using
 * {@code inferSchema=true}. Unlike {@link KNNSpark}, which lets Spark
 * make a full pass over the data just to discover column types,
 * this class declares the schema explicitly (derived from {@code target.length},
 * just like {@link KNNSparkJson}), eliminating that extra pass for CSV as well.
 * <p>
 * The rest of the pipeline (distance via UDF, orderBy+limit for the top-k, groupBy+count
 * for majority voting) is intentionally identical to the other variants, in order to isolate
 * the ingestion strategy as the sole variable of the experiment.
 */
public class KNNSparkCsvFixedSchema {

    private final SparkSession spark;
    private final boolean ownsSession;

    public KNNSparkCsvFixedSchema() {
        this(SparkSession.builder()
                .appName("KNN-Spark-Csv-FixedSchema")
                .master("local[*]")
                .getOrCreate(), true);
    }

    public KNNSparkCsvFixedSchema(SparkSession spark) {
        this(spark, false);
    }

    private KNNSparkCsvFixedSchema(SparkSession spark, boolean ownsSession) {
        this.spark = spark;
        this.ownsSession = ownsSession;
    }

    /**
     * @method predict
     * @brief Same as {@code KNNSpark.predict(String, double[], int)}, but reading the CSV
     * with an explicit schema (no inferSchema), built from
     * {@code target.length}.
     */
    public String predict(String csvPath, double[] target, int k) {
        Dataset<Row> raw = loadDataset(csvPath, target.length);
        return predict(raw, target, k);
    }

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
        spark.udf().register("euclideanDistanceCsvFixedSchema", euclideanDistance, DataTypes.DoubleType);

        Column[] featureColumns = new Column[featureCols.length];
        for (int i = 0; i < featureCols.length; i++) {
            featureColumns[i] = col(featureCols[i]);
        }

        Dataset<Row> withDistance = raw.withColumn("distance",
                callUDF("euclideanDistanceCsvFixedSchema", struct(featureColumns)));

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
     * @brief Ingests the CSV with an explicit schema, avoiding the extra pass that
     * Spark would otherwise make with {@code inferSchema=true}. Assumes a header row (used only
     * to skip the first line; column names come from the declared schema, not from the header).
     * @param csvPath     Path to the CSV file.
     * @param numFeatures Number of feature columns (derived from {@code target.length}
     *                    in {@link #predict(String, double[], int)}).
     * @return The ingested DataFrame, already typed.
     */
    private Dataset<Row> loadDataset(String csvPath, int numFeatures) {
        StructField[] fields = new StructField[numFeatures + 1];
        for (int i = 0; i < numFeatures; i++) {
            fields[i] = DataTypes.createStructField("feature_" + (i + 1), DataTypes.DoubleType, true);
        }
        fields[numFeatures] = DataTypes.createStructField("label", DataTypes.StringType, true);
        StructType schema = DataTypes.createStructType(fields);

        return spark.read()
                .format("csv")
                .option("header", "true")
                .schema(schema)
                .load(csvPath);
    }

    public void close() {
        if (ownsSession) {
            spark.stop();
        }
    }
}