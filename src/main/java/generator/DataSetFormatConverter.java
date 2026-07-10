package generator;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

public class DataSetFormatConverter {

    private static final String CSV_PATH = "dataset_high_dim.csv";

    private static final int NUM_FEATURES = 5;

    public static void main(String[] args) {
        String basePath = CSV_PATH.endsWith(".csv")
                ? CSV_PATH.substring(0, CSV_PATH.length() - 4)
                : CSV_PATH;

        SparkSession spark = SparkSession.builder()
                .appName("DataSet-Format-Converter")
                .master("local[*]")
                .getOrCreate();

        try {
            StructField[] fields = new StructField[NUM_FEATURES + 1];
            for (int i = 0; i < NUM_FEATURES; i++) {
                fields[i] = DataTypes.createStructField("feature_" + (i + 1), DataTypes.DoubleType, true);
            }
            fields[NUM_FEATURES] = DataTypes.createStructField("label", DataTypes.StringType, true);
            StructType schema = DataTypes.createStructType(fields);

            System.out.println("> lendo CSV com schema explícito: " + CSV_PATH);
            Dataset<Row> df = spark.read()
                    .format("csv")
                    .option("header", "true")
                    .schema(schema)
                    .load(CSV_PATH);

            df.cache();
            long rowCount = df.count();
            System.out.println("> " + rowCount + " linhas lidas, gravando nos formatos de saída...");

            String parquetPath = basePath + ".parquet";
            System.out.println("> escrevendo Parquet em: " + parquetPath);
            df.write().mode("overwrite").format("parquet").save(parquetPath);

            String orcPath = basePath + ".orc";
            System.out.println("> escrevendo ORC em: " + orcPath);
            df.write().mode("overwrite").format("orc").save(orcPath);

            String jsonPath = basePath + ".json";
            System.out.println("> escrevendo JSON em: " + jsonPath);
            df.write().mode("overwrite").format("json").save(jsonPath);

            System.out.println("> conversão concluída.");
        } finally {
            spark.stop();
        }
    }
}
