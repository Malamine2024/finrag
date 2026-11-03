"""How much of each annual report repeats the previous year's? A corpus analysis with PySpark.

Finding 5 of the README says that annual reports repeat the same paragraphs every year, so that the search
returned the right page of the wrong year until company and year filters were added. This script measures it
on the whole corpus, with Spark:

1. Corpus statistics per report (company x fiscal year): passages, pages, share of table passages, length.
2. Near-duplicate passages between consecutive reports of the same company, with MinHash LSH (Spark ML):
   each text passage becomes a set of word 3-grams; two passages are near-duplicates when the Jaccard
   similarity of their sets is at least --threshold (0.8 by default). For every report, the share of its
   passages that already appear, almost word for word, in the previous year's report.

    python -m finrag.spark_corpus                  # whole corpus (about 51,000 passages; a few minutes)
    python -m finrag.spark_corpus --threshold 0.9  # stricter near-duplicates

Output: data/processed/corpus_stats.csv and corpus_repetition.csv (written with pandas: Spark itself writes
nothing to disk, which avoids the Hadoop "winutils" setup on Windows).
Requirements: pip install pyspark, and Java 17 or later (JAVA_HOME).
"""
import argparse
import os
import sys

from .config import CHUNKS_FILE, PROCESSED_DIR

# Windows: Spark starts its Python workers with "python", which may not be this venv's interpreter.
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

MIN_CHARS = 200          # shorter passages (titles, single lines) repeat by nature and are left out
HEADER_RE = r"^\[[^\]]*\]\s*"      # the context header of a chunk: "[Table | Company: ... | Title: ...]"


def spark_session(memory: str):
    from pyspark.sql import SparkSession
    spark = (SparkSession.builder.master("local[*]").appName("finrag-corpus")
             .config("spark.driver.memory", memory)
             .config("spark.sql.shuffle.partitions", "64")
             .config("spark.ui.showConsoleProgress", "false")
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def load(spark, path):
    from pyspark.sql import functions as F
    df = spark.read.json(path.resolve().as_uri())
    return (df.withColumn("body", F.trim(F.regexp_replace("text", HEADER_RE, "")))
              .withColumn("chars", F.length("body"))
              .withColumn("year", F.col("year").cast("int")))


def corpus_stats(df):
    from pyspark.sql import functions as F
    return (df.groupBy("company", "year")
              .agg(F.count("*").alias("passages"),
                   F.countDistinct("page").alias("pages"),
                   F.round(F.avg(F.when(F.col("kind") == "table", 1).otherwise(0)) * 100, 1).alias("table_pct"),
                   F.round(F.avg("chars"), 0).cast("int").alias("avg_chars"))
              .orderBy("company", "year"))


def repetition(df, threshold: float, hash_tables: int):
    """Share of each report's text passages that have a near-duplicate in the previous year's report."""
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import HashingTF, MinHashLSH, NGram, RegexTokenizer
    from pyspark.sql import functions as F

    text = df.filter((F.col("kind") == "text") & (F.col("chars") >= MIN_CHARS)).select("id", "company", "year", "body")
    features = Pipeline(stages=[
        # Letters and digits of any language (French accents included), lower case, words of 2+ characters.
        RegexTokenizer(inputCol="body", outputCol="words", pattern=r"[^\p{L}\p{N}]+", minTokenLength=2),
        NGram(n=3, inputCol="words", outputCol="shingles"),
        HashingTF(inputCol="shingles", outputCol="features", numFeatures=1 << 20, binary=True),
    ]).fit(text).transform(text).filter(F.size("shingles") > 0).cache()

    lsh = MinHashLSH(inputCol="features", outputCol="hashes", numHashTables=hash_tables).fit(features)
    # Candidate pairs within the Jaccard distance, then only the same company and consecutive years.
    pairs = (lsh.approxSimilarityJoin(features, features, 1 - threshold, distCol="distance")
                .filter((F.col("datasetA.company") == F.col("datasetB.company"))
                        & (F.col("datasetB.year") == F.col("datasetA.year") + 1))
                .select(F.col("datasetB.id").alias("id"), F.col("datasetB.company").alias("company"),
                        F.col("datasetB.year").alias("year"), "distance",
                        F.col("datasetA.body").alias("previous"), F.col("datasetB.body").alias("current")))
    repeated = pairs.select("id").distinct().withColumn("repeated", F.lit(1))
    per_report = (features.select("id", "company", "year")
                  .join(repeated, "id", "left")
                  .groupBy("company", "year")
                  .agg(F.count("*").alias("text_passages"), F.sum(F.coalesce("repeated", F.lit(0))).alias("repeated"))
                  .withColumn("repeated_pct", F.round(F.col("repeated") / F.col("text_passages") * 100, 1)))
    # The first report of each company has no previous year to compare with.
    first = features.groupBy("company").agg(F.min("year").alias("first_year"))
    per_report = (per_report.join(first, "company").filter(F.col("year") > F.col("first_year"))
                  .drop("first_year").orderBy("company", "year"))
    # The closest pair that is not an exact copy: what "almost word for word" looks like.
    example = pairs.filter(F.col("distance") > 0).orderBy("distance").limit(1)
    return per_report, example


def to_pandas(sdf):
    """A small Spark result as a pandas DataFrame (collect: no pyarrow or distutils needed)."""
    import pandas as pd
    return pd.DataFrame([r.asDict() for r in sdf.collect()], columns=sdf.columns)


def main() -> None:
    parser = argparse.ArgumentParser(description="Corpus statistics and year-to-year repetition, with PySpark.")
    parser.add_argument("--threshold", type=float, default=0.8, help="Jaccard similarity of near-duplicates")
    parser.add_argument("--hash-tables", type=int, default=5, help="MinHash tables (more = fewer missed pairs)")
    parser.add_argument("--memory", default="4g", help="Spark driver memory")
    args = parser.parse_args()
    if not CHUNKS_FILE.exists():
        raise SystemExit(f"{CHUNKS_FILE} not found: run python -m finrag.ingest first")

    spark = spark_session(args.memory)
    df = load(spark, CHUNKS_FILE).cache()
    print(f"{df.count():,} passages in {df.select('company', 'year').distinct().count()} reports\n")

    stats = to_pandas(corpus_stats(df))
    print(stats.to_string(index=False))

    per_report, example = repetition(df, args.threshold, args.hash_tables)
    rep = to_pandas(per_report)
    print(f"\nText passages (>= {MIN_CHARS} characters) with a near-duplicate (Jaccard >= {args.threshold}) "
          "in the previous year's report:")
    print(rep.pivot(index="company", columns="year", values="repeated_pct").to_string(na_rep="-"))
    by_company = rep.groupby("company")[["repeated", "text_passages"]].sum()
    by_company["repeated_pct"] = (by_company["repeated"] / by_company["text_passages"] * 100).round(1)
    total = rep["repeated"].sum() / rep["text_passages"].sum() * 100
    print("\nBy company:")
    print(by_company.to_string())
    print(f"\nWhole corpus: {total:.1f}% of text passages repeat the previous year's report almost word for word.")

    ex = to_pandas(example)
    if len(ex):
        e = ex.iloc[0]
        print(f"\nClosest non-identical pair ({e['company']} {e['year'] - 1} -> {e['year']}, Jaccard {1 - e['distance']:.2f}):")
        print(f"  {e['year'] - 1}: {e['previous'][:200]}...")
        print(f"  {e['year']}: {e['current'][:200]}...")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    stats.to_csv(PROCESSED_DIR / "corpus_stats.csv", index=False, sep=";")
    rep.to_csv(PROCESSED_DIR / "corpus_repetition.csv", index=False, sep=";")
    print(f"\n-> {PROCESSED_DIR / 'corpus_stats.csv'}\n-> {PROCESSED_DIR / 'corpus_repetition.csv'}")
    spark.stop()


if __name__ == "__main__":
    main()
