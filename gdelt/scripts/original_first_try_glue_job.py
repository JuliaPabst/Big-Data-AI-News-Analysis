import sys
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.utils import getResolvedOptions
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.functions import col

# 1. Initialize Glue Context
# This is the standard boilerplate for AWS Glue
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session

print("--- Starting Glue Job Locally ---")

# 2. Load Data
# Using the Docker container I created earlier
input_path = "file:///home/glue_user/workspace/data/gdelt_ml_features.parquet"
df = spark.read.parquet(input_path)

# 3. Data Preprocessing
# Convert the string label ('week_feb', 'week_may') into numbers (0, 1)
label_indexer = StringIndexer(inputCol="label_week", outputCol="label")
df_indexed = label_indexer.fit(df).transform(df)

# Define the list of features to use for training
feature_cols = [
    'url_length', 
    'num_themes', 
    'num_orgs', 
    'k_openai', 
    'k_google', 
    'k_anthropic',
    'v2tone_1', # Tone (Pos - Neg)
    'v2tone_2', # Positive Score
    'v2tone_3', # Negative Score
    'v2tone_4', # Polarity
    'v2tone_5', # Activity Ref Density
    'v2tone_6', # Self/Group Ref Density
    'v2tone_7'  # Word Count
]

# Handle missing values (fill with 0 just in case)
df_clean = df_indexed.fillna(0, subset=feature_cols)

# Combine all input columns into a single "features" vector column
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
data_final = assembler.transform(df_clean)

# 4. Split Data into Train and Test sets
train_data, test_data = data_final.randomSplit([0.8, 0.2], seed=42)

# 5. Train the Model (Logistic Regression)
# I use Logistic Regression because it's easy to interpret feature importance
lr = LogisticRegression(featuresCol="features", labelCol="label")
lr_model = lr.fit(train_data)

# 6. Evaluate the Model
predictions = lr_model.transform(test_data)
evaluator = BinaryClassificationEvaluator(labelCol="label")
auc = evaluator.evaluate(predictions)

print(f"\nModel Performance (AUC): {auc:.4f}")
print("A value close to 1.0 means the model can perfectly distinguish Feb from May.")
print("A value of 0.5 means the model is guessing randomly.\n")

# 7. Extract Insights (Coefficients)
# This answers: Which information can I read from it?
coefficients = lr_model.coefficients
intercept = lr_model.intercept

print("--- Feature Importance (Coefficients) ---")
print(f"{'Feature':<20} | {'Coefficient':<10} | {'Impact'}")
print("-" * 50)

# Get the mapping of label 0/1 to original strings
label_mapping = df_indexed.schema["label"].metadata["ml_attr"]["vals"]
positive_label = label_mapping[1] 

for i, feature in enumerate(feature_cols):
    coeff = coefficients[i]
    # Simple interpretation string
    if coeff > 0.1:
        impact = f"Linked to {positive_label}"
    elif coeff < -0.1:
        impact = f"Linked to {label_mapping[0]}"
    else:
        impact = "Neutral / Low Impact"
        
    print(f"{feature:<20} | {coeff:>10.4f} | {impact}")

print("-" * 50)
print("Done.")