import sys
import pandas as pd
# Use Agg backend to ensure images save even without a monitor/display
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from pyspark.sql.functions import col, explode, desc, lit
from pyspark.ml.feature import CountVectorizer, StringIndexer
from pyspark.ml.classification import LogisticRegression

# --- 1. SETUP ---
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
spark.sparkContext.setLogLevel("ERROR")

DATA_DIR = "file:///home/glue_user/workspace/data/"
OUTPUT_DIR = "/home/glue_user/workspace/result_markdowns/"
GRAPH_DIR = "/home/glue_user/workspace/graphs/"

print("--- Starting Theme Analysis ---")

# Load Data
df = spark.read.parquet(DATA_DIR + "gdelt_core_fixed.parquet")

# ==========================================
# PART A: DOMINANT THEMES PER COMPANY
# ==========================================
print("1. Analyzing Company Themes...")

def get_top_themes(company_col, company_name, limit=10):
    return df.filter(col(company_col) == 1) \
             .select(explode(col("themes_arr")).alias("theme")) \
             .groupBy("theme") \
             .count() \
             .orderBy(desc("count")) \
             .limit(limit) \
             .withColumn("Company", lit(company_name)) \
             .toPandas()

# Get data for each
df_google = get_top_themes("k_google", "Google", 8)
df_openai = get_top_themes("k_openai", "OpenAI", 8)
df_anthropic = get_top_themes("k_anthropic", "Anthropic", 8)

# Clean theme names
for d in [df_google, df_openai, df_anthropic]:
    if not d.empty:
        d['theme_short'] = d['theme'].apply(lambda x: x.split('_')[0] if len(x) > 15 else x)

# ==========================================
# PART B: ML ANALYSIS (FEB VS MAY)
# ==========================================
print("2. Training Theme Prediction Model...")

# 1. Prepare Data
df_ml = df.select("label_week", "themes_arr").filter(col("themes_arr").isNotNull())
cv = CountVectorizer(inputCol="themes_arr", outputCol="features", vocabSize=1000, minDF=5.0)
cv_model = cv.fit(df_ml)
vectorized_data = cv_model.transform(df_ml)

# 2. Index Labels
indexer = StringIndexer(inputCol="label_week", outputCol="label")
idx_model = indexer.fit(vectorized_data)
final_data = idx_model.transform(vectorized_data)

# 3. Train
lr = LogisticRegression(featuresCol="features", labelCol="label", regParam=0.1)
lr_model = lr.fit(final_data)

# 4. Extract Top Themes
vocab = cv_model.vocabulary
weights = lr_model.coefficients.toArray()
coeff_df = pd.DataFrame({'Theme': vocab, 'Score': weights})

top_feb = coeff_df.sort_values(by='Score', ascending=False).head(8)
top_may = coeff_df.sort_values(by='Score', ascending=True).head(8)

# ==========================================
# PART C: GENERATE MARKDOWN REPORT
# ==========================================
print("3. Generating theme_report.md...")

md_content = f"""# GDELT THEMATIC ANALYSIS REPORT

**Analysis:** Dominant narratives by Company and Time Period.

## 1. COMPANY NARRATIVES (What are they talking about?)

Each company triggers different global themes.
*(Top 5 unique themes shown)*

### Google 
| Rank | Theme | Count |
| :--- | :--- | :--- |
"""
for i, row in df_google.head(5).iterrows():
    md_content += f"| {i+1} | {row['theme']} | {row['count']} |\n"

md_content += """
### OpenAI 
| Rank | Theme | Count |
| :--- | :--- | :--- |
"""
for i, row in df_openai.head(5).iterrows():
    md_content += f"| {i+1} | {row['theme']} | {row['count']} |\n"

md_content += """
## 2. TEMPORAL SHIFT (Machine Learning Results)
We used Logistic Regression to find themes that best distinguish **February** from **May**.

### The "February" Themes (Google Era)
*Linked to Index 1 (Positive Coefficients)*
| Theme | Strength | Interpretation |
| :--- | :--- | :--- |
"""
for i, row in top_feb.iterrows():
    md_content += f"| **{row['Theme']}** | +{row['Score']:.3f} | Linked to **FEB** |\n"

md_content += """
### The "May" Themes (OpenAI Era)
*Linked to Index 0 (Negative Coefficients)*
| Theme | Strength | Interpretation |
| :--- | :--- | :--- |
"""
for i, row in top_may.iterrows():
    md_content += f"| **{row['Theme']}** | {row['Score']:.3f} | Linked to **MAY** |\n"

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(OUTPUT_DIR + "theme_report.md", "w") as f:
    f.write(md_content)
print(f"Saved {OUTPUT_DIR}theme_report.md")

# ==========================================
# PART D: GENERATE GRAPHS (PURE MATPLOTLIB)
# ==========================================
print("4. Generating Graphs...")

# --- Graph 1: Dominant Themes (Subplots) ---
fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=False)
fig.suptitle('Dominant Themes by Company', fontsize=16)

# Helper to plot on axes
def plot_company_themes(ax, df_data, title, color):
    if not df_data.empty:
        # Sort for chart
        df_sorted = df_data.sort_values('count', ascending=True)
        ax.barh(df_sorted['theme_short'], df_sorted['count'], color=color)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel("Articles")
    else:
        ax.text(0.5, 0.5, "No Data", ha='center')

plot_company_themes(axes[0], df_google, "Google", "orange")
plot_company_themes(axes[1], df_openai, "OpenAI", "green")
plot_company_themes(axes[2], df_anthropic, "Anthropic", "purple")

plt.tight_layout()
plt.subplots_adjust(top=0.85)
import os
os.makedirs(GRAPH_DIR, exist_ok=True)
plt.savefig(GRAPH_DIR + "graph_dominant_themes.png")
print("Saved graph_dominant_themes.png")

# --- Graph 2: ML Shift (Butterfly Chart) ---
plt.figure(figsize=(10, 8))

# Combine top Feb and top May
ml_plot_df = pd.concat([top_feb, top_may])
ml_plot_df['Color'] = ['orange' if x > 0 else 'green' for x in ml_plot_df['Score']]
ml_plot_df = ml_plot_df.sort_values('Score')

plt.barh(ml_plot_df['Theme'], ml_plot_df['Score'], color=ml_plot_df['Color'])
plt.title("Thematic Shift: Feb vs May", fontsize=14)
plt.xlabel("Model Coefficient (Left=May, Right=Feb)")
plt.axvline(0, color='black', linewidth=0.8)

# Add Labels
if not ml_plot_df.empty:
    min_score = ml_plot_df['Score'].min()
    max_score = ml_plot_df['Score'].max()
    plt.text(min_score/2, len(ml_plot_df)-0.5, "MAY Themes", color='green', fontweight='bold', ha='center')
    plt.text(max_score/2, 0, "FEB Themes", color='orange', fontweight='bold', ha='center')

plt.tight_layout()
plt.savefig(GRAPH_DIR + "graph_theme_ml_shift.png")
print("Saved graph_theme_ml_shift.png")

print("--- Job Complete ---")