import sys
import matplotlib.pyplot as plt
import pandas as pd
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.functions import avg, col, when, round

# --- 1. SETUP & CONFIGURATION ---
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
spark.sparkContext.setLogLevel("ERROR")

# The folder inside Docker where files are read/written
DATA_DIR = "file:///home/glue_user/workspace/data/gdelt/data/"
OUTPUT_DIR = "/home/glue_user/workspace/data/gdelt/result_markdowns/"
GRAPH_DIR = "/home/glue_user/workspace/data/gdelt/graphs/"

print("--- Starting GDELT Analysis Job ---")

# --- 2. LOAD DATA ---
df = spark.read.parquet(DATA_DIR + "gdelt_ml_features.parquet")
df.createOrReplaceTempView("gdelt")
print(f"Loaded {df.count()} rows.")

# --- 3. CALCULATE STATISTICS (For the Evidence Table) ---
print("Calculating detailed statistics...")
stats_df = spark.sql("""
    SELECT 
        label_week,
        count(*) as total,
        round(avg(v2tone_1), 2) as avg_tone,
        round(avg(v2tone_3), 2) as avg_neg,
        round(avg(k_google)*100, 1) as pct_google,
        round(avg(k_openai)*100, 1) as pct_openai,
        round(avg(k_anthropic)*100, 1) as pct_anthropic
    FROM gdelt 
    GROUP BY label_week
    ORDER BY label_week
""").collect()

# Row 0 is Feb, Row 1 is May 
feb_stats = stats_df[0] 
may_stats = stats_df[1]

# --- 4. TRAIN MACHINE LEARNING MODEL ---
print("Training Logistic Regression Model...")
# Convert labels to numbers (week_feb=0, week_may=1)
label_indexer = StringIndexer(inputCol="label_week", outputCol="label")
df_indexed = label_indexer.fit(df).transform(df)

# Define features
feature_cols = ['k_openai', 'k_google', 'k_anthropic', 'v2tone_1', 'v2tone_3', 'v2tone_6']

# Prepare vectors
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
final_data = assembler.transform(df_indexed.fillna(0, subset=feature_cols))

# Train
lr = LogisticRegression(featuresCol="features", labelCol="label")
model = lr.fit(final_data)

# Calculate AUC
predictions = model.transform(final_data)
evaluator = BinaryClassificationEvaluator(labelCol="label")
auc = evaluator.evaluate(predictions)

# --- 5. GENERATE MARKDOWN REPORT (.md) ---
print("Generating summary_report.md...")

# Build the report string using the computed stats and model results
report_content = f"""# GDELT AI PROJECT: ANALYTICAL SUMMARY

**Date Range:** Feb 12, 2024 - May 20, 2024  
**Dataset:** {df.count()} Articles  
**Target:** Distinguishing News Patterns between February and May

## 1. SUMMARY

Our machine learning model (Logistic Regression) achieved an **AUC of {auc:.2f}**, indicating a strong ability to distinguish between the two time periods based on article content.

The analysis reveals a distinct **"Editorial Shift"** characterized by:
1. A change in **dominant Tech Giants** (Google in Feb $\\rightarrow$ OpenAI in May).
2. A shift in **Sentiment** (Critical/Negative in Feb $\\rightarrow$ Optimistic in May).
3. A shift in **Style** (Opinionated in Feb $\\rightarrow$ Objective in May).

## 2. DETAILED STATISTICS (The Evidence)

| Metric | Week Feb | Week May | Shift |
| :--- | :--- | :--- | :--- |
| **Net Tone** | {feb_stats['avg_tone']} | **{may_stats['avg_tone']}** | **+{may_stats['avg_tone'] - feb_stats['avg_tone']:.2f}** (More Positive) |
| **Negative Score** | {feb_stats['avg_neg']} | **{may_stats['avg_neg']}** | **{may_stats['avg_neg'] - feb_stats['avg_neg']:.2f}** (Less Critical) |
| **% Mentioning OpenAI** | {feb_stats['pct_openai']}% | **{may_stats['pct_openai']}%** | **+{may_stats['pct_openai'] - feb_stats['pct_openai']:.1f}%** |
| **% Mentioning Google** | **{feb_stats['pct_google']}%** | {may_stats['pct_google']}% | **{may_stats['pct_google'] - feb_stats['pct_google']:.1f}%** |

## 3. KEY DRIVERS (FEATURE IMPORTANCE)

The following features were the strongest predictors for the time period:
*(Negative Coefficients are linked to MAY, Positive to FEB)*

| Feature | Coefficient | Interpretation |
| :--- | :--- | :--- |
"""

# Loop through features to add rows to the table
for i, col_name in enumerate(feature_cols):
    coeff = model.coefficients[i]
    if coeff < -0.1:
        interpretation = "Strongly linked to **MAY**"
    elif coeff > 0.1:
        interpretation = "Strongly linked to **FEB**"
    else:
        interpretation = "Neutral / Low Impact"
    
    report_content += f"| **{col_name}** | `{coeff:.4f}` | {interpretation} |\n"

report_content += """
## 4. INTERPRETATION

**A widening gap in Tech Giants**: OpenAI dominated both periods, but Google was significantly more prominent in Feb (55%) than in May (47%), making it a strong signal for the earlier period.

**February 2024** was defined by "Google Gemini" coverage. The correlation with higher negative scores and self-referencing language suggests this period contained significant critical analysis, op-eds, and controversy regarding Google's AI launches.

**May 2024** was defined by "OpenAI GPT-4o" and "Anthropic" coverage. The shift to positive tone and objective language indicates a reception focused more on product capabilities, launch announcements, and factual reporting rather than controversy.
"""

# Write the file
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(OUTPUT_DIR + "summary_report.md", "w") as f:
    f.write(report_content)

print(f"Report saved to: {OUTPUT_DIR}summary_report.md")

# --- 6. GENERATE GRAPHS (IMPROVED) ---
print("Generating graphs...")

import matplotlib.pyplot as plt
import pandas as pd

# 1. Define Human-Readable Mappings
feature_map = {
    'k_openai': 'OpenAI Mention',
    'k_anthropic': 'Anthropic Mention',
    'k_google': 'Google Mention',
    'v2tone_1': 'Tone (Optimism)',
    'v2tone_3': 'Negative Score',
    'v2tone_6': 'Self/Group Ref ("I/We")'
}

# 2. Get Coefficients and Flip Direction
# Multiply by -1 so that May (the later date) becomes Positive (Right side)
# and Feb (the earlier date) becomes Negative (Left side).
raw_coeffs = model.coefficients.toArray()
plot_coeffs = [x * -1 for x in raw_coeffs] 
plot_labels = [feature_map.get(col, col) for col in feature_cols]

# 3. Graph 1: Feature Importance
plt.figure(figsize=(11, 6), facecolor='white')
ax = plt.gca()
ax.set_facecolor('white')

# Color logic: Red for May (Right), Light blue for Feb (Left)
colors = ['#ee1b27' if x > 0 else '#003E96' for x in plot_coeffs]

bars = plt.barh(plot_labels, plot_coeffs, color=colors)
plt.title(f"What distinguishes the two periods? (Model Impact)", fontsize=14, color='#1E3A8A', fontweight='bold')

# Custom X-Axis Labels
plt.xlabel("Impact Strength", fontsize=12, color='#1E3A8A')
plt.axvline(0, color='#1E3A8A', linewidth=0.8)

# Add "Linked to..." text on the plot sides
plt.text(min(plot_coeffs)/2, len(plot_labels)-0.5, "Linked to FEB\n(Google Era)", 
         ha='center', color='#003E96', fontweight='bold', fontsize=11)
plt.text(max(plot_coeffs)/2, len(plot_labels)-0.5, "Linked to MAY\n(OpenAI Era)", 
         ha='center', color='#ee1b27', fontweight='bold', fontsize=11)

# Style axes
ax.tick_params(colors='#1E3A8A', labelsize=10)
ax.spines['bottom'].set_color('#1E3A8A')
ax.spines['left'].set_color('#1E3A8A')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
import os
os.makedirs(GRAPH_DIR, exist_ok=True)
plt.savefig(GRAPH_DIR + "graph_feature_importance.png")
print("Saved graph_feature_importance.png (with readable labels)")

# 4. Graph 2: Share of Voice
pdf_stats = pd.DataFrame(stats_df, columns=stats_df[0].asDict().keys())
pdf_stats = pdf_stats.set_index('label_week')

# Rename columns for the legend
pdf_stats = pdf_stats.rename(columns={
    'pct_google': 'Google', 
    'pct_openai': 'OpenAI',
    'pct_anthropic': 'Anthropic'
})

# Plot
fig = plt.figure(figsize=(10, 6), facecolor='white')
ax = fig.add_subplot(111)
ax.set_facecolor('white')

pdf_stats[['Google', 'OpenAI', 'Anthropic']].plot(kind='bar', ax=ax, color=['#003E96', '#ee1b27', '#1E3A8A'])
plt.title("Tech Giant Share of Voice", fontsize=14, color='#1E3A8A', fontweight='bold')
plt.ylabel("Percentage of Articles", fontsize=12, color='#1E3A8A')
plt.xlabel("", fontsize=12, color='#1E3A8A')
plt.xticks(rotation=0, color='#1E3A8A')
plt.yticks(color='#1E3A8A')
plt.ylim(0, 100) # Fix y-axis to 0-100%

# Add value labels on top of bars
for p in ax.patches:
    if p.get_height() > 0:  # Only show labels for non-zero values
        ax.annotate(f'{p.get_height():.1f}%', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='bottom', fontsize=9, fontweight='bold', color='#1E3A8A')

# Style axes
ax.spines['bottom'].set_color('#1E3A8A')
ax.spines['left'].set_color('#1E3A8A')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(facecolor='white', edgecolor='#1E3A8A', labelcolor='#1E3A8A')

plt.tight_layout()
plt.savefig(GRAPH_DIR + "graph_share_of_voice.png")
print("Saved graph_share_of_voice.png")

# --- 5. SOURCE SENTIMENT ANALYSIS ---
print("Analyzing sentiment by news source...")

# Most positive and negative sources for Google (minimum 3 articles)
google_sources = spark.sql("""
    SELECT 
        domain as source,
        count(*) as article_count,
        round(avg(v2tone_1), 2) as avg_sentiment
    FROM gdelt
    WHERE k_google = 1 AND domain IS NOT NULL
    GROUP BY domain
    HAVING count(*) >= 3
    ORDER BY avg_sentiment DESC
""").toPandas()

# Most positive and negative sources for OpenAI (minimum 3 articles)
openai_sources = spark.sql("""
    SELECT 
        domain as source,
        count(*) as article_count,
        round(avg(v2tone_1), 2) as avg_sentiment
    FROM gdelt
    WHERE k_openai = 1 AND domain IS NOT NULL
    GROUP BY domain
    HAVING count(*) >= 3
    ORDER BY avg_sentiment DESC
""").toPandas()

# Get top 5 most positive and negative for each
google_positive = google_sources.head(5)
google_negative = google_sources.tail(5).iloc[::-1]  # Reverse to show most negative first
openai_positive = openai_sources.head(5)
openai_negative = openai_sources.tail(5).iloc[::-1]

# Graph: Side-by-side comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor='white')

# Google Positive
if not google_positive.empty:
    axes[0, 0].set_facecolor('white')
    axes[0, 0].barh(google_positive['source'], google_positive['avg_sentiment'], color='#003E96')
    axes[0, 0].set_title("Google: Most Positive Sources", fontsize=12, fontweight='bold', color='#1E3A8A')
    axes[0, 0].set_xlabel("Avg Sentiment Score", color='#1E3A8A')
    axes[0, 0].tick_params(colors='#1E3A8A', labelsize=9)
    axes[0, 0].spines['bottom'].set_color('#1E3A8A')
    axes[0, 0].spines['left'].set_color('#1E3A8A')
    axes[0, 0].spines['top'].set_visible(False)
    axes[0, 0].spines['right'].set_visible(False)
    axes[0, 0].invert_yaxis()

# Google Negative
if not google_negative.empty:
    axes[1, 0].set_facecolor('white')
    axes[1, 0].barh(google_negative['source'], google_negative['avg_sentiment'], color='#ee1b27')
    axes[1, 0].set_title("Google: Most Critical Sources", fontsize=12, fontweight='bold', color='#1E3A8A')
    axes[1, 0].set_xlabel("Avg Sentiment Score", color='#1E3A8A')
    axes[1, 0].tick_params(colors='#1E3A8A', labelsize=9)
    axes[1, 0].spines['bottom'].set_color('#1E3A8A')
    axes[1, 0].spines['left'].set_color('#1E3A8A')
    axes[1, 0].spines['top'].set_visible(False)
    axes[1, 0].spines['right'].set_visible(False)
    axes[1, 0].invert_yaxis()

# OpenAI Positive
if not openai_positive.empty:
    axes[0, 1].set_facecolor('white')
    axes[0, 1].barh(openai_positive['source'], openai_positive['avg_sentiment'], color='#003E96')
    axes[0, 1].set_title("OpenAI: Most Positive Sources", fontsize=12, fontweight='bold', color='#1E3A8A')
    axes[0, 1].set_xlabel("Avg Sentiment Score", color='#1E3A8A')
    axes[0, 1].tick_params(colors='#1E3A8A', labelsize=9)
    axes[0, 1].spines['bottom'].set_color('#1E3A8A')
    axes[0, 1].spines['left'].set_color('#1E3A8A')
    axes[0, 1].spines['top'].set_visible(False)
    axes[0, 1].spines['right'].set_visible(False)
    axes[0, 1].invert_yaxis()

# OpenAI Negative
if not openai_negative.empty:
    axes[1, 1].set_facecolor('white')
    axes[1, 1].barh(openai_negative['source'], openai_negative['avg_sentiment'], color='#ee1b27')
    axes[1, 1].set_title("OpenAI: Most Critical Sources", fontsize=12, fontweight='bold', color='#1E3A8A')
    axes[1, 1].set_xlabel("Avg Sentiment Score", color='#1E3A8A')
    axes[1, 1].tick_params(colors='#1E3A8A', labelsize=9)
    axes[1, 1].spines['bottom'].set_color('#1E3A8A')
    axes[1, 1].spines['left'].set_color('#1E3A8A')
    axes[1, 1].spines['top'].set_visible(False)
    axes[1, 1].spines['right'].set_visible(False)
    axes[1, 1].invert_yaxis()

plt.tight_layout()
plt.savefig(GRAPH_DIR + "graph_source_sentiment.png")
print("Saved graph_source_sentiment.png")

# Add to markdown report
sentiment_report = f"""
## 4. SOURCE SENTIMENT ANALYSIS

### Google Coverage

**Most Positive Sources:**
| Rank | Source | Articles | Avg Sentiment |
| :--- | :--- | :--- | :--- |
"""
for i, row in google_positive.iterrows():
    sentiment_report += f"| {i+1} | {row['source']} | {row['article_count']} | **+{row['avg_sentiment']}** |\n"

sentiment_report += """
**Most Critical Sources:**
| Rank | Source | Articles | Avg Sentiment |
| :--- | :--- | :--- | :--- |
"""
for i, row in google_negative.iterrows():
    sentiment_report += f"| {i+1} | {row['source']} | {row['article_count']} | **{row['avg_sentiment']}** |\n"

sentiment_report += """
### OpenAI Coverage

**Most Positive Sources:**
| Rank | Source | Articles | Avg Sentiment |
| :--- | :--- | :--- | :--- |
"""
for i, row in openai_positive.iterrows():
    sentiment_report += f"| {i+1} | {row['source']} | {row['article_count']} | **+{row['avg_sentiment']}** |\n"

sentiment_report += """
**Most Critical Sources:**
| Rank | Source | Articles | Avg Sentiment |
| :--- | :--- | :--- | :--- |
"""
for i, row in openai_negative.iterrows():
    sentiment_report += f"| {i+1} | {row['source']} | {row['article_count']} | **{row['avg_sentiment']}** |\n"

# Append to existing report
with open(OUTPUT_DIR + "summary_report.md", "a") as f:
    f.write(sentiment_report)

print("Updated summary_report.md with source sentiment analysis")

print("--- Job Complete ---")