import sys
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from pyspark.sql.functions import avg, round

# --- SETUP ---
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
spark.sparkContext.setLogLevel("ERROR")

OUTPUT_DIR = "/home/glue_user/workspace/data/gdelt/result_markdowns/"
GRAPH_DIR = "/home/glue_user/workspace/data/gdelt/graphs/"
print("--- Starting Entity Sentiment Analysis ---")

# 1. LOAD DATA
df = spark.read.parquet("file:///home/glue_user/workspace/data/gdelt/data/gdelt_ml_features.parquet")
df.createOrReplaceTempView("gdelt")

# 2. RUN "STACKED" QUERY
# Calculate the average tone ONLY for articles that mention specific companies
print("Calculating sentiment per company...")
sql_query = """
    SELECT 'Google' as Company, label_week, avg(v2tone_1) as avg_tone, count(*) as count 
    FROM gdelt WHERE k_google = 1 GROUP BY label_week
    UNION ALL
    SELECT 'OpenAI' as Company, label_week, avg(v2tone_1) as avg_tone, count(*) as count 
    FROM gdelt WHERE k_openai = 1 GROUP BY label_week
    UNION ALL
    SELECT 'Anthropic' as Company, label_week, avg(v2tone_1) as avg_tone, count(*) as count 
    FROM gdelt WHERE k_anthropic = 1 GROUP BY label_week
"""
sentiment_df = spark.sql(sql_query).toPandas()

# Sort for consistent plotting (Feb first, then May)
sentiment_df = sentiment_df.sort_values(by=['Company', 'label_week'])

print("\n--- RESULTS TABLE ---")
print(sentiment_df)

# 3. GENERATE REPORT SNIPPET
# This text can be added to Markdown report
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(OUTPUT_DIR + "entity_sentiment_snippet.md", "w") as f:
    f.write("## ENTITY SENTIMENT (Did everyone get happier?)\n\n")
    f.write("| Company | Feb Tone | May Tone | Change |\n")
    f.write("| :--- | :--- | :--- | :--- |\n")
    
    for company in ['Google', 'OpenAI', 'Anthropic']:
        rows = sentiment_df[sentiment_df['Company'] == company]
        # Get Feb/May values safely
        try:
            feb_val = rows[rows['label_week'] == 'week_feb']['avg_tone'].values[0]
            may_val = rows[rows['label_week'] == 'week_may']['avg_tone'].values[0]
            change = may_val - feb_val
            f.write(f"| **{company}** | {feb_val:.2f} | {may_val:.2f} | {change:+.2f} |\n")
        except IndexError:
             f.write(f"| **{company}** | N/A | N/A | N/A |\n")

print(f"Saved snippet to {OUTPUT_DIR}entity_sentiment_snippet.md")

# 4. PLOT GRAPH
fig = plt.figure(figsize=(10, 6), facecolor='white')
sns.set_style("whitegrid")
ax = sns.barplot(data=sentiment_df, x='Company', y='avg_tone', hue='label_week', 
                 palette={'week_feb': '#1E3A8A', 'week_may': '#ee1b27'})
ax.set_facecolor('white')

plt.title("How did Sentiment change for each Tech Giant?", fontsize=14, color='#1E3A8A', fontweight='bold')
plt.ylabel("Average Net Tone (Higher is Better)", fontsize=12, color='#1E3A8A')
plt.xlabel("", color='#1E3A8A')
ax.tick_params(colors='#1E3A8A')
plt.axhline(0, color='#1E3A8A', linewidth=0.8)

# Add Legend with better labels
handles, labels = ax.get_legend_handles_labels()
legend = ax.legend(handles, ["Feb 2024", "May 2024"], title="Time Period", facecolor='white', edgecolor='#1E3A8A')
plt.setp(legend.get_texts(), color='#1E3A8A')
plt.setp(legend.get_title(), color='#1E3A8A')
ax.spines['bottom'].set_color('#1E3A8A')
ax.spines['left'].set_color('#1E3A8A')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Save
import os
os.makedirs(GRAPH_DIR, exist_ok=True)
plt.tight_layout()
plt.savefig(GRAPH_DIR + "graph_entity_sentiment.png")
print("Saved graph_entity_sentiment.png")