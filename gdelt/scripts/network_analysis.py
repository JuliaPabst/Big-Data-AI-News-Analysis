import sys
import pandas as pd
# Use Agg backend for headless plotting
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from pyspark.context import SparkContext
from awsglue.context import GlueContext
# FIX: Import 'count' and 'round' explicitly
from pyspark.sql.functions import col, explode, desc, lit, avg, count, round

# --- 1. SETUP ---
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
spark.sparkContext.setLogLevel("ERROR")

DATA_DIR = "file:///home/glue_user/workspace/data/gdelt/data/"
OUTPUT_DIR = "/home/glue_user/workspace/data/gdelt/result_markdowns/"
GRAPH_DIR = "/home/glue_user/workspace/data/gdelt/graphs/"

print("--- Starting Source & Network Analysis ---")

# Load FIXED Data
try:
    df = spark.read.parquet(DATA_DIR + "gdelt_core_fixed.parquet")
except:
    print("Fixed file not found, trying original...")
    df = spark.read.parquet(DATA_DIR + "gdelt_core.parquet")


# PART A: WHO IS WRITING? (Source Analysis)
print("1. Analyzing Top News Sources...")

def get_top_sources(company_col, company_name, limit=10):
    return df.filter(col(company_col) == 1) \
             .groupBy("domain") \
             .agg(
                 col("domain").alias("source"),
                 count("*").alias("article_count"),       # FIX: Use count() directly
                 round(avg("v2tone_1"), 2).alias("avg_sentiment") # FIX: Use round() directly
             ) \
             .orderBy(desc("article_count")) \
             .limit(limit) \
             .withColumn("Company", lit(company_name)) \
             .toPandas()

# Get top 10 sources for each
src_google = get_top_sources("k_google", "Google", 10)
src_openai = get_top_sources("k_openai", "OpenAI", 10)


# PART B: WHO ARE THE ALLIES? (Network Analysis)
print("2. Analyzing Co-occurring Organizations...")

def get_top_partners(company_col, company_name, exclude_list, limit=10):
    # Explode the orgs_arr to get one row per organization
    exploded = df.filter(col(company_col) == 1) \
                 .select(explode(col("orgs_arr")).alias("partner"))
    
    # Filter out the company itself and common noise words
    return exploded.filter(~col("partner").isin(exclude_list)) \
                   .groupBy("partner") \
                   .count() \
                   .orderBy(desc("count")) \
                   .limit(limit) \
                   .withColumn("Main_Entity", lit(company_name)) \
                   .toPandas()

# Define exclusion lists
ignore_common = ['reuters', 'associated press', 'bloomberg']
ignore_google = ignore_common + ['google', 'alphabet', 'google inc', 'alphabet inc']
ignore_openai = ignore_common + ['openai', 'chatgpt', 'open ai']

# Get partners
part_google = get_top_partners("k_google", "Google", ignore_google, 10)
part_openai = get_top_partners("k_openai", "OpenAI", ignore_openai, 10)

# PART C: GENERATE MARKDOWN REPORT
print("3. Generating network_report.md...")

md_content = f"""# GDELT NETWORK ANALYSIS

**Analysis:** News Sources and Corporate Partnerships.

## 1. TOP NEWS SOURCES (Who controls the narrative?)
Which publishers are most obsessed with each company?

### Writing about Google:
| Rank | Source | Articles | Avg Tone |
| :--- | :--- | :--- | :--- |
"""
for i, row in src_google.iterrows():
    md_content += f"| {i+1} | {row['source']} | {row['article_count']} | {row['avg_sentiment']} |\n"

md_content += """
### Writing about OpenAI:
| Rank | Source | Articles | Avg Tone |
| :--- | :--- | :--- | :--- |
"""
for i, row in src_openai.iterrows():
    md_content += f"| {i+1} | {row['source']} | {row['article_count']} | {row['avg_sentiment']} |\n"

md_content += """
## 2. CORPORATE ECOSYSTEM (Co-occurrence)
When these companies appear in the news, who is standing next to them?

### Google's Ecosystem:
| Rank | Partner | Co-mentions |
| :--- | :--- | :--- |
"""
for i, row in part_google.iterrows():
    md_content += f"| {i+1} | {row['partner']} | {row['count']} |\n"

md_content += """
### OpenAI's Ecosystem:
| Rank | Partner | Co-mentions |
| :--- | :--- | :--- |
"""
for i, row in part_openai.iterrows():
    md_content += f"| {i+1} | {row['partner']} | {row['count']} |\n"

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(OUTPUT_DIR + "network_report.md", "w") as f:
    f.write(md_content)
print(f"Saved {OUTPUT_DIR}network_report.md")

# PART D: GENERATE GRAPHS
print("4. Generating Graphs...")

# --- Graph 1: Top Sources Comparison ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor='white')

def plot_bar(ax, df, title, color):
    ax.set_facecolor('white')
    if not df.empty:
        df = df.sort_values('article_count', ascending=True) 
        ax.barh(df['source'], df['article_count'], color=color)
        ax.set_title(title, fontsize=12, fontweight='bold', color='#1E3A8A')
        ax.set_xlabel("Number of Articles", color='#1E3A8A')
        ax.tick_params(colors='#1E3A8A')
        ax.spines['bottom'].set_color('#1E3A8A')
        ax.spines['left'].set_color('#1E3A8A')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    else:
        ax.text(0.5, 0.5, "No Data", ha='center', color='#1E3A8A')

plot_bar(axes[0], src_google, "Who writes about Google?", "#003E96")
plot_bar(axes[1], src_openai, "Who writes about OpenAI?", "#ee1b27")

plt.tight_layout()
import os
os.makedirs(GRAPH_DIR, exist_ok=True)
plt.savefig(GRAPH_DIR + "graph_top_sources.png")
print("Saved graph_top_sources.png")

# --- Graph 2: Partnerships Comparison ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor='white')

def plot_partners(ax, df, title, color):
    ax.set_facecolor('white')
    if not df.empty:
        df = df.sort_values('count', ascending=True)
        # Truncate long partner names
        short_names = [n[:15] + ".." if len(n)>15 else n for n in df['partner']]
        ax.barh(short_names, df['count'], color=color)
        ax.set_title(title, fontsize=12, fontweight='bold', color='#1E3A8A')
        ax.set_xlabel("Co-mentions", color='#1E3A8A')
        ax.tick_params(colors='#1E3A8A')
        ax.spines['bottom'].set_color('#1E3A8A')
        ax.spines['left'].set_color('#1E3A8A')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    else:
        ax.text(0.5, 0.5, "No Data", ha='center', color='#1E3A8A')

plot_partners(axes[0], part_google, "Google's Orbit", "#003E96")
plot_partners(axes[1], part_openai, "OpenAI's Orbit", "#ee1b27")

plt.tight_layout()
plt.savefig(GRAPH_DIR + "graph_partnerships.png")
print("Saved graph_partnerships.png")

print("--- Job Complete ---")