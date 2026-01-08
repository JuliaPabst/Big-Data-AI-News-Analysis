import pyarrow.parquet as pq
import pandas as pd
from collections import Counter

CORE_FILE = "gdelt_core.parquet"
ML_FILE = "gdelt_ml_features.parquet"

core = pq.read_table(CORE_FILE).to_pandas()
ml = pq.read_table(ML_FILE).to_pandas()

# --- QA Summary ---
label_counts = core["label_week"].value_counts().to_dict()
top_domains = core["domain"].value_counts().head(10).to_dict()

pct_missing_orgs = (core["orgs_arr"].apply(lambda x: x is None or len(x)==0)).mean() * 100
pct_missing_themes = (core["themes_arr"].apply(lambda x: x is None or len(x)==0)).mean() * 100

date_min = pd.to_datetime(core["date_ts"]).min()
date_max = pd.to_datetime(core["date_ts"]).max()

qa_lines = []
qa_lines.append("QA SUMMARY: Mapping AI Project (GDELT Export)\n")
qa_lines.append(f"Rows (core): {len(core)}")
qa_lines.append(f"Rows (ml):   {len(ml)}")
qa_lines.append(f"Unique URLs (core): {core['url'].nunique()}")
qa_lines.append(f"Date range: {date_min}  ->  {date_max}\n")

qa_lines.append("Label counts (label_week):")
for k,v in label_counts.items():
    qa_lines.append(f"  - {k}: {v}")
qa_lines.append("")

qa_lines.append("Top 10 domains:")
for k,v in top_domains.items():
    qa_lines.append(f"  - {k}: {v}")
qa_lines.append("")

qa_lines.append(f"Missing orgs_arr (empty/null): {pct_missing_orgs:.2f}%")
qa_lines.append(f"Missing themes_arr (empty/null): {pct_missing_themes:.2f}%")
qa_lines.append("")

# v2tone missing
qa_lines.append("Missing rate per v2tone column:")
for i in range(1,8):
    col = f"v2tone_{i}"
    miss = core[col].isna().mean() * 100
    qa_lines.append(f"  - {col}: {miss:.2f}%")

with open("QA_SUMMARY.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(qa_lines))

# --- Data Dictionary ---
dd = []
dd.append("# DATA DICTIONARY\n")
dd.append("Files:")
dd.append("- gdelt_core.parquet: one row per URL (deduplicated), includes raw fields + some basic derived features")
dd.append("- gdelt_ml_features.parquet: compact ML feature table (numerical + flags) for modeling\n")

dd.append("## gdelt_core.parquet columns\n")
dd.append("| Column | Type | Description | Example |")
dd.append("|---|---|---|---|")

def ex(col):
    val = core[col].iloc[0]
    s = str(val)
    return (s[:60] + "…") if len(s) > 60 else s

core_types = core.dtypes.astype(str).to_dict()
core_desc = {
    "GKGRECORDID": "GDELT record id (from BigQuery export).",
    "url": "Article URL (DocumentIdentifier). Unique key after dedup.",
    "domain": "Extracted registrable domain from URL (e.g. theverge.com).",
    "date_ts": "Timestamp parsed from DATE (YYYYMMDDhhmmss).",
    "day": "Date part (YYYY-MM-DD).",
    "label_week": "Week label: week_feb (2024-02-12..19) or week_may (2024-05-13..20).",
    "themes_arr": "Themes split into an array (from Themes, split by ';').",
    "orgs_arr": "Organizations split into an array (from Organizations, split by ';').",
    "v2tone_1": "V2Tone field #1 parsed from V2Tone (comma-separated).",
    "v2tone_2": "V2Tone field #2 parsed from V2Tone (comma-separated).",
    "v2tone_3": "V2Tone field #3 parsed from V2Tone (comma-separated).",
    "v2tone_4": "V2Tone field #4 parsed from V2Tone (comma-separated).",
    "v2tone_5": "V2Tone field #5 parsed from V2Tone (comma-separated).",
    "v2tone_6": "V2Tone field #6 parsed from V2Tone (comma-separated).",
    "v2tone_7": "V2Tone field #7 parsed from V2Tone (comma-separated).",
    "url_tokens": "Tokenized URL (split on non-alphanumeric). Useful for TF-IDF on URL tokens.",
    "url_length": "Length of URL string.",
    "num_themes": "Count of themes in themes_arr.",
    "num_orgs": "Count of orgs in orgs_arr.",
    "k_openai": "Flag (0/1): URL tokens contain OpenAI/GPT/ChatGPT/Sora terms.",
    "k_google": "Flag (0/1): URL tokens contain Google/Gemini/Alphabet terms.",
    "k_anthropic": "Flag (0/1): URL tokens contain Anthropic/Claude terms.",
}

for col in core.columns:
    dd.append(f"| {col} | {core_types.get(col,'')} | {core_desc.get(col,'')} | {ex(col)} |")

dd.append("\n## gdelt_ml_features.parquet columns\n")
dd.append("| Column | Type | Description | Example |")
dd.append("|---|---|---|---|")

ml_types = ml.dtypes.astype(str).to_dict()
ml_desc = {
    "url": "Join key back to core.",
    "domain": "Registrable domain.",
    "day": "Date part.",
    "label_week": "Target label (week_feb vs week_may).",
    "url_length": "Length of URL string.",
    "num_themes": "Count of themes.",
    "num_orgs": "Count of organizations.",
    "k_openai": "OpenAI keyword flag (0/1).",
    "k_google": "Google keyword flag (0/1).",
    "k_anthropic": "Anthropic keyword flag (0/1).",
    "v2tone_1": "V2Tone field #1 parsed.",
    "v2tone_2": "V2Tone field #2 parsed.",
    "v2tone_3": "V2Tone field #3 parsed.",
    "v2tone_4": "V2Tone field #4 parsed.",
    "v2tone_5": "V2Tone field #5 parsed.",
    "v2tone_6": "V2Tone field #6 parsed.",
    "v2tone_7": "V2Tone field #7 parsed.",
}

def ex_ml(col):
    val = ml[col].iloc[0]
    s = str(val)
    return (s[:60] + "…") if len(s) > 60 else s

for col in ml.columns:
    dd.append(f"| {col} | {ml_types.get(col,'')} | {ml_desc.get(col,'')} | {ex_ml(col)} |")

with open("DATA_DICTIONARY.md", "w", encoding="utf-8") as f:
    f.write("\n".join(dd))

print("Created QA_SUMMARY.txt and DATA_DICTIONARY.md")
