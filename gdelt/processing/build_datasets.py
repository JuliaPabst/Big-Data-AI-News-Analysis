import json
import re
from datetime import datetime
import pandas as pd
import tldextract

INFILE = "bquxjob_4cd3ca9c_19b553d5fe2.json"

def parse_date(s):
    # format: YYYYMMDDhhmmss
    return datetime.strptime(s, "%Y%m%d%H%M%S")

def get_domain(url):
    ext = tldextract.extract(url)
    if ext.suffix:
        return f"{ext.domain}.{ext.suffix}"
    return ext.domain

def safe_split_semicolon(s):
    if s is None or str(s).strip() == "":
        return []
    return [x.strip() for x in str(s).split(";") if x.strip()]

def parse_v2tone(s):
    # v2tone is comma-separated numbers
    if s is None or str(s).strip() == "":
        return [None]*7
    parts = [p.strip() for p in str(s).split(",")]
    # pad / cut to 7
    parts = (parts + [None]*7)[:7]
    out = []
    for p in parts:
        try:
            out.append(float(p))
        except:
            out.append(None)
    return out

def label_week(day):
    # day is date object
    if datetime(2024,2,12).date() <= day <= datetime(2024,2,19).date():
        return "week_feb"
    if datetime(2024,5,13).date() <= day <= datetime(2024,5,20).date():
        return "week_may"
    return "other"

def url_tokens(url):
    # simple tokenization from URL path + slug
    toks = re.split(r"[^a-zA-Z0-9]+", url.lower())
    toks = [t for t in toks if t and not t.isdigit()]
    return toks

def contains_any(tokens, keywords):
    s = " ".join(tokens)
    return int(any(k in s for k in keywords))

# Load JSON (could be array or JSONL)
with open(INFILE, "r", encoding="utf-8") as f:
    txt = f.read().strip()

records = []
if txt.startswith("["):
    records = json.loads(txt)
else:
    # JSONL fallback
    for line in txt.splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))

df = pd.DataFrame(records)

# Normalize column names (based on export)
# expected columns: GKGRECORDID, DATE, URL, Themes, Organizations, V2Tone
df["date_ts"] = df["DATE"].apply(parse_date)
df["day"] = df["date_ts"].dt.date
df["label_week"] = df["day"].apply(label_week)

df["url"] = df["URL"]
df["domain"] = df["url"].apply(get_domain)

df["themes_arr"] = df["Themes"].apply(safe_split_semicolon)
df["orgs_arr"] = df["Organizations"].apply(safe_split_semicolon)

v2 = df["V2Tone"].apply(parse_v2tone)
v2cols = ["v2tone_1","v2tone_2","v2tone_3","v2tone_4","v2tone_5","v2tone_6","v2tone_7"]
df[v2cols] = pd.DataFrame(v2.tolist(), index=df.index)

# Dedup by URL (keep first)
df = df.dropna(subset=["url"]).drop_duplicates(subset=["url"], keep="first")

# Core dataset
core_cols = ["GKGRECORDID","url","domain","date_ts","day","label_week","themes_arr","orgs_arr"] + v2cols
core = df[core_cols].copy()

# ML features (URL-token based + basic counts)
core["url_tokens"] = core["url"].apply(url_tokens)
core["url_length"] = core["url"].str.len()
core["num_themes"] = core["themes_arr"].apply(len)
core["num_orgs"] = core["orgs_arr"].apply(len)

keywords_openai = ["openai","gpt","gpt4","gpt-4","gpt4o","chatgpt","sora"]
keywords_google = ["google","alphabet","gemini","io","i-o"]
keywords_anthropic = ["anthropic","claude","claude3","claude-3"]

core["k_openai"] = core["url_tokens"].apply(lambda t: contains_any(t, keywords_openai))
core["k_google"] = core["url_tokens"].apply(lambda t: contains_any(t, keywords_google))
core["k_anthropic"] = core["url_tokens"].apply(lambda t: contains_any(t, keywords_anthropic))

ml_cols = ["url","domain","day","label_week","url_length","num_themes","num_orgs","k_openai","k_google","k_anthropic"] + v2cols
ml = core[ml_cols].copy()

# Write outputs
core.to_parquet("gdelt_core.parquet", index=False)
ml.to_parquet("gdelt_ml_features.parquet", index=False)

core.head(200).to_csv("gdelt_core_sample_200.csv", index=False)

print("Wrote: gdelt_core.parquet, gdelt_ml_features.parquet, gdelt_core_sample_200.csv")
print("Rows core:", len(core))
