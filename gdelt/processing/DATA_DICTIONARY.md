# DATA DICTIONARY

Files:
- gdelt_core.parquet: one row per URL (deduplicated), includes raw fields + some basic derived features
- gdelt_ml_features.parquet: compact ML feature table (numerical + flags) for modeling

## gdelt_core.parquet columns

| Column | Type | Description | Example |
|---|---|---|---|
| GKGRECORDID | object | GDELT record id (from BigQuery export). | 20240216233000-19 |
| url | object | Article URL (DocumentIdentifier). Unique key after dedup. | https://www.wthitv.com/news/leading-tech-firms-pledge-to-add… |
| domain | object | Extracted registrable domain from URL (e.g. theverge.com). | wthitv.com |
| date_ts | datetime64[ns] | Timestamp parsed from DATE (YYYYMMDDhhmmss). | 2024-02-16 23:30:00 |
| day | object | Date part (YYYY-MM-DD). | 2024-02-16 |
| label_week | object | Week label: week_feb (2024-02-12..19) or week_may (2024-05-13..20). | week_feb |
| themes_arr | object | Themes split into an array (from Themes, split by ';'). | ['ELECTION' 'TAX_FNCACT' 'TAX_FNCACT_LEADERS' 'LEADER'
 'TAX… |
| orgs_arr | object | Organizations split into an array (from Organizations, split by ';'). | ['cable news network inc' 'warner bros' 'microsoft' 'google'… |
| v2tone_1 | float64 | V2Tone field #1 parsed from V2Tone (comma-separated). | -1.99637023593466 |
| v2tone_2 | float64 | V2Tone field #2 parsed from V2Tone (comma-separated). | 3.62976406533575 |
| v2tone_3 | float64 | V2Tone field #3 parsed from V2Tone (comma-separated). | 5.62613430127042 |
| v2tone_4 | float64 | V2Tone field #4 parsed from V2Tone (comma-separated). | 9.25589836660617 |
| v2tone_5 | float64 | V2Tone field #5 parsed from V2Tone (comma-separated). | 21.5970961887477 |
| v2tone_6 | float64 | V2Tone field #6 parsed from V2Tone (comma-separated). | 0.725952813067151 |
| v2tone_7 | float64 | V2Tone field #7 parsed from V2Tone (comma-separated). | 490.0 |
| url_tokens | object | Tokenized URL (split on non-alphanumeric). Useful for TF-IDF on URL tokens. | ['https' 'www' 'wthitv' 'com' 'news' 'leading' 'tech' 'firms… |
| url_length | int64 | Length of URL string. | 141 |
| num_themes | int64 | Count of themes in themes_arr. | 36 |
| num_orgs | int64 | Count of orgs in orgs_arr. | 5 |
| k_openai | int64 | Flag (0/1): URL tokens contain OpenAI/GPT/ChatGPT/Sora terms. | 0 |
| k_google | int64 | Flag (0/1): URL tokens contain Google/Gemini/Alphabet terms. | 1 |
| k_anthropic | int64 | Flag (0/1): URL tokens contain Anthropic/Claude terms. | 0 |

## gdelt_ml_features.parquet columns

| Column | Type | Description | Example |
|---|---|---|---|
| url | object | Join key back to core. | https://www.wthitv.com/news/leading-tech-firms-pledge-to-add… |
| domain | object | Registrable domain. | wthitv.com |
| day | object | Date part. | 2024-02-16 |
| label_week | object | Target label (week_feb vs week_may). | week_feb |
| url_length | int64 | Length of URL string. | 141 |
| num_themes | int64 | Count of themes. | 36 |
| num_orgs | int64 | Count of organizations. | 5 |
| k_openai | int64 | OpenAI keyword flag (0/1). | 0 |
| k_google | int64 | Google keyword flag (0/1). | 1 |
| k_anthropic | int64 | Anthropic keyword flag (0/1). | 0 |
| v2tone_1 | float64 | V2Tone field #1 parsed. | -1.99637023593466 |
| v2tone_2 | float64 | V2Tone field #2 parsed. | 3.62976406533575 |
| v2tone_3 | float64 | V2Tone field #3 parsed. | 5.62613430127042 |
| v2tone_4 | float64 | V2Tone field #4 parsed. | 9.25589836660617 |
| v2tone_5 | float64 | V2Tone field #5 parsed. | 21.5970961887477 |
| v2tone_6 | float64 | V2Tone field #6 parsed. | 0.725952813067151 |
| v2tone_7 | float64 | V2Tone field #7 parsed. | 490.0 |