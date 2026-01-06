# GDELT AI PROJECT: ANALYTICAL SUMMARY

**Date Range:** Feb 12, 2024 - May 20, 2024  
**Dataset:** 3415 Articles  
**Target:** Distinguishing News Patterns between February and May

## 1. SUMMARY

Our machine learning model (Logistic Regression) achieved an **AUC of 0.72**, indicating a strong ability to distinguish between the two time periods based on article content.

The analysis reveals a distinct **"Editorial Shift"** characterized by:
1. A change in **dominant Tech Giants** (Google in Feb $\rightarrow$ OpenAI in May).
2. A shift in **Sentiment** (Critical/Negative in Feb $\rightarrow$ Optimistic in May).
3. A shift in **Style** (Opinionated in Feb $\rightarrow$ Objective in May).

## 2. DETAILED STATISTICS (The Evidence)

| Metric | Week Feb | Week May | Shift |
| :--- | :--- | :--- | :--- |
| **Net Tone** | 0.59 | **2.06** | **+1.47** (More Positive) |
| **Negative Score** | 2.77 | **1.87** | **-0.90** (Less Critical) |
| **% Mentioning OpenAI** | 68.5% | **79.1%** | **+10.6%** |
| **% Mentioning Google** | **55.6%** | 47.5% | **-8.1%** |

## 3. KEY DRIVERS (FEATURE IMPORTANCE)

The following features were the strongest predictors for the time period:
*(Negative Coefficients are linked to MAY, Positive to FEB)*

| Feature | Coefficient | Interpretation |
| :--- | :--- | :--- |
| **k_openai** | `-0.9480` | Strongly linked to **MAY** |
| **k_google** | `0.2414` | Strongly linked to **FEB** |
| **k_anthropic** | `-1.1265` | Strongly linked to **MAY** |
| **v2tone_1** | `-0.1802` | Strongly linked to **MAY** |
| **v2tone_3** | `0.2120` | Strongly linked to **FEB** |
| **v2tone_6** | `-0.0043` | Neutral / Low Impact |

## 4. INTERPRETATION

**A widening gap in Tech Giants**: OpenAI dominated both periods, but Google was significantly more prominent in Feb (55%) than in May (47%), making it a strong signal for the earlier period.

**February 2024** was defined by "Google Gemini" coverage. The correlation with higher negative scores and self-referencing language suggests this period contained significant critical analysis, op-eds, and controversy regarding Google's AI launches.

**May 2024** was defined by "OpenAI GPT-4o" and "Anthropic" coverage. The shift to positive tone and objective language indicates a reception focused more on product capabilities, launch announcements, and factual reporting rather than controversy.

## 4. SOURCE SENTIMENT ANALYSIS

### Google Coverage

**Most Positive Sources:**
| Rank | Source | Articles | Avg Sentiment |
| :--- | :--- | :--- | :--- |
| 1 | geeky-gadgets.com | 14 | **+6.71** |
| 2 | pc-tablet.co.in | 16 | **+5.76** |
| 3 | yourstory.com | 5 | **+5.31** |
| 4 | fonearena.com | 4 | **+4.81** |
| 5 | thenorthlines.com | 3 | **+4.76** |

**Most Critical Sources:**
| Rank | Source | Articles | Avg Sentiment |
| :--- | :--- | :--- | :--- |
| 148 | breitbart.com | 5 | **-2.3** |
| 147 | gizmodo.com | 8 | **-2.06** |
| 146 | bnnbloomberg.ca | 4 | **-0.69** |
| 145 | droid-life.com | 3 | **-0.34** |
| 144 | cnn.com | 11 | **-0.24** |

### OpenAI Coverage

**Most Positive Sources:**
| Rank | Source | Articles | Avg Sentiment |
| :--- | :--- | :--- | :--- |
| 1 | geeky-gadgets.com | 16 | **+6.24** |
| 2 | thenorthlines.com | 4 | **+5.88** |
| 3 | pc-tablet.co.in | 13 | **+4.13** |
| 4 | gizbot.com | 4 | **+4.05** |
| 5 | newsx.com | 3 | **+4.04** |

**Most Critical Sources:**
| Rank | Source | Articles | Avg Sentiment |
| :--- | :--- | :--- | :--- |
| 247 | outlookindia.com | 3 | **-2.55** |
| 246 | breitbart.com | 6 | **-2.03** |
| 245 | theregister.com | 6 | **-1.69** |
| 244 | sakshipost.com | 3 | **-1.63** |
| 243 | jobsnhire.com | 3 | **-1.42** |
