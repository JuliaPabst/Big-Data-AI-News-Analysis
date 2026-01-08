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

**February 2024** was defined by "Google Gemini" coverage. The correlation with higher negative scores and self-referencing language suggests this period contained significant critical analysis, op-eds, and controversy regarding Google's AI launches.

**May 2024** was defined by "OpenAI GPT-4o" and "Anthropic" coverage. The shift to positive tone and objective language indicates a reception focused more on product capabilities, launch announcements, and factual reporting rather than controversy.
