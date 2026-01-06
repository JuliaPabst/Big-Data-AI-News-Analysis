# GDELT THEMATIC ANALYSIS REPORT

**Analysis:** Dominant narratives by Company and Time Period.

## 1. COMPANY NARRATIVES (What are they talking about?)

Each company triggers different global themes.
*(Top 5 unique themes shown)*

### Google 
| Rank | Theme | Count |
| :--- | :--- | :--- |
| 1 | TAX_FNCACT | 1311 |
| 2 | MEDIA_SOCIAL | 586 |
| 3 | WB_133_INFORMATION_AND_COMMUNICATION_TECHNOLOGIES | 520 |
| 4 | WB_678_DIGITAL_GOVERNMENT | 517 |
| 5 | TAX_FNCACT_CEO | 462 |

### OpenAI 
| Rank | Theme | Count |
| :--- | :--- | :--- |
| 1 | TAX_FNCACT | 2019 |
| 2 | MEDIA_SOCIAL | 1055 |
| 3 | TAX_FNCACT_CEO | 1009 |
| 4 | WB_133_INFORMATION_AND_COMMUNICATION_TECHNOLOGIES | 918 |
| 5 | WB_678_DIGITAL_GOVERNMENT | 914 |

## 2. TEMPORAL SHIFT (Machine Learning Results)
We used Logistic Regression to find themes that best distinguish **February** from **May**.

### The "February" Themes (Google Era)
*Linked to Index 1 (Positive Coefficients)*
| Theme | Strength | Interpretation |
| :--- | :--- | :--- |
| **WB_653_ENTERPRISE_ARCHITECTURE** | +2.461 | Linked to **FEB** |
| **WB_2931_IRON** | +2.227 | Linked to **FEB** |
| **TAX_FNCACT_ORGANIZERS** | +1.995 | Linked to **FEB** |
| **TAX_DISEASE_ALLERGIC** | +1.947 | Linked to **FEB** |
| **TAX_FNCACT_INVENTOR** | +1.937 | Linked to **FEB** |
| **TAX_FNCACT_BARD** | +1.711 | Linked to **FEB** |
| **TAX_FNCACT_STAFFERS** | +1.413 | Linked to **FEB** |
| **TAX_ETHNICITY_AUSTRALIANS** | +1.390 | Linked to **FEB** |

### The "May" Themes (OpenAI Era)
*Linked to Index 0 (Negative Coefficients)*
| Theme | Strength | Interpretation |
| :--- | :--- | :--- |
| **WB_1226_INDUSTRIAL_CLUSTERS_AND_VALUE_CHAINS** | -2.115 | Linked to **MAY** |
| **TAX_FNCACT_AMBASSADOR** | -1.605 | Linked to **MAY** |
| **TOURISM** | -1.436 | Linked to **MAY** |
| **SOC_SUSPICIOUSACTIVITY** | -1.426 | Linked to **MAY** |
| **TAX_DISEASE_FATIGUE** | -1.219 | Linked to **MAY** |
| **TAX_MILITARY_TITLE_SOLDIERS** | -1.135 | Linked to **MAY** |
| **TAX_FNCACT_SOLDIERS** | -1.135 | Linked to **MAY** |
| **TAX_FNCACT_ADVISERS** | -1.067 | Linked to **MAY** |
