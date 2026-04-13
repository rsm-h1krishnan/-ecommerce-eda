# 🛒 E-Commerce Behavioral & Revenue EDA

> **A comprehensive exploratory data analysis of a multi-year e-commerce operation — uncovering revenue drivers, customer segments, channel efficiency, and the hidden cost of discounting.**

---

## 📌 Project Overview

This project performs an end-to-end EDA on a synthetic but realistic e-commerce dataset spanning **3 years (2022–2024)**, **75,000 transactions**, and **8,000 customers** across 8 product categories. The goal is to surface actionable business intelligence that a revenue operations or business analyst team could act on immediately.

### Business Questions Answered
- Which product categories drive the most revenue — and at what margin cost?
- How does customer acquisition channel correlate with lifetime value?
- Are discounts actually driving incremental revenue, or just eroding margin?
- Which customer segments are at-risk vs. high-value, and how do we tell?
- Is there a measurable Q4 seasonality effect, and how significant is it?

---

## 📊 Analysis Highlights

| Module | Technique | Key Finding |
|---|---|---|
| Revenue Trend | Time-series + rolling averages | Q4 drives ~38% of annual revenue |
| Category Analysis | Multi-metric bar + scatter | Electronics tops revenue; Beauty has highest margin rate |
| Customer Segmentation | RFM scoring + K-Means clustering | 4 clusters identified; "Champions" (8% of base) = 31% of revenue |
| Cohort Retention | Year-over-year cohort matrix | 2022 cohort retains 61% into Year 2 vs. 47% for 2023 |
| Discount Impact | Binned analysis + margin tracking | 25-30% discounts reduce gross margin by 22% with no uplift in AOV |
| Channel Efficiency | Revenue + engagement scatter | Email and Direct channels outperform on AOV despite lower volume |
| Statistical Testing | Pearson correlation + one-way ANOVA | Segment differences in revenue are statistically significant (F=141.3, p<0.001) |

---

## 🗂️ Project Structure

```
ecommerce-eda/
├── data/
│   ├── customers.csv       # 8,000 customer profiles
│   └── transactions.csv    # 75,000 transaction records
├── src/
│   ├── generate_data.py    # Synthetic dataset generation
│   └── eda_analysis.py     # Full EDA pipeline (all 8 figures)
├── outputs/
│   ├── 01_revenue_trend.png
│   ├── 02_category_breakdown.png
│   ├── 03_segment_analysis.png
│   ├── 04_channel_device.png
│   ├── 05_rfm_clustering.png
│   ├── 06_discount_analysis.png
│   ├── 07_cohort_retention.png
│   └── 08_statistical_analysis.png
├── requirements.txt
└── README.md
```

---

## 🔧 Tech Stack

| Tool | Purpose |
|---|---|
| **Python 3.12** | Core language |
| **pandas / numpy** | Data wrangling & feature engineering |
| **matplotlib / seaborn** | All visualizations |
| **scikit-learn** | K-Means clustering, PCA, StandardScaler |
| **scipy** | ANOVA, statistical testing |
| **Faker** | Realistic synthetic data generation |

---

## 🚀 How to Run

```bash
# 1. Clone the repo
git clone https://github.com/rsm-h1krishnan/ecommerce-eda.git
cd ecommerce-eda

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate the dataset
python src/generate_data.py

# 4. Run the full EDA
python src/eda_analysis.py
# → All 8 charts saved to outputs/
```

---

## 📈 Key Findings

**1. Seasonality is Real and Predictable**
Q4 (Oct–Dec) accounts for an outsized share of revenue each year, with November and December together representing ~24% of annual GMV. This creates a clear case for pre-Q4 inventory and staffing ramp-up.

**2. Champions Are a Tiny but Critical Segment**
Using RFM-based K-Means clustering, we identify a "Champions" cluster (high recency, high frequency, high monetary) that comprises ~8% of the customer base but generates ~31% of total revenue. Churn prevention for this group has enormous leverage.

**3. Discounts Above 15% Are Value-Destructive**
Tiered discount analysis shows that order volume does not increase meaningfully beyond the 10% discount tier, while gross margins erode sharply from 15%+. The optimal discount window appears to be 5–10% for volume lift without significant margin sacrifice.

**4. Email & Direct Have Superior Unit Economics**
Despite lower transaction volumes, Email and Direct channels produce the highest average order values and the most engaged sessions (pages viewed, session duration). Paid Search drives volume but at a lower AOV.

**5. Cohort Retention Declines with Each Acquisition Year**
Year-over-year retention analysis shows that more recently acquired cohorts retain at lower rates — a warning sign that the quality of new customer acquisition may be declining, or that onboarding/early experience needs improvement.

---

## 📁 Dataset Schema

### `customers.csv`
| Column | Type | Description |
|---|---|---|
| customer_id | string | Unique identifier (C00001 format) |
| age | int | Customer age |
| gender | string | Male / Female / Non-binary |
| region | string | US geographic region |
| segment | string | Business-defined lifecycle segment |
| acquisition_channel | string | How they were acquired |
| join_year | int | Year of first registration |
| loyalty_points | int | Accumulated loyalty program points |
| email_opt_in | bool | Marketing email consent |

### `transactions.csv`
| Column | Type | Description |
|---|---|---|
| transaction_id | string | Unique transaction ID |
| customer_id | string | FK to customers |
| date | date | Transaction date |
| category | string | Product category |
| product_price | float | Unit price |
| quantity | int | Items purchased |
| discount_pct | float | Discount applied (0–0.30) |
| gross_revenue | float | Revenue before discount |
| net_revenue | float | Revenue after discount |
| gross_margin | float | Estimated gross margin $ |
| channel | string | Purchase acquisition channel |
| device | string | Mobile / Desktop / Tablet |
| is_returned | bool | Whether item was returned |
| session_minutes | float | Session duration |
| pages_viewed | int | Pages visited in session |

---

*Built as part of Hamsavi Krishnan's data analytics portfolio. See more at [Portfolio](https://www.notion.so/277d3d2c0f0d802bb6abe0b639d54cfc).*
