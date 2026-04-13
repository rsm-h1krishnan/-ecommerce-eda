"""
E-Commerce Behavioral & Revenue EDA
Full exploratory analysis pipeline producing all charts for the case study.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

# ─── THEME ────────────────────────────────────────────────────────────────────
PALETTE = {
    "primary":   "#2D6A4F",
    "secondary": "#52B788",
    "accent":    "#F4A261",
    "danger":    "#E76F51",
    "neutral":   "#6B6B6B",
    "light":     "#F0F4F3",
    "dark":      "#1B2B23",
}
COLORS = [PALETTE["primary"], PALETTE["secondary"], PALETTE["accent"],
          PALETTE["danger"], "#457B9D", "#A8DADC", "#E9C46A", "#264653"]

sns.set_style("whitegrid")
plt.rcParams.update({
    "font.family":    "DejaVu Sans",
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "grid.alpha":       0.3,
    "grid.color":       "#CCCCCC",
})

import os
os.makedirs("outputs", exist_ok=True)

# ─── LOAD DATA ────────────────────────────────────────────────────────────────
print("Loading data...")
cust = pd.read_csv("data/customers.csv")
txn  = pd.read_csv("data/transactions.csv", parse_dates=["date"])

df = txn.merge(cust, on="customer_id", how="left")
df["year"]    = df["date"].dt.year
df["month"]   = df["date"].dt.month
df["quarter"] = df["date"].dt.quarter
df["dow"]     = df["date"].dt.day_name()
df["week"]    = df["date"].dt.isocalendar().week.astype(int)
df["yearmonth"] = df["date"].dt.to_period("M")

print(f"  Dataset: {len(df):,} rows × {df.shape[1]} columns")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 1 – REVENUE TREND + SEASONALITY
# ─────────────────────────────────────────────────────────────────────────────
print("Fig 1: Revenue trend…")
monthly = df.groupby("yearmonth").agg(
    revenue=("net_revenue", "sum"),
    orders=("transaction_id", "count"),
    aov=("net_revenue", "mean")
).reset_index()
monthly["yearmonth_dt"] = monthly["yearmonth"].dt.to_timestamp()
monthly["rolling_3m"]   = monthly["revenue"].rolling(3).mean()

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
fig.suptitle("Monthly Revenue Trend & Order Volume (Jan 2022 – Dec 2024)",
             fontsize=16, fontweight="bold", y=1.01)

ax1 = axes[0]
ax1.fill_between(monthly["yearmonth_dt"], monthly["revenue"]/1e6,
                 alpha=0.18, color=PALETTE["primary"])
ax1.plot(monthly["yearmonth_dt"], monthly["revenue"]/1e6,
         color=PALETTE["primary"], lw=2, label="Monthly Revenue")
ax1.plot(monthly["yearmonth_dt"], monthly["rolling_3m"]/1e6,
         color=PALETTE["accent"], lw=2.5, ls="--", label="3-Month Rolling Avg")
ax1.set_ylabel("Net Revenue ($M)")
ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.1fM"))
ax1.legend(frameon=False)

# Shade Q4 each year
for yr in [2022, 2023, 2024]:
    ax1.axvspan(pd.Timestamp(f"{yr}-10-01"), pd.Timestamp(f"{yr}-12-31"),
                alpha=0.07, color=PALETTE["accent"], label="_nolegend_")
ax1.text(pd.Timestamp("2022-11-01"), monthly["revenue"].max()/1e6 * 0.88,
         "Q4", fontsize=8, color=PALETTE["accent"], ha="center", style="italic")

ax2 = axes[1]
ax2.bar(monthly["yearmonth_dt"], monthly["orders"],
        width=20, color=PALETTE["secondary"], alpha=0.75, label="Order Count")
ax2.set_ylabel("Order Count")
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")
ax2.legend(frameon=False)

plt.tight_layout()
plt.savefig("outputs/01_revenue_trend.png", dpi=160, bbox_inches="tight")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# FIG 2 – CATEGORY DEEP-DIVE
# ─────────────────────────────────────────────────────────────────────────────
print("Fig 2: Category deep-dive…")
cat_stats = df.groupby("category").agg(
    revenue=("net_revenue", "sum"),
    orders=("transaction_id", "count"),
    aov=("net_revenue", "mean"),
    avg_margin=("gross_margin", "mean"),
    return_rate=("is_returned", "mean"),
).sort_values("revenue", ascending=False).reset_index()

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Category Performance Breakdown", fontsize=16, fontweight="bold")

# Revenue bar
sns.barplot(ax=axes[0], data=cat_stats, x="revenue", y="category",
            palette=COLORS[:len(cat_stats)], orient="h")
axes[0].set_title("Total Net Revenue")
axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M"))
axes[0].set_xlabel("Net Revenue")
axes[0].set_ylabel("")

# AOV vs Margin scatter
sc = axes[1].scatter(cat_stats["aov"], cat_stats["avg_margin"],
                     s=cat_stats["orders"]/30,
                     c=COLORS[:len(cat_stats)], alpha=0.85, edgecolors="white", lw=0.5)
for _, row in cat_stats.iterrows():
    axes[1].annotate(row["category"], (row["aov"], row["avg_margin"]),
                     fontsize=7.5, ha="center", va="bottom",
                     xytext=(0, 5), textcoords="offset points")
axes[1].set_xlabel("Avg Order Value ($)")
axes[1].set_ylabel("Avg Gross Margin ($)")
axes[1].set_title("AOV vs. Margin\n(bubble = order volume)")

# Return rate
sns.barplot(ax=axes[2], data=cat_stats.sort_values("return_rate", ascending=False),
            x="return_rate", y="category", palette="Reds_r", orient="h")
axes[2].set_title("Return Rate by Category")
axes[2].xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
axes[2].set_xlabel("Return Rate")
axes[2].set_ylabel("")
axes[2].axvline(cat_stats["return_rate"].mean(), ls="--",
                color=PALETTE["neutral"], lw=1.5, label=f"Avg: {cat_stats['return_rate'].mean():.1%}")
axes[2].legend(frameon=False, fontsize=8)

plt.tight_layout()
plt.savefig("outputs/02_category_breakdown.png", dpi=160, bbox_inches="tight")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# FIG 3 – CUSTOMER SEGMENT HEATMAP + CLV BOXPLOT
# ─────────────────────────────────────────────────────────────────────────────
print("Fig 3: Segment analysis…")
seg_cat = df.groupby(["segment", "category"])["net_revenue"].sum().unstack(fill_value=0)
seg_cat_norm = seg_cat.div(seg_cat.sum(axis=1), axis=0)

clv = df.groupby("customer_id").agg(
    total_spend=("net_revenue", "sum"),
    segment=("segment", "first"),
).reset_index()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Customer Segment Intelligence", fontsize=16, fontweight="bold")

# Heatmap
seg_order = ["New", "Occasional", "Regular", "Loyal", "Champion"]
sns.heatmap(seg_cat_norm.loc[seg_order] * 100, ax=axes[0],
            annot=True, fmt=".1f", cmap="Greens",
            linewidths=0.5, cbar_kws={"label": "Revenue Share (%)", "shrink": 0.8})
axes[0].set_title("Revenue Mix by Segment × Category (%)")
axes[0].set_xlabel("")
axes[0].set_ylabel("Customer Segment")
axes[0].tick_params(axis="x", rotation=30)

# CLV boxplot
seg_palette = {s: c for s, c in zip(seg_order, COLORS[:5])}
bp = sns.boxplot(ax=axes[1], data=clv, x="segment", y="total_spend",
                 order=seg_order, palette=seg_palette, showfliers=False,
                 boxprops=dict(alpha=0.85))
sns.stripplot(ax=axes[1], data=clv.sample(min(2000, len(clv))),
              x="segment", y="total_spend", order=seg_order,
              palette=seg_palette, alpha=0.2, size=3, jitter=True)
axes[1].set_title("Customer Lifetime Value Distribution by Segment\n(box = IQR, dots = individual customers)")
axes[1].set_xlabel("Customer Segment")
axes[1].set_ylabel("Total Spend ($)")
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

plt.tight_layout()
plt.savefig("outputs/03_segment_analysis.png", dpi=160, bbox_inches="tight")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# FIG 4 – CHANNEL & DEVICE FUNNEL
# ─────────────────────────────────────────────────────────────────────────────
print("Fig 4: Channel & device…")
ch_stats = df.groupby("channel").agg(
    revenue=("net_revenue", "sum"),
    orders=("transaction_id", "count"),
    aov=("net_revenue", "mean"),
    session=("session_minutes", "mean"),
    pages=("pages_viewed", "mean"),
).sort_values("revenue", ascending=False).reset_index()

dev_stats = df.groupby("device").agg(
    revenue=("net_revenue", "sum"),
    aov=("net_revenue", "mean"),
).reset_index()

fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle("Acquisition Channel & Device Performance", fontsize=16, fontweight="bold")

# Channel revenue
bars = axes[0].barh(ch_stats["channel"][::-1], ch_stats["revenue"][::-1]/1e6,
                    color=COLORS[:len(ch_stats)], edgecolor="white")
axes[0].xaxis.set_major_formatter(mticker.FormatStrFormatter("$%.1fM"))
axes[0].set_title("Revenue by Channel")
axes[0].set_xlabel("Net Revenue")
for bar, val in zip(bars, ch_stats["revenue"][::-1]/1e6):
    axes[0].text(val + 0.02, bar.get_y() + bar.get_height()/2,
                 f"${val:.1f}M", va="center", fontsize=8)

# Engagement scatter (session vs pages, sized by revenue)
sc = axes[1].scatter(ch_stats["session"], ch_stats["pages"],
                     s=ch_stats["revenue"]/5000,
                     c=COLORS[:len(ch_stats)], alpha=0.85,
                     edgecolors="white", lw=0.8)
for _, row in ch_stats.iterrows():
    axes[1].annotate(row["channel"], (row["session"], row["pages"]),
                     fontsize=7.5, ha="center", va="bottom",
                     xytext=(0, 6), textcoords="offset points")
axes[1].set_xlabel("Avg Session Duration (min)")
axes[1].set_ylabel("Avg Pages Viewed")
axes[1].set_title("Engagement Quality by Channel\n(bubble = total revenue)")

# Device donut
wedges, texts, autotexts = axes[2].pie(
    dev_stats["revenue"], labels=dev_stats["device"],
    autopct="%1.1f%%", startangle=90,
    colors=[PALETTE["primary"], PALETTE["secondary"], PALETTE["accent"]],
    wedgeprops=dict(width=0.5), pctdistance=0.75,
    textprops={"fontsize": 10}
)
for at in autotexts:
    at.set_fontweight("bold")
axes[2].set_title("Revenue Share by Device")

plt.tight_layout()
plt.savefig("outputs/04_channel_device.png", dpi=160, bbox_inches="tight")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# FIG 5 – RFM CLUSTERING
# ─────────────────────────────────────────────────────────────────────────────
print("Fig 5: RFM clustering…")
SNAPSHOT = pd.Timestamp("2025-01-01")
rfm_raw = txn.groupby("customer_id").agg(
    recency=("date",   lambda x: (SNAPSHOT - x.max()).days),
    frequency=("transaction_id", "count"),
    monetary=("net_revenue", "sum"),
).reset_index()

scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_raw[["recency", "frequency", "monetary"]])

# Elbow
inertia = []
K_RANGE = range(2, 9)
for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(rfm_scaled)
    inertia.append(km.inertia_)

km_final = KMeans(n_clusters=4, random_state=42, n_init=10)
rfm_raw["cluster"] = km_final.fit_predict(rfm_scaled)

# PCA for 2D viz
pca = PCA(n_components=2, random_state=42)
pca_coords = pca.fit_transform(rfm_scaled)
rfm_raw["pca1"] = pca_coords[:, 0]
rfm_raw["pca2"] = pca_coords[:, 1]

# Cluster labels (heuristic: rank by monetary mean)
cluster_means = rfm_raw.groupby("cluster")[["recency", "frequency", "monetary"]].mean()
# Lower recency is better, higher freq/monetary is better
cluster_means["score"] = (
    -cluster_means["recency"] / cluster_means["recency"].max()
    + cluster_means["frequency"] / cluster_means["frequency"].max()
    + cluster_means["monetary"]  / cluster_means["monetary"].max()
)
label_map = {row: lbl for row, lbl in zip(
    cluster_means.sort_values("score").index,
    ["At-Risk", "Passive", "Potential Loyalist", "Champions"]
)}
rfm_raw["cluster_label"] = rfm_raw["cluster"].map(label_map)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("RFM-Based Customer Segmentation (K-Means Clustering)", fontsize=16, fontweight="bold")

# Elbow
axes[0].plot(K_RANGE, inertia, marker="o", color=PALETTE["primary"], lw=2)
axes[0].axvline(4, ls="--", color=PALETTE["accent"], lw=1.5, label="Optimal k=4")
axes[0].set_xlabel("Number of Clusters (k)")
axes[0].set_ylabel("Inertia (WCSS)")
axes[0].set_title("Elbow Method for Optimal k")
axes[0].legend(frameon=False)

# PCA scatter
cluster_colors = {
    "At-Risk": PALETTE["danger"],
    "Passive": PALETTE["neutral"],
    "Potential Loyalist": PALETTE["secondary"],
    "Champions": PALETTE["primary"],
}
for lbl, grp in rfm_raw.groupby("cluster_label"):
    axes[1].scatter(grp["pca1"], grp["pca2"], label=lbl,
                    color=cluster_colors[lbl], alpha=0.45, s=8, edgecolors="none")
axes[1].set_title("Customer Clusters in PCA Space")
axes[1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
axes[1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
axes[1].legend(frameon=False, markerscale=3, fontsize=9)

# Cluster profile radar data as grouped bar
metrics = ["recency", "frequency", "monetary"]
cluster_profile = rfm_raw.groupby("cluster_label")[metrics].mean()
cluster_profile_norm = (cluster_profile - cluster_profile.min()) / (cluster_profile.max() - cluster_profile.min())
cluster_profile_norm["recency"] = 1 - cluster_profile_norm["recency"]  # flip: lower recency = better

x = np.arange(3)
width = 0.18
labels_order = ["Champions", "Potential Loyalist", "Passive", "At-Risk"]
for i, lbl in enumerate(labels_order):
    if lbl in cluster_profile_norm.index:
        vals = cluster_profile_norm.loc[lbl, metrics].values
        axes[2].bar(x + i * width, vals, width, label=lbl,
                    color=cluster_colors[lbl], alpha=0.85)
axes[2].set_xticks(x + width * 1.5)
axes[2].set_xticklabels(["Recency\n(inv.)", "Frequency", "Monetary"], fontsize=9)
axes[2].set_ylabel("Normalized Score (0–1)")
axes[2].set_title("Cluster Profile Comparison\n(Normalized Metrics)")
axes[2].legend(frameon=False, fontsize=8)

plt.tight_layout()
plt.savefig("outputs/05_rfm_clustering.png", dpi=160, bbox_inches="tight")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# FIG 6 – DISCOUNT IMPACT ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
print("Fig 6: Discount analysis…")
disc_bins = pd.cut(df["discount_pct"], bins=[-0.01, 0, 0.05, 0.10, 0.15, 0.20, 0.30],
                   labels=["0%", "5%", "10%", "15%", "20%", "25-30%"])
df["disc_bucket"] = disc_bins

disc_stats = df.groupby("disc_bucket", observed=True).agg(
    orders=("transaction_id", "count"),
    revenue=("net_revenue", "sum"),
    avg_margin=("gross_margin", "mean"),
    avg_qty=("quantity", "mean"),
    return_rate=("is_returned", "mean"),
).reset_index()

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("Discount Strategy Impact Analysis", fontsize=16, fontweight="bold")

# Orders by discount tier
axes[0,0].bar(disc_stats["disc_bucket"].astype(str), disc_stats["orders"],
              color=COLORS[:len(disc_stats)], edgecolor="white")
axes[0,0].set_title("Order Volume by Discount Tier")
axes[0,0].set_xlabel("Discount %")
axes[0,0].set_ylabel("Number of Orders")
axes[0,0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

# Revenue by discount tier
axes[0,1].bar(disc_stats["disc_bucket"].astype(str), disc_stats["revenue"]/1e6,
              color=COLORS[:len(disc_stats)], edgecolor="white")
axes[0,1].set_title("Net Revenue by Discount Tier")
axes[0,1].set_xlabel("Discount %")
axes[0,1].set_ylabel("Net Revenue ($M)")
axes[0,1].yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.1fM"))

# Margin erosion
axes[1,0].plot(disc_stats["disc_bucket"].astype(str), disc_stats["avg_margin"],
               marker="o", color=PALETTE["primary"], lw=2)
axes[1,0].fill_between(range(len(disc_stats)), disc_stats["avg_margin"],
                        alpha=0.12, color=PALETTE["primary"])
axes[1,0].set_title("Avg Gross Margin vs. Discount Level")
axes[1,0].set_xlabel("Discount %")
axes[1,0].set_ylabel("Avg Gross Margin ($)")
axes[1,0].yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.2f"))

# Return rate by discount
axes[1,1].bar(disc_stats["disc_bucket"].astype(str), disc_stats["return_rate"] * 100,
              color=[PALETTE["danger"] if r > disc_stats["return_rate"].mean() else PALETTE["secondary"]
                     for r in disc_stats["return_rate"]],
              edgecolor="white")
axes[1,1].axhline(disc_stats["return_rate"].mean() * 100, ls="--",
                   color=PALETTE["neutral"], lw=1.5,
                   label=f"Avg: {disc_stats['return_rate'].mean():.1%}")
axes[1,1].set_title("Return Rate by Discount Tier")
axes[1,1].set_xlabel("Discount %")
axes[1,1].set_ylabel("Return Rate (%)")
axes[1,1].legend(frameon=False)

plt.tight_layout()
plt.savefig("outputs/06_discount_analysis.png", dpi=160, bbox_inches="tight")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# FIG 7 – COHORT RETENTION
# ─────────────────────────────────────────────────────────────────────────────
print("Fig 7: Cohort retention…")
txn_ext = txn.copy()
txn_ext["year"] = txn_ext["date"].dt.year
first_year = txn_ext.groupby("customer_id")["year"].min().rename("cohort_year")
txn_ext = txn_ext.merge(first_year, on="customer_id")
txn_ext["years_since_join"] = txn_ext["year"] - txn_ext["cohort_year"]

cohort_counts = txn_ext.groupby(["cohort_year", "years_since_join"])["customer_id"].nunique().unstack()
cohort_size   = cohort_counts[0]
retention     = cohort_counts.div(cohort_size, axis=0) * 100

fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(retention.iloc[:, :4], ax=ax, annot=True, fmt=".1f",
            cmap="Greens", linewidths=0.5, vmin=0, vmax=100,
            cbar_kws={"label": "Retention Rate (%)", "shrink": 0.8},
            annot_kws={"size": 11})
ax.set_title("Annual Cohort Retention Matrix\n(% of cohort still purchasing in Year N)",
             fontsize=14, fontweight="bold")
ax.set_xlabel("Years Since First Purchase")
ax.set_ylabel("Acquisition Year (Cohort)")
ax.set_yticklabels([f"Cohort {y}" for y in retention.index], rotation=0)
ax.set_xticklabels([f"Year {x}" for x in range(len(retention.columns[:4]))], rotation=0)

plt.tight_layout()
plt.savefig("outputs/07_cohort_retention.png", dpi=160, bbox_inches="tight")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# FIG 8 – STATISTICAL TESTING (ANOVA + CORRELATION)
# ─────────────────────────────────────────────────────────────────────────────
print("Fig 8: Statistical testing…")

# ANOVA across segments
groups = [grp["net_revenue"].values for _, grp in df.groupby("segment")]
f_stat, p_val = stats.f_oneway(*groups)

# Correlation matrix for numeric features
num_cols = ["product_price", "quantity", "discount_pct", "net_revenue",
            "gross_margin", "session_minutes", "pages_viewed"]
corr = df[num_cols].corr()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Statistical Analysis: Correlations & Segment Revenue Distribution",
             fontsize=15, fontweight="bold")

# Correlation heatmap
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, ax=axes[0], annot=True, fmt=".2f", cmap="RdYlGn",
            linewidths=0.5, vmin=-1, vmax=1,
            cbar_kws={"shrink": 0.8},
            annot_kws={"size": 9})
axes[0].set_title("Pearson Correlation Matrix\n(Key Transaction Variables)")
axes[0].tick_params(axis="x", rotation=35)
axes[0].tick_params(axis="y", rotation=0)

# KDE per segment
seg_order = ["New", "Occasional", "Regular", "Loyal", "Champion"]
seg_palette2 = dict(zip(seg_order, COLORS[:5]))
for seg in seg_order:
    data = df[df["segment"] == seg]["net_revenue"]
    data_clipped = data[data < data.quantile(0.99)]
    sns.kdeplot(data_clipped, ax=axes[1], label=seg,
                color=seg_palette2[seg], lw=2, fill=True, alpha=0.12)
axes[1].set_title(f"Revenue Distribution by Segment\n"
                  f"(One-Way ANOVA: F={f_stat:.1f}, p={'<0.001' if p_val < 0.001 else f'{p_val:.4f}'})")
axes[1].set_xlabel("Net Revenue per Transaction ($)")
axes[1].set_ylabel("Density")
axes[1].legend(frameon=False, title="Segment")
axes[1].set_xlim(0, 500)

plt.tight_layout()
plt.savefig("outputs/08_statistical_analysis.png", dpi=160, bbox_inches="tight")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY STATS (used in README / Notion)
# ─────────────────────────────────────────────────────────────────────────────
print("\n─── SUMMARY ───────────────────────────────────────────")
print(f"Total Revenue:      ${df['net_revenue'].sum():>12,.0f}")
print(f"Total Orders:       {len(df):>12,}")
print(f"Unique Customers:   {df['customer_id'].nunique():>12,}")
print(f"Avg Order Value:    ${df['net_revenue'].mean():>12.2f}")
print(f"Avg Gross Margin:   ${df['gross_margin'].mean():>12.2f}")
print(f"Overall Return Rate:{df['is_returned'].mean():>11.1%}")
top_cat = df.groupby("category")["net_revenue"].sum().idxmax()
print(f"Top Category:       {top_cat:>15}")
top_ch  = df.groupby("channel")["net_revenue"].sum().idxmax()
print(f"Top Channel:        {top_ch:>15}")
print("────────────────────────────────────────────────────────")
print("All charts saved to outputs/")
