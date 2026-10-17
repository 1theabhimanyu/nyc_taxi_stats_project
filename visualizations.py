import os
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from config import PLOTS_DIR

def plot_h1_ttest(df: pd.DataFrame):
    df_h1 = df[df["pickup_borough"].isin(["Manhattan", "Queens"])].copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    sns.boxplot(data=df_h1, x="pickup_borough", y="fare", order=["Manhattan", "Queens"], palette=["#2b83ba", "#d95f02"], width=0.45, showfliers=False, ax=axes[0])
    axes[0].set_title("1A. Base Fare Distribution (Outliers Hidden)", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Base Meter Fare ($ USD)")
    
    sns.pointplot(data=df_h1, x="pickup_borough", y="fare", order=["Manhattan", "Queens"], errorbar=("ci", 95), capsize=0.15, color="#b2182b", markers="D", linestyles="--", ax=axes[1])
    axes[1].set_title("1B. Welch's t-Test: Mean Fare & 95% CI", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Population Mean Fare Estimate ($ USD)")
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "week3_viz1_ttest_fares.png"), bbox_inches="tight")
    plt.close()

def plot_h2_anova(df: pd.DataFrame):
    boroughs = ["Manhattan", "Queens", "Brooklyn", "Bronx"]
    df_b = df[df["pickup_borough"].isin(boroughs)].copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    sns.pointplot(data=df_b, x="pickup_borough", y="trip_duration_mins", order=boroughs, errorbar=("ci", 95), capsize=0.12, palette="Set2", markers="o", ax=axes[0])
    axes[0].set_title("2A. One-Way ANOVA: Mean Duration & 95% CI", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Mean Trip Duration (Minutes)")
    
    sns.violinplot(data=df_b, x="pickup_borough", y=np.log1p(df_b["trip_duration_mins"]), order=boroughs, palette="Set2", inner="quart", ax=axes[1])
    axes[1].set_title("2B. Log-Transformed Duration Density", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Log(1 + Trip Duration in Mins)")
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "week3_viz2_anova_duration.png"), bbox_inches="tight")
    plt.close()

def plot_h3_chisquare(df: pd.DataFrame):
    boroughs = ["Manhattan", "Queens", "Brooklyn", "Bronx"]
    df_b = df[df["pickup_borough"].isin(boroughs)].copy()
    contingency_table = pd.crosstab(df_b["pickup_borough"], df_b["payment"])
    chi2_stat, p, dof, expected_freq = stats.chi2_contingency(contingency_table)
    std_residuals = (contingency_table - expected_freq) / np.sqrt(expected_freq)
    prop_table = contingency_table.div(contingency_table.sum(axis=1), axis=0) * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    prop_table.loc[boroughs].plot(kind="bar", stacked=True, color=["#fdae61", "#2b83ba"], edgecolor="black", ax=axes[0])
    axes[0].set_title("3A. Payment Method Share by Pickup Borough (%)", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Percentage of Trips (%)")
    axes[0].tick_params(axis="x", rotation=0)

    sns.heatmap(std_residuals.loc[boroughs], annot=True, fmt="+.2f", cmap="RdBu_r", center=0, cbar_kws={"label": "Standardized Residual (z-score)"}, ax=axes[1])
    axes[1].set_title("3B. Chi-Square Standardized Residuals", fontsize=12, fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "week3_viz3_chisquare_payment.png"), bbox_inches="tight")
    plt.close()

def plot_h4_regression(df: pd.DataFrame):
    df_cc = df[df["payment"] == "credit card"].copy()
    plt.figure(figsize=(10, 6))
    sns.regplot(
        data=df_cc, x="distance", y="tip", scatter_kws={"alpha": 0.35, "color": "#2b83ba", "s": 25},
        line_kws={"color": "#d7191c", "linewidth": 2.2}, ci=95
    )
    plt.title("4. Credit Card Tip vs. Trip Distance with 95% OLS Confidence Interval", fontsize=13, fontweight="bold")
    plt.xlabel("Trip Distance (Miles)")
    plt.ylabel("Recorded Credit Card Tip ($ USD)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "week3_viz4_regression_tip.png"), bbox_inches="tight")
    plt.close()

def generate_all_plots(df: pd.DataFrame):
    print("\n[INFO] Generating statistical visualizations...")
    plot_h1_ttest(df)
    plot_h2_anova(df)
    plot_h3_chisquare(df)
    plot_h4_regression(df)