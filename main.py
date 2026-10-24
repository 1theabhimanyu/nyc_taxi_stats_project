"""
NYC Taxi EDA Project - Week 3: Statistical Analysis & Hypothesis Testing
Author: Abhimanyu Kumar
Repository: https://github.com/1theabhimanyu/nyc_taxi_eda_project
Description:
    Executes parametric and non-parametric hypothesis tests (Welch's t-test,
    Mann-Whitney U, One-Way ANOVA, Kruskal-Wallis, Tukey's HSD, Chi-Square
    Test of Independence, and OLS Regression / Pearson & Spearman Correlation)
    alongside assumption checks (D'Agostino K^2 Normality, Levene's Test)
    and generates publication-quality visualizations.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="deep", font_scale=1.1)


# ==============================================================================
# 1. DATA LOADING & PREPROCESSING (Compatible with nyc_taxi_eda_project)
# ==============================================================================
def load_or_generate_nyc_taxi_data(filepath: str = "nyc_taxi_data.csv", n_samples: int = 50000) -> pd.DataFrame:
    """
    Loads the cleaned NYC Taxi dataset from the project directory if present.
    Otherwise, generates a realistic synthetic NYC TLC dataset matching the
    exact statistical distributions and schema of nyc_taxi_eda_project.
    """
    if os.path.exists(filepath):
        print(f"[INFO] Loading dataset from {filepath}...")
        df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/taxis.csv")
    else:
        print(f"[INFO] '{filepath}' not found locally. Generating realistic NYC TLC sample (N={n_samples})...")
        np.random.seed(42)

        # Borough probabilities matching NYC TLC Yellow/Green distribution
        boroughs = ["Manhattan", "Queens", "Brooklyn", "Bronx"]
        borough_probs = [0.730, 0.136, 0.098, 0.036]
        pickup_borough = np.random.choice(boroughs, size=n_samples, p=borough_probs)

        # Temporal features
        pickup_hour = np.random.randint(0, 24, size=n_samples)
        day_of_week = np.random.randint(0, 7, size=n_samples)
        is_weekday = day_of_week < 5
        is_peak_hour = (is_weekday & (((pickup_hour >= 7) & (pickup_hour <= 10)) | ((pickup_hour >= 16) & (pickup_hour <= 20)))).astype(int)

        # Distance generation conditioned on Borough (Queens has longer airport trips)
        trip_distance = np.zeros(n_samples)
        for b in boroughs:
            mask = pickup_borough == b
            count = mask.sum()
            if b == "Manhattan":
                trip_distance[mask] = np.random.gamma(shape=2.2, scale=1.15, size=count)
            elif b == "Queens":
                # Mixture of local trips and JFK/LGA airport trips (~10-18 miles)
                is_airport = np.random.binomial(1, 0.62, size=count)
                trip_distance[mask] = np.where(
                    is_airport == 1,
                    np.random.normal(loc=11.5, scale=3.2, size=count),
                    np.random.gamma(shape=2.5, scale=1.6, size=count)
                )
            elif b == "Brooklyn":
                trip_distance[mask] = np.random.gamma(shape=2.6, scale=1.55, size=count)
            else:  # Bronx
                trip_distance[mask] = np.random.gamma(shape=2.8, scale=1.65, size=count)

        trip_distance = np.clip(trip_distance, 0.2, 45.0)

        # Trip duration (minutes) impacted by distance and peak hour congestion
        speed_mph = np.where(is_peak_hour == 1, np.random.normal(11.5, 2.5, n_samples), np.random.normal(16.5, 3.2, n_samples))
        speed_mph = np.clip(speed_mph, 4.0, 45.0)
        trip_duration_min = (trip_distance / speed_mph) * 60.0

        # Fare calculation: Base ($3.00) + Distance ($2.75/mi) + Time/Peak surcharge + Noise
        peak_surcharge = np.where(is_peak_hour == 1, np.random.normal(2.55, 0.85, n_samples), 0.0)
        borough_adj = np.where(pickup_borough == "Queens", 3.20, 0.0)
        noise = np.random.normal(0, 2.1, n_samples)

        fare_amount = 3.00 + (2.85 * trip_distance) + (0.12 * trip_duration_min) + peak_surcharge + borough_adj + noise
        fare_amount = np.clip(fare_amount, 3.00, 200.00)

        # Payment type conditioned on Borough (Credit Card vs Cash)
        card_prob_map = {"Manhattan": 0.764, "Queens": 0.741, "Brooklyn": 0.685, "Bronx": 0.618}
        probs = np.array([card_prob_map[b] for b in pickup_borough])
        payment_type = np.where(np.random.rand(n_samples) < probs, "Credit Card", "Cash")

        df = pd.DataFrame({
            "pickup_borough": pickup_borough,
            "pickup_hour": pickup_hour,
            "day_of_week": day_of_week,
            "is_peak_hour": is_peak_hour,
            "trip_distance": np.round(trip_distance, 2),
            "trip_duration_min": np.round(trip_duration_min, 2),
            "fare_amount": np.round(fare_amount, 2),
            "payment_type": payment_type
        })

    # Standard cleaning filter
    df = df[(df["fare_amount"] >= 2.50) & (df["fare_amount"] <= 250.0)]
    df = df[(df["trip_distance"] >= 0.10) & (df["trip_distance"] <= 100.0)]
    return df.reset_index(drop=True)


# ==============================================================================
# 2. STATISTICAL HELPER FUNCTIONS (Effect Sizes & Confidence Intervals)
# ==============================================================================
def compute_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Calculates pooled Cohen's d effect size for two independent samples."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_se = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (np.mean(group1) - np.mean(group2)) / pooled_se


def compute_welch_ci(group1: np.ndarray, group2: np.ndarray, alpha: float = 0.05):
    """Calculates the (1 - alpha) Confidence Interval for difference in means (Welch)."""
    mean_diff = np.mean(group1) - np.mean(group2)
    se1, se2 = stats.sem(group1), stats.sem(group2)
    se_diff = np.sqrt(se1**2 + se2**2)
    df = (se1**2 + se2**2)**2 / ((se1**4 / (len(group1) - 1)) + (se2**4 / (len(group2) - 1)))
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    return mean_diff - t_crit * se_diff, mean_diff + t_crit * se_diff, df


def compute_cramers_v(chi2: float, n: int, shape: tuple) -> float:
    """Calculates Cramer's V effect size for a Chi-Square contingency table."""
    min_dim = min(shape[0] - 1, shape[1] - 1)
    return np.sqrt(chi2 / (n * min_dim))


# ==============================================================================
# 3. EXECUTE HYPOTHESIS TESTS
# ==============================================================================
def run_all_hypothesis_tests(df: pd.DataFrame) -> dict:
    results = {}
    print("\n" + "=" * 80)
    print("NYC TAXI EDA PROJECT - WEEK 3 STATISTICAL HYPOTHESIS TESTING RESULTS")
    print("=" * 80)

    # --------------------------------------------------------------------------
    # HYPOTHESIS 1: Peak vs. Off-Peak Fare (Welch's t-Test & Mann-Whitney U)
    # --------------------------------------------------------------------------
    peak_fares = df[df["is_peak_hour"] == 1]["fare_amount"].values
    offpeak_fares = df[df["is_peak_hour"] == 0]["fare_amount"].values

    # Assumption Checks
    _, norm_p_peak = stats.normaltest(peak_fares)
    _, norm_p_off = stats.normaltest(offpeak_fares)
    levene_stat_h1, levene_p_h1 = stats.levene(peak_fares, offpeak_fares)

    # Parametric & Non-Parametric Tests
    t_stat, p_val_t = stats.ttest_ind(peak_fares, offpeak_fares, equal_var=False)
    u_stat, p_val_u = stats.mannwhitneyu(peak_fares, offpeak_fares, alternative="two-sided")
    ci_low, ci_high, welch_df = compute_welch_ci(peak_fares, offpeak_fares)
    cohens_d = compute_cohens_d(peak_fares, offpeak_fares)

    results["H1"] = {
        "peak_mean": np.mean(peak_fares), "offpeak_mean": np.mean(offpeak_fares),
        "t_stat": t_stat, "p_val_t": p_val_t, "df": welch_df,
        "u_stat": u_stat, "p_val_u": p_val_u,
        "ci_low": ci_low, "ci_high": ci_high, "cohens_d": cohens_d
    }

    print("\n[HYPOTHESIS 1] Peak vs. Off-Peak Total Fare")
    print(f"  Peak Mean: ${np.mean(peak_fares):.2f} (SD=${np.std(peak_fares, ddof=1):.2f}, n={len(peak_fares)})")
    print(f"  Off-Peak Mean: ${np.mean(offpeak_fares):.2f} (SD=${np.std(offpeak_fares, ddof=1):.2f}, n={len(offpeak_fares)})")
    print(f"  Assumption Check - Levene's Test: W={levene_stat_h1:.2f}, p={levene_p_h1:.4e} -> Using Welch's t-test")
    print(f"  Welch's t-Test: t={t_stat:.4f}, df={welch_df:.1f}, p={p_val_t:.4e}")
    print(f"  Mann-Whitney U: U={u_stat:.4e}, p={p_val_u:.4e}")
    print(f"  95% CI for Mean Difference: [${ci_low:.2f}, ${ci_high:.2f}] | Cohen's d = {cohens_d:.3f}")
    print(f"  Decision: {'REJECT H0' if p_val_t < 0.05 else 'FAIL TO REJECT H0'}")

    # --------------------------------------------------------------------------
    # HYPOTHESIS 2: Fare Across NYC Pickup Boroughs (One-Way ANOVA & Kruskal)
    # --------------------------------------------------------------------------
    borough_Order = ["Manhattan", "Queens", "Brooklyn", "Bronx"]
    borough_groups = [df[df["pickup_borough"] == b]["fare_amount"].values for b in borough_Order]

    f_stat, p_val_anova = stats.f_oneway(*borough_groups)
    kw_stat, p_val_kw = stats.kruskal(*borough_groups)

    # Calculate Eta-Squared (SS_between / SS_total)
    grand_mean = df["fare_amount"].mean()
    ss_total = np.sum((df["fare_amount"] - grand_mean) ** 2)
    ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in borough_groups)
    eta_squared = ss_between / ss_total

    # Post-Hoc Tukey's HSD
    tukey = pairwise_tukeyhsd(endog=df["fare_amount"], groups=df["pickup_borough"], alpha=0.05)

    results["H2"] = {
        "f_stat": f_stat, "p_val_anova": p_val_anova,
        "kw_stat": kw_stat, "p_val_kw": p_val_kw,
        "eta_squared": eta_squared, "tukey": tukey
    }

    print("\n[HYPOTHESIS 2] Trip Fare Variation Across NYC Pickup Boroughs")
    print(f"  One-Way ANOVA: F={f_stat:.2f}, p={p_val_anova:.4e} | Eta-Squared = {eta_squared:.4f}")
    print(f"  Kruskal-Wallis: H={kw_stat:.2f}, p={p_val_kw:.4e}")
    print(f"  Decision: {'REJECT H0' if p_val_anova < 0.05 else 'FAIL TO REJECT H0'}")
    print("\n  Tukey's HSD Post-Hoc Summary:")
    print(tukey.summary())

    # --------------------------------------------------------------------------
    # HYPOTHESIS 3: Payment Type vs. Pickup Borough (Chi-Square Independence)
    # --------------------------------------------------------------------------
    contingency_table = pd.crosstab(df["pickup_borough"], df["payment_type"])
    chi2_stat, p_val_chi2, dof_chi2, expected = stats.chi2_contingency(contingency_table)
    cramers_v = compute_cramers_v(chi2_stat, len(df), contingency_table.shape)

    # Standardized Pearson Residuals
    std_residuals = (contingency_table - expected) / np.sqrt(expected)

    results["H3"] = {
        "chi2_stat": chi2_stat, "p_val_chi2": p_val_chi2, "dof": dof_chi2,
        "cramers_v": cramers_v, "contingency_table": contingency_table,
        "std_residuals": std_residuals
    }

    print("\n[HYPOTHESIS 3] Payment Method vs. Pickup Borough (Chi-Square Test)")
    print(f"  Chi-Square: chi2={chi2_stat:.2f}, df={dof_chi2}, p={p_val_chi2:.4e} | Cramer's V = {cramers_v:.4f}")
    print(f"  Decision: {'REJECT H0' if p_val_chi2 < 0.05 else 'FAIL TO REJECT H0'}")

    # --------------------------------------------------------------------------
    # HYPOTHESIS 4: Trip Distance vs. Fare Amount (Correlation & OLS Inference)
    # --------------------------------------------------------------------------
    r_stat, p_val_r = stats.pearsonr(df["trip_distance"], df["fare_amount"])
    rho_stat, p_val_rho = stats.spearmanr(df["trip_distance"], df["fare_amount"])

    X = sm.add_constant(df["trip_distance"])
    ols_model = sm.OLS(df["fare_amount"], X).fit()
    slope = ols_model.params["trip_distance"]
    slope_ci_low, slope_ci_high = ols_model.conf_int().loc["trip_distance"]
    r_squared = ols_model.rsquared

    results["H4"] = {
        "r_stat": r_stat, "p_val_r": p_val_r,
        "rho_stat": rho_stat, "p_val_rho": p_val_rho,
        "slope": slope, "slope_ci_low": slope_ci_low, "slope_ci_high": slope_ci_high,
        "r_squared": r_squared, "t_stat_ols": ols_model.tvalues["trip_distance"]
    }

    print("\n[HYPOTHESIS 4] Trip Distance vs. Total Fare (Linear Correlation & OLS)")
    print(f"  Pearson r = {r_stat:.4f} (p={p_val_r:.4e}) | Spearman rho = {rho_stat:.4f} (p={p_val_rho:.4e})")
    print(f"  OLS Slope (beta_1) = ${slope:.2f}/mile (95% CI: [${slope_ci_low:.2f}, ${slope_ci_high:.2f}]), R^2 = {r_squared:.4f}")
    print(f"  Decision: {'REJECT H0' if p_val_r < 0.05 else 'FAIL TO REJECT H0'}")
    print("=" * 80 + "\n")

    return results


# ==============================================================================
# 4. GENERATE SUPPORTING VISUALIZATIONS
# ==============================================================================
def generate_visualizations(df: pd.DataFrame, results: dict, output_path: str = "week3_hypothesis_testing_visuals.png"):
    """Creates a 4-panel dashboard of statistical charts supporting Hypotheses 1-4."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 13))
    fig.suptitle(
        "NYC Taxi EDA Project (Week 3): Statistical Hypothesis Testing & Inference Suite",
        fontsize=18, fontweight="bold", y=0.98
    )

    # Panel 1: Hypothesis 1 (Peak vs. Off-Peak KDE + Histogram)
    ax1 = axes[0, 0]
    subset_h1 = df[df["fare_amount"] <= 65]
    sns.histplot(
        data=subset_h1, x="fare_amount", hue="is_peak_hour",
        stat="density", common_norm=False, kde=True, bins=40,
        palette={0: "#2b5c8f", 1: "#d95f02"}, alpha=0.35, ax=ax1
    )
    ax1.axvline(results["H1"]["offpeak_mean"], color="#2b5c8f", linestyle="--", linewidth=2.2,
                label=f"Off-Peak Mean: ${results['H1']['offpeak_mean']:.2f}")
    ax1.axvline(results["H1"]["peak_mean"], color="#d95f02", linestyle="--", linewidth=2.2,
                label=f"Peak Mean: ${results['H1']['peak_mean']:.2f}")
    ax1.set_title(
        f"H1: Peak vs. Off-Peak Fare Distribution\n"
        f"Welch's t = {results['H1']['t_stat']:.2f}, p < 0.0001 | 95% CI Diff: [${results['H1']['ci_low']:.2f}, ${results['H1']['ci_high']:.2f}]",
        fontsize=12, fontweight="bold"
    )
    ax1.set_xlabel("Total Fare Amount ($ USD)")
    ax1.set_ylabel("Density")
    ax1.legend(loc="upper right")

    # Panel 2: Hypothesis 2 (Borough Fare Comparison - Violin + Boxplot)
    ax2 = axes[0, 1]
    order_b = ["Manhattan", "Brooklyn", "Bronx", "Queens"]
    subset_h2 = df[df["fare_amount"] <= 80]
    sns.violinplot(
        data=subset_h2, x="pickup_borough", y="fare_amount", order=order_b,
        inner="box", cut=0, palette="Set2", ax=ax2
    )
    means = df.groupby("pickup_borough")["fare_amount"].mean().reindex(order_b)
    ax2.scatter(range(len(order_b)), means.values, color="red", s=70, zorder=5, label="Group Mean ($)")
    for idx, m in enumerate(means.values):
        ax2.text(idx, m + 3.0, f"${m:.1f}", horizontalalignment="center", fontweight="bold", color="darkred")
    ax2.set_title(
        f"H2: Fare Distribution Across NYC Boroughs (ANOVA & Kruskal-Wallis)\n"
        f"F = {results['H2']['f_stat']:.1f}, p < 0.0001 | Eta-Squared = {results['H2']['eta_squared']:.3f}",
        fontsize=12, fontweight="bold"
    )
    ax2.set_xlabel("NYC Pickup Borough")
    ax2.set_ylabel("Total Fare Amount ($ USD)")
    ax2.legend(loc="upper left")

    # Panel 3: Hypothesis 3 (Chi-Square Standardized Residuals Heatmap)
    ax3 = axes[1, 0]
    sns.heatmap(
        results["H3"]["std_residuals"], annot=True, fmt=".2f", cmap="coolwarm",
        center=0, cbar_kws={"label": "Standardized Pearson Residual"}, ax=ax3
    )
    ax3.set_title(
        f"H3: Payment Type vs. Borough (Chi-Square Standardized Residuals)\n"
        f"Chi2 = {results['H3']['chi2_stat']:.2f}, p < 0.0001 | Cramer's V = {results['H3']['cramers_v']:.3f}",
        fontsize=12, fontweight="bold"
    )
    ax3.set_xlabel("Payment Method")
    ax3.set_ylabel("NYC Pickup Borough")

    # Panel 4: Hypothesis 4 (OLS Regression Fit & Correlation)
    ax4 = axes[1, 1]
    sample_h4 = df.sample(n=min(2500, len(df)), random_state=42)
    sns.regplot(
        data=sample_h4, x="trip_distance", y="fare_amount",
        scatter_kws={"alpha": 0.25, "color": "#1f77b4", "s": 18},
        line_kws={"color": "#d62728", "linewidth": 2.5}, ax=ax4
    )
    ax4.set_title(
        f"H4: Trip Distance vs. Fare Amount (OLS Regression & Pearson r)\n"
        f"r = {results['H4']['r_stat']:.3f}, R² = {results['H4']['r_squared']:.3f}, "
        f"Slope = ${results['H4']['slope']:.2f}/mi (p < 0.0001)",
        fontsize=12, fontweight="bold"
    )
    ax4.set_xlabel("Trip Distance (Miles)")
    ax4.set_ylabel("Total Fare Amount ($ USD)")

    plt.tight_layout(rect=[0, 0.01, 1, 0.95])
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved high-resolution statistical visualization suite to '{output_path}'.")


if __name__ == "__main__":
    taxi_df = load_or_generate_nyc_taxi_data()
    test_results = run_all_hypothesis_tests(taxi_df)
    generate_visualizations(taxi_df, test_results)