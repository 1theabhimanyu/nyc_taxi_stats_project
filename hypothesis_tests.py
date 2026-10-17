import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

def run_all_hypothesis_tests(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("EXECUTING INFERENTIAL HYPOTHESIS TESTS")
    print("=" * 60)

    # H1: Welch's t-Test & Mann-Whitney U (Queens vs Manhattan Fare)
    qns_fare = df.loc[df["pickup_borough"] == "Queens", "fare"]
    man_fare = df.loc[df["pickup_borough"] == "Manhattan", "fare"]
    t_stat, p_val_t = stats.ttest_ind(qns_fare, man_fare, equal_var=False, alternative="greater")
    u_stat, p_val_u = stats.mannwhitneyu(qns_fare, man_fare, alternative="greater")
    
    diff_mean = qns_fare.mean() - man_fare.mean()
    pooled_sd = np.sqrt(((len(qns_fare)-1)*qns_fare.var(ddof=1) + (len(man_fare)-1)*man_fare.var(ddof=1)) / (len(qns_fare)+len(man_fare)-2))
    cohens_d = diff_mean / pooled_sd
    print(f"[H1: Two-Sample Means] Welch's t = {t_stat:.4f}, p = {p_val_t:.4e} | Cohen's d = {cohens_d:.3f}")

    # H2: One-Way ANOVA & Kruskal-Wallis (Trip Duration Across Boroughs)
    boroughs = ["Manhattan", "Queens", "Brooklyn", "Bronx"]
    df_b = df[df["pickup_borough"].isin(boroughs)].copy()
    groups = [df_b.loc[df_b["pickup_borough"] == b, "trip_duration_mins"].values for b in boroughs]
    
    f_stat, p_val_f = stats.f_oneway(*groups)
    kw_stat, p_val_kw = stats.kruskal(*groups)
    
    grand_mean = df_b["trip_duration_mins"].mean()
    ss_total = ((df_b["trip_duration_mins"] - grand_mean)**2).sum()
    ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for g in groups)
    eta_sq = ss_between / ss_total
    print(f"[H2: ANOVA] F = {f_stat:.4f}, p = {p_val_f:.4e} | Eta-squared = {eta_sq:.4f}")
    
    # H3: Chi-Square Test of Independence (Payment vs Borough)
    contingency_table = pd.crosstab(df_b["pickup_borough"], df_b["payment"])
    chi2_stat, p_val_chi2, dof_chi2, expected_freq = stats.chi2_contingency(contingency_table)
    cramers_v = np.sqrt(chi2_stat / (contingency_table.sum().sum() * (min(contingency_table.shape) - 1)))
    print(f"[H3: Chi-Square] Chi^2 = {chi2_stat:.4f}, p = {p_val_chi2:.4e} | Cramer's V = {cramers_v:.4f}")

    # H4: Pearson Correlation & OLS (Distance vs. Credit Card Tip)
    df_cc = df[df["payment"] == "credit card"].copy()
    r_stat, p_val_r = stats.pearsonr(df_cc["distance"], df_cc["tip"])
    
    X_ols = sm.add_constant(df_cc["distance"])
    ols_model = sm.OLS(df_cc["tip"], X_ols).fit()
    slope = ols_model.params["distance"]
    print(f"[H4: Correlation/OLS] Pearson r = {r_stat:.4f}, p = {p_val_r:.4e} | OLS Slope = ${slope:.3f}/mile")