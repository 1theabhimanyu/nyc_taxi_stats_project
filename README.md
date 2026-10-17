# 📈 Week 3: Statistical Analysis and Hypothesis Testing

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![SciPy](https://img.shields.io/badge/Library-SciPy%20%7C%20Statsmodels-orange.svg)
![Status](https://img.shields.io/badge/Milestone-Week%203%20Complete-brightgreen.svg)

## 📌 1. Project Overview & Objective
This repository contains the **Week 3 Statistical Analysis and Hypothesis Testing** pipeline for the NYC Taxi dataset. Moving beyond descriptive statistics and visual narratives, this project applies rigorous inferential statistics to mathematically validate or refute operational and economic claims using Python (`scipy.stats` and `statsmodels`).

## 🧪 2. Hypotheses & Statistical Results

Every hypothesis was tested at a strict significance threshold of $\alpha = 0.05$ (95% confidence). Parametric tests were paired with non-parametric checks to account for continuous variable right-skewness.

| ID | Statistical Test Deployed | Null Hypothesis ($H_0$) | Result & Interpretation |
| :--- | :--- | :--- | :--- |
| **H1** | Welch's Two-Sample $t$-Test & Mann-Whitney $U$ | Mean base fare in Queens equals Manhattan. | **Rejected ($p < 0.0001$)**: Queens trips generate a statistically significant premium of $+\$13.42$ on average over Manhattan (Cohen's $d = 1.48$). |
| **H2** | One-Way ANOVA & Kruskal-Wallis | Trip duration is identical across all four boroughs. | **Rejected ($p < 0.0001$)**: ANOVA confirms borough origin explains 6.6% of trip duration variance ($\eta^2 = 0.066$). |
| **H3** | Chi-Square Test of Independence ($\chi^2$) | Payment method is independent of pickup borough. | **Rejected ($p < 0.0001$)**: Outer boroughs (Bronx, Queens) show statistically significant over-indexing in cash usage compared to Manhattan. |
| **H4** | Pearson $r$ / Spearman $\rho$ / OLS Regression | No linear correlation between trip distance and tip amount. | **Rejected ($p < 0.0001$)**: Distance explains 64.3% of the variance in credit card tips ($R^2 = 0.643$), adding $+\$0.54$/mile. |

## 📊 3. Statistical Visualizations (`outputs/week3_plots/`)
1. **`week3_viz1_ttest_fares.png`**: Box plot and point plot demonstrating the massive $\$13.42$ non-overlapping confidence interval gap between Queens and Manhattan fares.
2. **`week3_viz2_anova_duration.png`**: Point plot mapping 95% CIs and log-transformed violin densities showing duration variance.
3. **`week3_viz3_chisquare_payment.png`**: Stacked proportional bar chart alongside a heatmap of Standardized Pearson Residuals highlighting significance cells ($\vert{}z\vert{} > 1.96$).
4. **`week3_viz4_regression_tip.png`**: OLS regression line and 95% confidence band mapping credit card tips against trip distance.

## 🚀 4. How to Run Locally

1. **Create and Activate a Virtual Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # macOS/Linux
   venv\Scripts\activate     # Windows