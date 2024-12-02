"""
==========================
Data Aggregation Design:
==========================

Combines county-level simulation results:
    - Loads individual county data
    - Calcs system-wide metrics
    - Generates summary stats
    - Dumps to csv for analysis
"""

import pandas as pd
import numpy as np
import os

# grab raw county data
df_atlantic = pd.read_csv("../data/Atlantic_simulation_results.csv")
df_camden = pd.read_csv("../data/Camden_simulation_results.csv")
df_capemay = pd.read_csv("../data/Cape May_simulation_results.csv")

# setup aggregation df
aggregated_results = pd.DataFrame()

# grab shared cols
aggregated_results["phi"] = df_atlantic["phi"]
aggregated_results["sim_number"] = df_atlantic["sim_number"]

# calc ICU usage per county + combined
aggregated_results["atlantic_max_icu"] = df_atlantic["max_icu_usage"]
aggregated_results["camden_max_icu"] = df_camden["max_icu_usage"]
aggregated_results["capemay_max_icu"] = df_capemay["max_icu_usage"]
aggregated_results["combined_max_icu"] = (
    aggregated_results["atlantic_max_icu"]
    + aggregated_results["camden_max_icu"]
    + aggregated_results["capemay_max_icu"]
)

# add capacity baselines
aggregated_results["atlantic_capacity"] = df_atlantic["icu_capacity"]
aggregated_results["camden_capacity"] = df_camden["icu_capacity"]
aggregated_results["capemay_capacity"] = df_capemay["icu_capacity"]
aggregated_results["combined_capacity"] = (
    aggregated_results["atlantic_capacity"]
    + aggregated_results["camden_capacity"]
    + aggregated_results["capemay_capacity"]
)

# calc peak usage ratios
aggregated_results["atlantic_peak_ratio"] = df_atlantic["peak_icu_ratio"]
aggregated_results["camden_peak_ratio"] = df_camden["peak_icu_ratio"]
aggregated_results["capemay_peak_ratio"] = df_capemay["peak_icu_ratio"]
aggregated_results["combined_peak_ratio"] = (
    aggregated_results["combined_max_icu"] / aggregated_results["combined_capacity"]
)

# calc capacity breach metrics
aggregated_results["atlantic_days_exceeded"] = df_atlantic["days_exceeded"]
aggregated_results["camden_days_exceeded"] = df_camden["days_exceeded"]
aggregated_results["capemay_days_exceeded"] = df_capemay["days_exceeded"]
aggregated_results["combined_days_exceeded"] = (
    aggregated_results["atlantic_days_exceeded"]
    + aggregated_results["camden_days_exceeded"]
    + aggregated_results["capemay_days_exceeded"]
)

aggregated_results["atlantic_prob_exceeded"] = df_atlantic["prob_exceeded"]
aggregated_results["camden_prob_exceeded"] = df_camden["prob_exceeded"]
aggregated_results["capemay_prob_exceeded"] = df_capemay["prob_exceeded"]
aggregated_results["combined_prob_exceeded"] = (
    aggregated_results["combined_peak_ratio"] > 1.0
).astype(float)

# setup output path
output_dir = "../data/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# dump results
output_file = os.path.join(output_dir, "aggregated_simulation_results.csv")
aggregated_results.to_csv(output_file, index=False)

# print summary stats
print("\nSummary Statistics:")
print(f"Total simulations: {len(aggregated_results)}")
print("\nICU Capacity:")
print(f"Atlantic County: {aggregated_results['atlantic_capacity'].iloc[0]}")
print(f"Camden County: {aggregated_results['camden_capacity'].iloc[0]}")
print(f"Cape May County: {aggregated_results['capemay_capacity'].iloc[0]}")
print(f"Combined System: {aggregated_results['combined_capacity'].iloc[0]}")
print("\nPeak ICU Ratio Statistics:")
print(
    f"Atlantic County - Mean: {aggregated_results['atlantic_peak_ratio'].mean():.3f}, Max: {aggregated_results['atlantic_peak_ratio'].max():.3f}"
)
print(
    f"Camden County - Mean: {aggregated_results['camden_peak_ratio'].mean():.3f}, Max: {aggregated_results['camden_peak_ratio'].max():.3f}"
)
print(
    f"Cape May County - Mean: {aggregated_results['capemay_peak_ratio'].mean():.3f}, Max: {aggregated_results['capemay_peak_ratio'].max():.3f}"
)
print(
    f"Combined System - Mean: {aggregated_results['combined_peak_ratio'].mean():.3f}, Max: {aggregated_results['combined_peak_ratio'].max():.3f}"
)
print(f"\nFile saved as: {output_file}")

# debug check
print("\nFirst few rows of the aggregated results:")
print(aggregated_results.head())
