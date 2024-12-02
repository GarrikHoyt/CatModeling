"""
==========================
Exceedance Plot Design:
==========================

Generates exceedance curves for ICU demand:
    - Sorts max ICU vals
    - Calcs exceedance probs
    - Plots county comparisons
    - Dumps key stats
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# grab county data
df_atlantic = pd.read_csv("../data/Atlantic_simulation_results.csv")
df_camden = pd.read_csv("../data/Camden_simulation_results.csv")
df_capemay = pd.read_csv("../data/Cape May_simulation_results.csv")

# extract ICU peaks
atlantic_max_icu = df_atlantic["max_icu_usage"].values
camden_max_icu = df_camden["max_icu_usage"].values
capemay_max_icu = df_capemay["max_icu_usage"].values

# sort for exceedance calcs
sorted_atlantic_icu = np.sort(atlantic_max_icu)
sorted_camden_icu = np.sort(camden_max_icu)
sorted_capemay_icu = np.sort(capemay_max_icu)

# calc exceedance probs
n = len(sorted_atlantic_icu)  # same len for all
exceedance_prob = 1.0 - np.arange(1, n + 1) / n

# setup plot
plt.figure(figsize=(10, 6))

# plot county curves
plt.step(
    sorted_atlantic_icu,
    exceedance_prob,
    where="post",
    label="Atlantic County",
    color="#4e79a7",  # soft blue
    linewidth=2,
)
plt.step(
    sorted_camden_icu,
    exceedance_prob,
    where="post",
    label="Camden County",
    color="#f28e2b",  # soft orange
    linewidth=2,
)
plt.step(
    sorted_capemay_icu,
    exceedance_prob,
    where="post",
    label="Cape May County",
    color="#59a14f",  # soft green
    linewidth=2,
)

plt.xlabel("Max ICU Demand")
plt.ylabel("Exceedance Probability")
plt.title("Exceedance Probability of Max ICU Demand by County")
plt.grid(True)
plt.legend()

# setup output path
figures_dir = "../figures/"
if not os.path.exists(figures_dir):
    os.makedirs(figures_dir)

# dump plot
plt.savefig(os.path.join(figures_dir, "exceedance_probability_plot_counties.png"))

# debug view
plt.show()

# dump summary stats
print("\nSummary Statistics:")
print(
    f"Atlantic County Max ICU - Mean: {np.mean(atlantic_max_icu):.1f}, Max: {np.max(atlantic_max_icu):.1f}"
)
print(
    f"Camden County Max ICU - Mean: {np.mean(camden_max_icu):.1f}, Max: {np.max(camden_max_icu):.1f}"
)
print(
    f"Cape May County Max ICU - Mean: {np.mean(capemay_max_icu):.1f}, Max: {np.max(capemay_max_icu):.1f}"
)
