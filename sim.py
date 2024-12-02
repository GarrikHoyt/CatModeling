import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from joblib import Parallel, delayed
from tqdm import tqdm
import matplotlib
import os

matplotlib.use("Agg")

"""
==========================
Project Architecture:
==========================

Key components:
    - SEICICUR epidemiological model implementation
    - Monte Carlo param exploration
    - County-level + combined analysis
    - Visual output generation
"""

# helper consts for param exploration
phi_values = np.linspace(0.001, 0.026, 250)


def load_and_preprocess_data():
    """
    helper to grab raw data and prep for model:
        - loads case/death counts
        - grabs hospital resource info
        - processes county population stats
        - calcs daily case deltas
    """
    cases_df = pd.read_csv("nj_cases_deaths_2020.csv")
    cases_df["date"] = pd.to_datetime(cases_df["date"])
    resources_df = pd.read_csv("nj_hosp_resources.csv")

    # grab population data
    pop_df = pd.read_csv("nj_county_pop.csv")
    # clean county names, convert pop to numeric
    pop_df["County"] = (
        pop_df["County"].str.replace(", New Jersey", "").str.replace(".", "")
    )
    pop_df["County"] = pop_df["County"].str.replace(" County", "")
    pop_df["population_2020"] = pd.to_numeric(
        pop_df["population_2020"].str.replace(",", "")
    )

    # get daily case deltas for model calibration
    cases_df = cases_df.sort_values(["county", "date"])
    cases_df["new_cases"] = (
        cases_df.groupby("county")["cumulative_cases"].diff().fillna(0)
    )

    return cases_df, resources_df, pop_df


class SEICICURModel:
    """
    ==========================
    SEICICUR Model Design:
    ==========================

    Implements proper case-to-ICU dynamics with:
        - phi: proportion of cases needing ICU
        - xi: rate of case confirmation
        - mu: ICU discharge rate
        - gamma: non-ICU recovery rate
    """

    def __init__(self, N, E0, I0, C0, ICU0, R0, beta, sigma, gamma, phi, mu, xi):
        self.N = N
        self.beta = beta
        self.sigma = sigma  # exposed -> infectious rate
        self.gamma = gamma  # confirmed -> recovered rate
        self.phi = phi  # ICU admission rate
        self.mu = mu  # ICU discharge rate
        self.xi = xi  # case confirmation rate

        # set initial compartment sizes
        self.S0 = N - E0 - I0 - C0 - ICU0 - R0
        self.E0 = E0
        self.I0 = I0
        self.C0 = C0
        self.ICU0 = ICU0
        self.R0 = R0

    def derivatives(self, t, state):
        """helper to calc compartment derivatives for solver"""
        S, E, I, C, ICU, R = state

        # core transmission dynamics
        dSdt = -self.beta * S * I / self.N
        dEdt = self.beta * S * I / self.N - self.sigma * E
        dIdt = self.sigma * E - self.xi * I
        dCdt = (1 - self.phi) * self.xi * I - self.gamma * C  # non-ICU track
        dICUdt = self.phi * self.xi * I - self.mu * ICU  # ICU track
        dRdt = self.gamma * C + self.mu * ICU  # recovery flows

        return [dSdt, dEdt, dIdt, dCdt, dICUdt, dRdt]

    def simulate(self, t):
        """run solver over timespan t"""
        initial_state = [self.S0, self.E0, self.I0, self.C0, self.ICU0, self.R0]
        solution = solve_ivp(
            self.derivatives, [t[0], t[-1]], initial_state, t_eval=t, method="RK45"
        )
        return solution.y.T


def run_simulation(
    N, E0, I0, C0, ICU0, R0, beta, sigma, gamma, phi, mu, xi, t, icu_capacity
):
    """
    helper to run single sim and extract key metrics:
        - peak ICU usage
        - days over capacity
        - failure probs
        - ICU utilization ratios
    """
    model = SEICICURModel(N, E0, I0, C0, ICU0, R0, beta, sigma, gamma, phi, mu, xi)
    solution = model.simulate(t)

    # grab ICU metrics
    icu_usage = solution[:, 4]  # ICU compartment
    max_icu_usage = np.max(icu_usage)
    days_exceeded = np.sum(icu_usage > icu_capacity)
    prob_exceeded = days_exceeded / len(icu_usage)
    peak_icu_ratio = max_icu_usage / icu_capacity

    return {
        "max_icu_usage": max_icu_usage,
        "days_exceeded": days_exceeded,
        "prob_exceeded": prob_exceeded,
        "peak_icu_ratio": peak_icu_ratio,
    }


def run_monte_carlo_simulation(
    county_data,
    resources_df,
    population,
    county_name,
    n_simulations=4000,
    icu_capacity=None,
):
    """
    ==========================
    Monte Carlo Design:
    ==========================

    - Runs n_simulations with param variations
    - Tests range of ICU admission rates (phi)
    - Tracks capacity breaches & system stress
    - Parallel execution for speed
    """
    if icu_capacity is None:
        county_resources = resources_df[resources_df["NAME"] == f"{county_name} County"]
        icu_capacity = int(county_resources["Beds_ICU"].iloc[0])

    N = int(population)

    # grab case data for calibration
    observed_cases = county_data["cumulative_cases"].values
    t = np.linspace(0, len(county_data) - 1, len(county_data))

    # helper to calibrate model with case data
    def objective(params):
        beta, xi = params

        # lit-based fixed params
        sigma = 1 / 5.2  # ~5d incubation
        gamma = 1 / 14  # ~14d recovery
        mu = 1 / 10  # ~10d ICU stay
        phi = 0.01  # start low

        # ini conditions from day 1
        E0 = 10
        I0 = observed_cases[0] / xi
        C0 = (1 - phi) * observed_cases[0]  # non-ICU confirmed
        ICU0 = phi * observed_cases[0]  # ICU cases
        R0 = 0

        model = SEICICURModel(N, E0, I0, C0, ICU0, R0, beta, sigma, gamma, phi, mu, xi)
        solution = model.simulate(t)
        predicted_total_cases = solution[:, 3] + solution[:, 4]  # C + ICU

        return np.mean((predicted_total_cases - observed_cases) ** 2)

    # optimize beta and xi fit
    result = minimize(objective, [0.3, 0.3], bounds=[(0.1, 0.5), (0.1, 0.5)])
    beta_fit, xi_fit = result.x

    results = []

    # param distributions from lit
    betas = np.random.uniform(0.21, 0.3, n_simulations)
    sigmas = 1 / np.random.uniform(2.2, 6, n_simulations)
    gammas = 1 / np.random.uniform(4, 14, n_simulations)
    xis = np.clip(np.random.normal(xi_fit, 0.02, n_simulations), 0.1, 0.5)
    E0s = np.random.uniform(5, 15, n_simulations)

    # fixed params
    mu = 1 / 10  # ~10d ICU stay

    # ini from day 1 data
    I0_fixed = observed_cases[0] / xi_fit
    R0_fixed = 0

    # prep sim args
    sim_args = []
    for phi in phi_values:
        # calc ini conditions per phi
        C0_fixed = (1 - phi) * observed_cases[0]
        ICU0_fixed = phi * observed_cases[0]

        for i in range(n_simulations):
            sim_args.append(
                (
                    N,
                    E0s[i],
                    I0_fixed,
                    C0_fixed,
                    ICU0_fixed,
                    R0_fixed,
                    betas[i],
                    sigmas[i],
                    gammas[i],
                    phi,
                    mu,
                    xis[i],
                    t,
                    icu_capacity,
                )
            )

    # parallel sim execution
    n_jobs = -1  # max cores
    simulations = Parallel(n_jobs=n_jobs)(
        delayed(run_simulation)(*args)
        for args in tqdm(sim_args, desc=f"Processing {county_name}")
    )

    # organize outputs
    idx = 0
    for phi in phi_values:
        for sim_number in range(n_simulations):
            sim_result = simulations[idx]
            idx += 1
            results.append(
                {
                    "phi": phi,
                    "sim_number": sim_number,
                    "max_icu_usage": sim_result["max_icu_usage"],
                    "days_exceeded": sim_result["days_exceeded"],
                    "prob_exceeded": sim_result["prob_exceeded"],
                    "peak_icu_ratio": sim_result["peak_icu_ratio"],
                    "icu_capacity": icu_capacity,
                    # track params
                    "N": N,
                    "E0": E0s[sim_number],
                    "I0": I0_fixed,
                    "C0": C0_fixed,
                    "ICU0": ICU0_fixed,
                    "R0": R0_fixed,
                    "beta": betas[sim_number],
                    "sigma": sigmas[sim_number],
                    "gamma": gammas[sim_number],
                    "mu": mu,
                    "xi": xis[sim_number],
                }
            )

    return pd.DataFrame(results)


def analyze_failure_distribution(monte_carlo_results):
    """
    helper to analyze system failures:
        - groups by ICU admission rate
        - calcs key percentiles
        - gets failure probs at diff thresholds
    """
    # group by phi
    grouped = monte_carlo_results.groupby("phi")

    failure_stats = grouped.agg(
        {
            "prob_exceeded": ["mean", "std", lambda x: np.percentile(x, 95)],
            "peak_icu_ratio": ["mean", "std", lambda x: np.percentile(x, 95)],
            "days_exceeded": ["mean", "std", lambda x: np.percentile(x, 95)],
        }
    ).round(3)

    failure_prob = grouped["peak_icu_ratio"].apply(lambda x: (x > 1.0).mean()).round(3)
    failure_prob_75 = (
        grouped["peak_icu_ratio"].apply(lambda x: (x > 0.75).mean()).round(3)
    )
    failure_prob_50 = (
        grouped["peak_icu_ratio"].apply(lambda x: (x > 0.5).mean()).round(3)
    )

    return failure_stats, failure_prob, failure_prob_75, failure_prob_50


def create_visualizations(
    county, results, failure_prob, failure_prob75, failure_prob50
):
    """helper to make failure analysis plots"""
    # new fig per plot for thread safety
    fig = plt.figure(figsize=(15, 8))

    # ICU ratio plot
    ax1 = fig.add_subplot(2, 1, 1)
    sns.lineplot(data=results, x="phi", y="peak_icu_ratio", errorbar=("ci", 95), ax=ax1)
    ax1.axhline(y=1.0, color="r", linestyle="--", label="Capacity Threshold")
    ax1.set_title(f"{county} - ICU Demand/Capacity Ratio vs Phi")
    ax1.set_ylabel("Peak ICU Demand/Capacity Ratio")
    ax1.legend()

    # failure prob plot
    ax2 = fig.add_subplot(2, 1, 2)
    failure_prob.plot(
        ax=ax2, label="Failure Probability 100%", linestyle="-", color="red"
    )
    failure_prob75.plot(
        ax=ax2, label="Failure Probability 75%", linestyle=":", color="blue"
    )
    failure_prob50.plot(
        ax=ax2, label="Failure Probability 50%", linestyle="--", color="green"
    )
    ax2.set_title("Probability of System Failure vs Phi")
    ax2.set_xlabel("Phi (ICU Admission Rate)")
    ax2.set_ylabel("Probability of Failure")
    ax2.legend()

    plt.tight_layout()
    fig.savefig(f"./figures/{county}_failure_analysis.png")
    plt.close(fig)


def create_comparison_plots(county_results, failure_distributions):
    """helper to compare counties"""
    # failure prob comparison
    plt.figure(figsize=(12, 8))
    counties = ["Atlantic", "Camden", "Cape May", "Combined"]
    colors = ["blue", "green", "red", "purple"]

    for county, color in zip(counties, colors):
        plt.plot(
            phi_values,
            failure_distributions[county]["failure_prob"],
            label=f"{county}",
            color=color,
        )

    plt.title("Failure Probability Comparison")
    plt.xlabel("Phi (ICU Admission Rate)")
    plt.ylabel("Probability of System Failure")
    plt.legend()
    plt.grid(True)
    plt.savefig("./figures/failure_probability_comparison.png")
    plt.close()

    # ICU usage plot
    plt.figure(figsize=(12, 8))
    for county, color in zip(counties, colors):
        data = county_results[county]
        sns.lineplot(
            data=data, x="phi", y="peak_icu_ratio", label=f"{county}", color=color
        )

    plt.axhline(y=1.0, color="black", linestyle="--", label="Capacity Threshold")
    plt.title("Peak ICU Usage Ratio Comparison")
    plt.xlabel("Phi (ICU Admission Rate)")
    plt.ylabel("Peak ICU Demand/Capacity Ratio")
    plt.legend()
    plt.grid(True)
    plt.savefig("./figures/icu_usage_comparison.png")
    plt.close()


def run_county_analysis(
    cases_df, resources_df, pop_df, county_name, n_simulations=4000, is_combined=False
):
    """
    helper for county-level analysis:
        - handles individual or combined counties
        - runs Monte Carlo sims
        - saves results to csv
    """
    if is_combined:
        # combined county calcs
        counties = ["Atlantic", "Camden", "Cape May"]
        county_cases = (
            cases_df[cases_df["county"].isin(counties)]
            .groupby("date")
            .sum()
            .reset_index()
        )
        county_cases["county"] = "Combined"
        total_population = pop_df[pop_df["County"].isin(counties)][
            "population_2020"
        ].sum()
        total_icu_beds = resources_df[
            resources_df["NAME"].isin([f"{c} County" for c in counties])
        ]["Beds_ICU"].sum()

        results = run_monte_carlo_simulation(
            county_cases,
            resources_df,
            total_population,
            "Combined",
            n_simulations=n_simulations,
            icu_capacity=total_icu_beds,
        )
    else:
        # single county calcs
        county_cases = cases_df[cases_df["county"] == county_name]
        county_resources = resources_df[resources_df["NAME"] == f"{county_name} County"]
        county_population = pop_df[pop_df["County"] == county_name][
            "population_2020"
        ].iloc[0]
        icu_capacity = int(county_resources["Beds_ICU"].iloc[0])

        results = run_monte_carlo_simulation(
            county_cases,
            resources_df,
            county_population,
            county_name,
            n_simulations=n_simulations,
            icu_capacity=icu_capacity,
        )

    # dump results
    results.to_csv(f"./data/{county_name}_simulation_results.csv", index=False)

    return results


def main():
    """
    ==========================
    Main Analysis Pipeline:
    ==========================

    - Sets up output dirs
    - Loads & preps data
    - Runs county-level analysis
    - Generates visualizations
    - Dumps results to files
    """
    # setup dirs
    os.makedirs("./figures", exist_ok=True)
    os.makedirs("./data", exist_ok=True)

    # grab data
    cases_df, resources_df, pop_df = load_and_preprocess_data()

    # target counties
    target_counties = ["Atlantic", "Camden", "Cape May"]

    # results storage
    county_results = {}
    failure_distributions = {}

    # run individual analyses
    for county in target_counties:
        print(f"\nAnalyzing {county} County...")
        results = run_county_analysis(cases_df, resources_df, pop_df, county)
        county_results[county] = results

        # get failure stats
        failure_stats, failure_prob, failure_prob_75, failure_prob_50 = (
            analyze_failure_distribution(results)
        )
        failure_distributions[county] = {
            "stats": failure_stats,
            "failure_prob50": failure_prob_50,
            "failure_prob75": failure_prob_75,
            "failure_prob": failure_prob,
        }

        # make county plots
        create_visualizations(
            county, results, failure_prob, failure_prob_75, failure_prob_50
        )

    # run combined analysis
    print("\nAnalyzing combined counties...")
    combined_results = run_county_analysis(
        cases_df, resources_df, pop_df, "Combined", is_combined=True
    )
    county_results["Combined"] = combined_results

    # get combined failure stats
    failure_stats, failure_prob, failure_prob_75, failure_prob_50 = (
        analyze_failure_distribution(combined_results)
    )
    failure_distributions["Combined"] = {
        "stats": failure_stats,
        "failure_prob50": failure_prob_50,
        "failure_prob75": failure_prob_75,
        "failure_prob": failure_prob,
    }

    # make combined plots
    create_visualizations(
        "Combined", combined_results, failure_prob, failure_prob_75, failure_prob_50
    )

    # comparison plots
    create_comparison_plots(county_results, failure_distributions)

    # dump failure probs to csv
    failure_prob_df = pd.DataFrame(
        {
            "phi": phi_values,
            "Atlantic_50": failure_distributions["Atlantic"]["failure_prob50"],
            "Camden_50": failure_distributions["Camden"]["failure_prob50"],
            "Cape_May_50": failure_distributions["Cape May"]["failure_prob50"],
            "Combined_50": failure_distributions["Combined"]["failure_prob50"],
            "Atlantic_75": failure_distributions["Atlantic"]["failure_prob75"],
            "Camden_75": failure_distributions["Camden"]["failure_prob75"],
            "Cape_May_75": failure_distributions["Cape May"]["failure_prob75"],
            "Combined_75": failure_distributions["Combined"]["failure_prob75"],
            "Atlantic_100": failure_distributions["Atlantic"]["failure_prob"],
            "Camden_100": failure_distributions["Camden"]["failure_prob"],
            "Cape_May_100": failure_distributions["Cape May"]["failure_prob"],
            "Combined_100": failure_distributions["Combined"]["failure_prob"],
        }
    )
    failure_prob_df.to_csv("failure_probabilities.csv", index=False)

    # dump stats summary
    summary_stats = {
        county: data["stats"] for county, data in failure_distributions.items()
    }
    with pd.ExcelWriter("./data/summary_statistics.xlsx") as writer:
        for county, stats in summary_stats.items():
            stats.to_excel(
                writer, sheet_name=county[:31]
            )  # Excel sheet names limited to 31 chars

    return county_results, failure_distributions


if __name__ == "__main__":
    results, distributions = main()

    # dump critical phi vals
    print("\nCritical phi values (50% failure probability threshold):")
    for county in ["Atlantic", "Camden", "Cape May", "Combined"]:
        critical_phi = phi_values[
            np.where(distributions[county]["failure_prob"] > 0.5)[0][0]
        ]
        print(f"{county}: {critical_phi:.4f}")
