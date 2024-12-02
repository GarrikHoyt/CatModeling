"""
==========================
Model Fit Analysis:
==========================

Compares SEICICUR predictions to actual case data:
    - Loads county data
    - Fits model params
    - Plots predicted vs actual cases
    - Shows fit quality by county
"""

import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import os


def load_and_preprocess_data():
    """helper to grab and prep county data"""
    cases_df = pd.read_csv("nj_cases_deaths_2020.csv")
    cases_df["date"] = pd.to_datetime(cases_df["date"])
    pop_df = pd.read_csv("nj_county_pop.csv")

    # clean county names
    pop_df["County"] = (
        pop_df["County"].str.replace(", New Jersey", "").str.replace(".", "")
    )
    pop_df["County"] = pop_df["County"].str.replace(" County", "")
    pop_df["population_2020"] = pd.to_numeric(
        pop_df["population_2020"].str.replace(",", "")
    )

    return cases_df, pop_df


class SEICICURModel:
    """core model implementation"""

    def __init__(self, N, E0, I0, C0, ICU0, R0, beta, sigma, gamma, phi, mu, xi):
        self.N = N
        self.beta = beta
        self.sigma = sigma  # exposed -> infectious rate
        self.gamma = gamma  # confirmed -> recovered
        self.phi = phi  # ICU admission rate
        self.mu = mu  # ICU discharge rate
        self.xi = xi  # case confirmation rate

        # ini states
        self.S0 = N - E0 - I0 - C0 - ICU0 - R0
        self.E0 = E0
        self.I0 = I0
        self.C0 = C0
        self.ICU0 = ICU0
        self.R0 = R0

    def derivatives(self, t, state):
        """calc state derivatives"""
        S, E, I, C, ICU, R = state

        dSdt = -self.beta * S * I / self.N
        dEdt = self.beta * S * I / self.N - self.sigma * E
        dIdt = self.sigma * E - self.xi * I
        dCdt = (1 - self.phi) * self.xi * I - self.gamma * C
        dICUdt = self.phi * self.xi * I - self.mu * ICU
        dRdt = self.gamma * C + self.mu * ICU

        return [dSdt, dEdt, dIdt, dCdt, dICUdt, dRdt]

    def simulate(self, t):
        """run solver"""
        initial_state = [self.S0, self.E0, self.I0, self.C0, self.ICU0, self.R0]
        solution = solve_ivp(
            self.derivatives, [t[0], t[-1]], initial_state, t_eval=t, method="RK45"
        )
        return solution.y.T


def fit_model(county_data, population):
    """helper to fit model to county data"""
    observed_cases = county_data["cumulative_cases"].values
    t = np.linspace(0, len(county_data) - 1, len(county_data))

    def objective(params):
        beta, xi = params

        # lit-based params
        sigma = 1 / 5.2  # ~5d incubation
        gamma = 1 / 14  # ~14d recovery
        mu = 1 / 10  # ~10d ICU stay
        phi = 0.01  # small ICU fraction

        # ini conditions
        E0 = 10
        I0 = observed_cases[0] / xi
        C0 = (1 - phi) * observed_cases[0]
        ICU0 = phi * observed_cases[0]
        R0 = 0

        model = SEICICURModel(
            population, E0, I0, C0, ICU0, R0, beta, sigma, gamma, phi, mu, xi
        )
        solution = model.simulate(t)
        predicted_cases = solution[:, 3] + solution[:, 4]  # C + ICU

        return np.mean((predicted_cases - observed_cases) ** 2)

    # optimize beta and xi
    result = minimize(objective, [0.3, 0.3], bounds=[(0.1, 0.5), (0.1, 0.5)])
    return result.x, t


def plot_model_fit(county_name, county_data, population, best_params, t):
    """helper to plot predicted vs actual cases"""
    beta_fit, xi_fit = best_params

    # fixed params from lit
    sigma = 1 / 5.2
    gamma = 1 / 14
    mu = 1 / 10
    phi = 0.01

    # ini conditions
    observed_cases = county_data["cumulative_cases"].values
    E0 = 10
    I0 = observed_cases[0] / xi_fit
    C0 = (1 - phi) * observed_cases[0]
    ICU0 = phi * observed_cases[0]
    R0 = 0

    # run model with best fit
    model = SEICICURModel(
        population, E0, I0, C0, ICU0, R0, beta_fit, sigma, gamma, phi, mu, xi_fit
    )
    solution = model.simulate(t)
    predicted_cases = solution[:, 3] + solution[:, 4]

    return predicted_cases


def main():
    """
    fits model and generates comparison plots for:
        - Atlantic
        - Camden
        - Cape May
    """
    # setup output dir
    os.makedirs("./figures", exist_ok=True)

    # grab data
    cases_df, pop_df = load_and_preprocess_data()
    target_counties = ["Atlantic", "Camden", "Cape May"]

    # setup plot
    plt.figure(figsize=(15, 10))
    colors = ["#4e79a7", "#f28e2b", "#59a14f"]

    for idx, county in enumerate(target_counties):
        print(f"\nFitting model for {county} County...")

        # grab county data
        county_data = cases_df[cases_df["county"] == county].copy()
        population = int(pop_df[pop_df["County"] == county]["population_2020"].iloc[0])

        # fit model
        best_params, t = fit_model(county_data, population)
        predicted_cases = plot_model_fit(
            county, county_data, population, best_params, t
        )

        # plot actual vs predicted
        plt.plot(
            t,
            county_data["cumulative_cases"],
            "-",
            color=colors[idx],
            label=f"{county} Actual",
        )
        plt.plot(
            t,
            predicted_cases,
            ":",
            color=colors[idx],
            label=f"{county} Predicted",
            linewidth=2,
        )

        # calc fit metrics
        mse = np.mean((predicted_cases - county_data["cumulative_cases"]) ** 2)
        print(f"MSE for {county}: {mse:.2f}")

    # plt.xlabel("Days since first case")
    plt.ylabel("Cumulative Cases")
    plt.title("Model vs. Actual County Case Data")
    plt.legend()
    plt.grid(True)

    # dump plot
    plt.savefig("./figures/model_fit_comparison.png")
    plt.show()


if __name__ == "__main__":
    main()
