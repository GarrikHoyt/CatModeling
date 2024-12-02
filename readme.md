# COVID-19 ICU Capacity Simulation Model for New Jersey

## Overview
This project implements a sophisticated SEICICUR (Susceptible-Exposed-Infectious-Confirmed-ICU-Recovered) epidemiological model to simulate and analyze COVID-19 transmission dynamics and ICU capacity requirements across New Jersey counties. The model specifically focuses on predicting ICU utilization and identifying potential capacity constraints in the healthcare system.

## Project Structure
```
├── sim.py                     # Main simulation model implementation
├── data/                      # Directory for processed data
├── figures/                   # Directory for generated plots and visualizations
├── scripts/                   # Directory for auxiliary scripts
│   ├── aggregate.py           # Data aggregation script
│   ├── create_epc.py         # Exceedance Probability Curve generation
├── county_populations.csv     # County-wise population data
├── hospital_resources.csv     # Hospital resource availability data
├── nj_county_pop.csv         # New Jersey county population data
├── nj_cases_deaths_2020.csv  # COVID-19 cases and deaths data for 2020
├── nj_cases_by_county.csv    # County-wise COVID-19 case data
└── nj_hosp_resources.csv     # Hospital resource data for New Jersey
```

## Model Description
The SEICICUR model is an extension of the traditional SEIR model, incorporating additional compartments to better represent the progression of COVID-19 cases through the healthcare system. The model includes:

- Susceptible (S)
- Exposed (E)
- Infectious (I)
- Confirmed (C)
- ICU (ICU)
- Recovered (R)

Key parameters include:
- β (beta): Transmission rate
- σ (sigma): Rate of progression from exposed to infectious
- γ (gamma): Recovery rate for non-ICU cases
- φ (phi): Proportion of confirmed cases requiring ICU care
- μ (mu): ICU discharge rate
- ξ (xi): Case confirmation rate

## Dependencies
- Python 3.x
- pandas
- numpy
- scipy
- matplotlib
- seaborn
- joblib
- tqdm

## Data Sources
The project utilizes various data sources for New Jersey:
- County-level population data
- COVID-19 case and death statistics
- Hospital resource availability
- ICU capacity information

## Usage
1. Ensure all required dependencies are installed
2. Place the required data files in the project root directory
3. Run the simulation using:
   ```python
   python sim.py
   ```

## Output
The simulation generates:
- ICU capacity utilization predictions
- Peak ICU usage estimates
- Probability of exceeding ICU capacity
- Various visualization plots in the `figures/` directory

## Analysis Capabilities
- County-wise COVID-19 transmission dynamics
- ICU capacity assessment
- Healthcare resource utilization forecasting
- Risk analysis for healthcare system overload

## Scripts
The `scripts/` directory contains utility scripts for data processing and visualization:

### aggregate.py
A data aggregation script that:
- Combines simulation results from multiple counties (Atlantic, Camden, Cape May)
- Calculates aggregate ICU metrics including:
  - Combined maximum ICU usage
  - Peak ICU ratios
  - Days exceeded capacity
  - Combined capacity utilization
- Generates consolidated datasets for multi-county analysis

### create_epc.py
An Exceedance Probability Curve (EPC) generation script that:
- Creates visualizations of ICU capacity exceedance probabilities
- Generates step plots for comparing ICU usage across counties
- Provides visual analysis of the likelihood of exceeding ICU capacity
- Uses color-coded plotting for different counties (Atlantic, Camden, Cape May)

These scripts are essential for post-processing simulation results and creating visualizations for risk assessment and capacity planning.

## Data Files
The project uses several CSV files containing New Jersey COVID-19 and healthcare data:

### COVID-19 Case Data
- `nj_cases_deaths_2020.csv`: Daily COVID-19 case and death counts
  - Time series data from March 2020
  - Includes: date, county, cumulative cases, cumulative deaths
  - FIPS codes for geographic identification

- `nj_cases_by_county.csv`: Detailed county-level case data
  - County-specific COVID-19 case information
  - Used for county-level transmission analysis

### Healthcare Resource Data
- `hospital_resources.csv`: Hospital capacity information
  - Licensed, staffed, and ICU bed counts
  - Ventilator availability
  - Age demographic breakdowns (65-85+ years)
  - County-specific healthcare infrastructure data

- `nj_hosp_resources.csv`: New Jersey hospital resource tracking
  - Specific to NJ healthcare facilities
  - ICU capacity metrics
  - Resource utilization data

### Population Data
- `county_populations.csv`: Basic population statistics
  - County-level population counts
  - Used for per capita calculations

- `nj_county_pop.csv`: Detailed NJ population data
  - 2020 population figures for all NJ counties
  - Used for demographic modeling parameters

All data files are structured in CSV format and are essential for:
- Model calibration
- Parameter estimation
- Capacity planning
- Risk assessment
- Results validation

## License
This project is licensed under the MIT License - see below for details:

MIT License

Copyright (c) 2024 [Garrik A. Hoyt, Sachin Ramnath Arunkumar]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Citation

For use before publication, please cite this repository:
G.A. Hoyt, S.R. Arunkumar. (2024). COVID-19 ICU Capacity Simulation Model for New Jersey. GitHub repository: https://github.com/GarrikHoyt/CatModeling

## Contributors
Garrik A. Hoyt, Sachin Ramnath Arunkumar

## Contact
gah223@lehigh.edu, sar624@lehigh.edu
