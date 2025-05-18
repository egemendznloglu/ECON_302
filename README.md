# fetch_and_process_financial_data.py

## Description  
This script fetches five financial time series at monthly frequency, merges them into a single DataFrame, performs stationarity tests, selects optimal lag lengths, estimates an OLS model with HAC standard errors, runs diagnostic tests, and visualizes results with two plots (coefficient forest plot and CUSUM parameter stability plot).

## Features  
- Download adjusted monthly returns for NASDAQ 100 (NDX) via Yahoo Finance  
- Download month-end series for federal funds rate (FEDFUNDS), 10-year Treasury yield (GS10), CPI (CPIAUCSL), and VIX (VIXCLS) via FRED  
- Validate date inputs (`YYYY-MM-DD`) and ensure start date is before end date  
- Run Augmented Dickey–Fuller tests on each series (levels and transformed)  
- Transform non-stationary series to stationarity (first differences or percentage changes)  
- Select optimal lag length by AIC, BIC, and FPE up to a user-defined maximum  
- Estimate final OLS regression of NDX returns on stationary transforms with HAC (Newey–West) standard errors  
- Log SSR, FPE, Durbin–Watson, Breusch–Godfrey, White heteroskedasticity, Jarque–Bera, and CUSUM parameter-stability tests  
- Generate two matplotlib plots:  
  1. Forest plot of coefficient estimates with 95% confidence intervals  
  2. CUSUM parameter stability plot  

## Requirements  
- Python 3.7 or higher  
- pandas  
- pandas_datareader  
- yfinance  
- statsmodels  
- matplotlib  

## Installation  
1. Create and activate a virtual environment (optional but recommended):  
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
1. Install dependencies:  
   ```bash
   pip install pandas pandas_datareader yfinance statsmodels matplotlib
   ```
## Usage
   ```bash
   python fetch_and_process_financial_data.py [start_date] [end_date]
   ```
 - `start_date` and `end_date` must use the format `YYYY-MM-DD`.
 - If no arguments are provided, defaults to `2020-01-01` through `2024-04-30`.

 Example:
   ```bash
    python fetch_and_process_financial_data.py 2019-01-01 2025-04-30
   ```

## Configuration
 - Logging is configured at the INFO level to output timestamps, log level, and messages to standard output.
 - Change `DATE_FMT` in the script to adjust the date parsing format if needed.
 - Adjust the `max_lag` parameter in `select_lag_length` to change the maximum number of lags tested.

## Output
- The script logs progress and results to the console. An example run on May 18, 2025 produced:
```text
2025-05-18 00:24:25,413 INFO Combined DataFrame columns: ['ndx_ret', 'ffr', 'gs10', 'cpi', 'vix']
2025-05-18 00:24:25,413 INFO Downloaded data with 52 rows.
2025-05-18 00:24:25,432 INFO ADF test ndx_ret: stat=-7.5258, p=3.6877e-11, crit(1%)=-3.5685, crit(5%)=-2.9214, crit(10%)=-2.5987
2025-05-18 00:24:25,438 INFO ADF test ffr: stat=-1.0159, p=7.4741e-01, crit(1%)=-3.5778, crit(5%)=-2.9253, crit(10%)=-2.6008
2025-05-18 00:24:25,445 INFO ADF test gs10: stat=-0.1679, p=9.4224e-01, crit(1%)=-3.5715, crit(5%)=-2.9226, crit(10%)=-2.5993
2025-05-18 00:24:25,451 INFO ADF test cpi: stat=-1.2456, p=6.5367e-01, crit(1%)=-3.5746, crit(5%)=-2.9240, crit(10%)=-2.6000
2025-05-18 00:24:25,456 INFO ADF test vix: stat=-3.1188, p=2.5190e-02, crit(1%)=-3.5715, crit(5%)=-2.9226, crit(10%)=-2.5993
2025-05-18 00:24:25,463 INFO ADF test ndx_ret: stat=-7.5258, p=3.6877e-11, crit(1%)=-3.5685, crit(5%)=-2.9214, crit(10%)=-2.5987
2025-05-18 00:24:25,468 INFO ADF test d_ffr: stat=-2.2139, p=2.0125e-01, crit(1%)=-3.5778, crit(5%)=-2.9253, crit(10%)=-2.6008
2025-05-18 00:24:25,472 INFO ADF test d_gs10: stat=-5.9474, p=2.1875e-07, crit(1%)=-3.5715, crit(5%)=-2.9226, crit(10%)=-2.5993
2025-05-18 00:24:25,475 INFO ADF test inflation: stat=-3.8440, p=2.4878e-03, crit(1%)=-3.5746, crit(5%)=-2.9240, crit(10%)=-2.6000
2025-05-18 00:24:25,480 INFO ADF test d_vix: stat=-10.0877, p=1.1436e-17, crit(1%)=-3.5685, crit(5%)=-2.9214, crit(10%)=-2.5987
2025-05-18 00:24:25,520 INFO Optimal lags by AIC/BIC/FPE: {'aic': 1, 'bic': 1, 'fpe': None}
2025-05-18 00:24:25,524 INFO                             OLS Regression Results
2025-05-18 00:24:25,524 INFO                              Dep. Variable: ndx_ret   R-squared: 0.511
2025-05-18 00:24:25,524 INFO                              Adj. R-squared: 0.468
2025-05-18 00:24:25,524 INFO                              Method:  Least Squares   F-statistic: 23.52
2025-05-18 00:24:25,524 INFO                              Prob (F-statistic): 1.24e-10
2025-05-18 00:24:25,524 INFO                              Date:  Sun, 18 May 2025   Time: 00:24:25
2025-05-18 00:24:25,524 INFO                              No. Observations: 51   AIC: -159.9
2025-05-18 00:24:25,524 INFO                              Df Residuals: 46   BIC: -150.3
2025-05-18 00:24:25,524 INFO                              Df Model: 4
2025-05-18 00:24:25,528 INFO SSR=0.106669, FPE=2.546244e-03
2025-05-18 00:24:25,531 INFO Durbin–Watson=2.3569, BG p=0.1894, White p=7.4613509693e-01
2025-05-18 00:24:25,533 INFO Jarque–Bera stat=0.4396, p=0.8027
2025-05-18 00:24:25,534 INFO CUSUM test stat=0.6536, p=0.7865
```

