# Crypto Strategy & Volatility Analysis System

A complete end-to-end machine learning and quantitative trading project that analyzes cryptocurrency price data, forecasts volatility using GARCH, and tests multiple strategies (Moving Average, Linear Regression, Portfolio Optimization). Includes both a CLI pipeline (main.py) and an optional Streamlit dashboard for interactive visualization.


# Project Structure
crypto-project/

├── main.py                # Full automatic pipeline (data → results)

├── streamlit_app.py       # Interactive Streamlit dashboard

├── requirements.txt       # All dependencies

├── data/                  # Input folder (coin_*.csv files)

├── results/               # Output folder (plots & CSV results)

├── README.md              # This file

# Requirements
	•	Python 3.11.x
	•	Works on Windows, macOS, or Linux
	•	Recommended: VS Code or PyCharm


# Installation

1) Create a virtual environment
python -m venv venv

2) Activate it
	•	Windows:
    venv\Scripts\activate
	•	macOS / Linux:
    source venv/bin/activate

3) Install dependencies
pip install -r requirements.txt

# How to Run the Full Pipeline

Run this to execute everything automatically:
python main.py

It will:

	•	Load all CSV files from /data
	•	Clean & combine them
	•	Detect price columns automatically
	•	Compute volatility & GARCH forecasts
	•	Run Moving Average strategy
	•	Run Linear Regression strategy
	•	Run portfolio builder
	•	Save:
	•	results_all_coins.csv
	•	volatility_forecasts.csv
	•	results_lr_all_coins.csv
	•	walkforward_summary.csv
	•	portfolio_eq_fixed.png
	•	top_equity_curves_fixed.png

All outputs are saved in the /results folder.


# Running the Streamlit App

If you want to visualize and interact with the strategies:

python -m streamlit run streamlit_app.py

Then open:

http://localhost:8501

Upload a CSV → choose your strategy → adjust sliders → run → see live results and charts.

# Required CSV Format

Each file should be named like:

coin_BTC.csv

coin_ETH.csv

and contain at least:

Date               Close
2022-01-01         47000
2022-01-02         47200
…                  …

# Key Features:


 Automated data merging for multiple coins
 Volatility estimation using GARCH(1,1) model
 Two strategies:
 
	•	Moving Average (MA)
	•	Linear Regression (LR)
	
 Walk-forward validation
 Portfolio optimization
 Streamlit dashboard for exploration
 Result visualization with equity curves



# Technologies Used
	•	Python 3.11
	•	Pandas, NumPy, Matplotlib
	•	scikit-learn
	•	ARCH (for GARCH models)
	•	Streamlit
	•	TQDM for progress tracking

# Output Files Explained:
| File     | Description| 
|-------|----------|
| volatility_forecasts.csv    | GARCH volatility forecasts (1-day, 5-day horizon) | 
| results_all_coins.csv   |  MA strategy results for all coins |
| results_lr_all_coins.csv  |  Linear Regression results | 
| walkforward_summary.csv |  Walk-forward validation output |
| portfolio_eq_fixed.png  |  Final portfolio equity curve | 
| top_equity_curves_fixed.png   |  Top-performing coins’ equity curves|


# How the Linear Regression Model Works in This Project

# Objective

The Linear Regression (LR) strategy predicts the next-day closing price of each cryptocurrency based on recent price patterns.
It acts as a supervised learning model that identifies trends in historical data to generate trading signals.



# Step-by-Step Workflow

1) Data Preparation:
   
	•	For each coin’s dataset, the system uses historical Close prices.
	•	The model creates lag features (previous few days’ prices) to capture short-term patterns.

Example:

|Date   |  Close |  Lag1  |  Lag2  | Lag3 |  
|-------|----------|-------|----------|----------|
| 2022-01-04   | 48000  | 47800    |   47600     | 47400 |


These features allow the model to learn how the last few days influence tomorrow’s price.


2) Model Training:
   
	•	Uses sklearn.linear_model.LinearRegression

	•	Learns the equation:

        y^​t​ = β0 ​+ β1​⋅Lag1 ​+ β2​⋅Lag2 ​+ ⋯ + βn​⋅Lagn​

where:

	•	y^​t​ = predicted price at time t
	•	βi = coefficient for each lag feature


3) Prediction & Signal Generation:
   
	•	The trained model predicts the next-day close price.

	•	It generates trading signals:

	•	If predicted price > current price → Buy

	•	If predicted price < current price → Sell


    This emulates how a trader reacts to expected price movement.


4) Backtesting:
   
	•	The model is tested on unseen (future) data using walk-forward validation, which simulates real trading conditions.

	•	Each step retrains the model using the most recent data.


5) Performance Metrics:
   
The system records:

	•	Prediction accuracy
	•	Strategy returns
	•	Equity curves
	•	Walk-forward results

Results saved as:

	•	results_lr_all_coins.csv
	•	walkforward_summary.csv



# Example Visualization

Generated graphs include:

	•	Predicted vs Actual Prices
	•	Equity Curve (Profit/Loss over time)
