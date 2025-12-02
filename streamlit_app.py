import streamlit as st
import pandas as pd
import numpy as np

st.title("Crypto Strategy Explorer")

uploaded = st.file_uploader("Upload a CSV file with Date + Close columns", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)

    st.subheader("Raw Data Preview")
    st.dataframe(df.head())

    # Detect price column
    price_col = None
    for col in ["Close", "close", "Price", "price"]:
        if col in df.columns:
            price_col = col
            break
    if not price_col:
        st.error("Could not find a price column.")
    else:
        st.line_chart(df[price_col])

    # MA settings
    short = st.slider("Short MA", 2, 50, 5)
    long = st.slider("Long MA", 10, 200, 20)
    fee  = st.number_input("Fee", 0.0, 0.01, 0.0005)

    if st.button("Run MA Strategy"):
        df["short_ma"] = df[price_col].rolling(short).mean()
        df["long_ma"]  = df[price_col].rolling(long).mean()

        df["signal"] = np.where(df["short_ma"] > df["long_ma"], 1, -1)

        # Backtest
        cash = 10000
        position = 0
        equity = []

        for i, row in df.iterrows():
            price = row[price_col]
            sig = row["signal"]

            if sig == 1 and cash > 0:
                qty = cash / (price * (1 + fee))
                cash -= qty * price * (1 + fee)
                position += qty
            elif sig == -1 and position > 0:
                cash += position * price * (1 - fee)
                position = 0

            equity.append(cash + position * price)

        st.subheader("Final Value")
        st.write(f"${equity[-1]:.2f}")

        st.subheader("Equity Curve")
        st.line_chart(equity)
