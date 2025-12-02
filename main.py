import os, math, traceback, argparse, warnings
warnings.filterwarnings("ignore", category=UserWarning)

# User settings 

DATA_DIR        = os.environ.get("DATA_DIR", "./data")
RESULTS_DIR     = os.environ.get("RESULTS_DIR", "./results")

# Strategy params
SHORT           = 5
LONG            = 20
FEE             = 0.0005
TOP_N_PLOT      = 6
INITIAL_CASH    = 10000.0

# Walk-forward params
REBALANCE               = "M"   # monthly
MIN_HISTORY_DAYS        = 10
WF_TOP_COINS            = 5
WF_SPLITS               = 5

# LR params
LR_WINDOW               = 10
LR_TRAIN_FRAC           = 0.7
LR_MIN_SAMPLES          = 50


# Imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# arch for GARCH
try:
    from arch import arch_model
    HAVE_ARCH = True
except Exception:
    HAVE_ARCH = False

# sklearn for Linear Regression
try:
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    HAVE_SKLEARN = True
except Exception:
    HAVE_SKLEARN = False



# I/O utilities

def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

def combine_csvs_to_df(data_path=DATA_DIR):
    csv_files = [f for f in os.listdir(data_path) if f.lower().endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_path}. Put your CSVs there.")
    dfs = []
    for file in csv_files:
        coin_label = file.replace("coin_", "").replace(".csv", "")
        p = os.path.join(data_path, file)
        try:
            tmp = pd.read_csv(p)
            tmp["Coin"] = coin_label
            dfs.append(tmp)
        except Exception as e:
            print(f"Failed to read {p}: {e}")
    if not dfs:
        raise RuntimeError("No CSVs could be read.")
    df_all = pd.concat(dfs, ignore_index=True, sort=False)
    return df_all

def prepare_df_all(df):
    # Add a proper Date column and sort
    date_candidates = [c for c in df.columns if "date" in c.lower()]
    if date_candidates:
        df["Date"] = pd.to_datetime(df[date_candidates[0]], errors="coerce")
    elif "timestamp" in df.columns:
        df["Date"] = pd.to_datetime(df["timestamp"], errors="coerce")
    else:
        raise KeyError("No date-like column found in dataset.")
    df = df.dropna(subset=["Date"]).sort_values(["Coin", "Date"]).reset_index(drop=True)
    return df



# Core helpers / metrics

def detect_price_col(df):
    candidates = ["Close","close","close_price","Close Price","Price","price","Adj Close","adj close","Last","last"]
    for c in candidates:
        if c in df.columns:
            return c
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric:
        return numeric[0]
    raise KeyError("No numeric price-like column found.")

def backtest_minimal(df, signal_col="signal", price_col=None, initial_cash=INITIAL_CASH, fee=FEE):
    if price_col is None:
        price_col = detect_price_col(df)
    cash = float(initial_cash)
    position = 0.0
    trades = []
    equity = []

    for _, row in df.iterrows():
        price = row[price_col]
        if price is None or (isinstance(price, float) and np.isnan(price)):
            equity.append(cash)
            continue
        sig = row.get(signal_col, 0)

        if sig == 1 and cash > price * 1e-9:
            qty = cash / (price * (1 + fee))
            cost = qty * price * (1 + fee)
            if cost > 0 and not np.isnan(cost):
                cash -= cost
                position += qty
                trades.append(("buy", float(price), float(qty)))

        elif sig == -1 and position > 0:
            proceeds = position * price * (1 - fee)
            cash += proceeds
            trades.append(("sell", float(price), float(position)))
            position = 0.0

        equity.append(cash + position * price)

    if len(equity) == 0:
        return {"final_value": initial_cash, "trades": trades, "equity": pd.Series(dtype=float)}
    equity_series = pd.Series(equity, index=df.index)
    return {"final_value": float(equity[-1]), "trades": trades, "equity": equity_series}

def risk_report(equity, periods_per_year=252.0):
    equity = pd.Series(equity).dropna()
    if equity.empty:
        return {"CAGR": np.nan, "Sharpe": np.nan, "Sortino": np.nan, "MaxDD": np.nan, "Calmar": np.nan}
    returns = equity.pct_change().dropna()
    if returns.empty:
        return {"CAGR": np.nan, "Sharpe": np.nan, "Sortino": np.nan, "MaxDD": np.nan, "Calmar": np.nan}

    try:
        days = (equity.index[-1] - equity.index[0]).days
        years = max(days / 365.25, 1.0/periods_per_year)
    except Exception:
        years = max(len(equity) / periods_per_year, 1.0/periods_per_year)

    cagr     = (equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1
    ann_mean = returns.mean() * periods_per_year
    ann_vol  = returns.std() * math.sqrt(periods_per_year)
    downside = returns[returns < 0].std() * math.sqrt(periods_per_year)

    sharpe  = ann_mean / (ann_vol + 1e-12)
    sortino = ann_mean / (downside + 1e-12)

    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0
    maxdd = dd.min()

    calmar = cagr / (abs(maxdd) + 1e-12)
    return {"CAGR": cagr, "Sharpe": sharpe, "Sortino": sortino, "MaxDD": float(maxdd), "Calmar": calmar}



# Volatility & GARCH forecasting

def compute_volatility(df):
    price_col = detect_price_col(df)
    df = df.copy().sort_values("Date")
    df["returns"] = df[price_col].pct_change()
    df["vol_7"]  = df["returns"].rolling(7).std()  * np.sqrt(365)
    df["vol_30"] = df["returns"].rolling(30).std() * np.sqrt(365)
    return df

def garch_forecast(df, horizon=5):
    if not HAVE_ARCH:
        raise RuntimeError("arch not available; install with `pip install arch`.")
    price_col = detect_price_col(df)
    returns = df.sort_values("Date")[price_col].pct_change().dropna() * 100
    model = arch_model(returns, vol="Garch", p=1, q=1)
    res = model.fit(update_freq=10, disp="off")
    forecast = res.forecast(horizon=horizon)
    vol_forecast = np.sqrt(forecast.variance.values[-1]) / 100
    return vol_forecast

def run_volatility_all(df_all, garch_horizon=5):
    coins = df_all["Coin"].unique()
    results = []
    for c in coins:
        dfc = df_all[df_all["Coin"] == c].copy()
        if len(dfc) < 100:
            continue
        dfc = compute_volatility(dfc)
        try:
            garch_vol = garch_forecast(dfc, horizon=garch_horizon)
            results.append({"Coin": c, "Vol_Forecast_D1": garch_vol[0], "Vol_Forecast_D5": garch_vol[-1]})
        except Exception:
            results.append({"Coin": c, "Vol_Forecast_D1": np.nan, "Vol_Forecast_D5": np.nan})
    return pd.DataFrame(results)



# Moving Average (MA) backtester

def run_ma_for_coin(df_coin, short=SHORT, long=LONG, fee=FEE, initial_cash=INITIAL_CASH):
    dfc = df_coin.copy().sort_values("Date").reset_index(drop=True)
    try:
        price_col = detect_price_col(dfc)
    except Exception:
        return {"final_value": np.nan, "trades": 0, "sharpe": np.nan, "equity": pd.Series(dtype=float)}
    dfc["short_ma"] = dfc[price_col].rolling(short).mean()
    dfc["long_ma"]  = dfc[price_col].rolling(long).mean()
    dfc["signal"] = 0
    dfc.loc[dfc["short_ma"] > dfc["long_ma"], "signal"] = 1
    dfc.loc[dfc["short_ma"] < dfc["long_ma"], "signal"] = -1
    dfc = dfc.dropna(subset=["short_ma","long_ma"]).copy()
    if dfc.empty:
        return {"final_value": np.nan, "trades": 0, "sharpe": np.nan, "equity": pd.Series(dtype=float)}

    res = backtest_minimal(dfc, signal_col="signal", price_col=price_col, initial_cash=initial_cash, fee=fee)
    eq = res["equity"]
    sharpe = np.nan
    if len(eq) > 1:
        r = eq.pct_change().dropna()
        ann = 252.0
        sharpe = (r.mean() * ann) / (r.std() * math.sqrt(ann) + 1e-12)
    return {"final_value": res["final_value"], "trades": len(res["trades"]), "sharpe": sharpe, "equity": eq}

def batch_run_ma_all(df_all, short=SHORT, long=LONG, fee=FEE):
    coins = sorted(df_all["Coin"].unique().tolist())
    results, equity_curves, errors = [], {}, []
    print(f"Running MA({short},{long}) across {len(coins)} coins...")
    for coin in tqdm(coins):
        out_rec = {"Coin": coin, "FinalValue": np.nan, "Trades": 0, "Sharpe": np.nan}
        try:
            dfc = df_all[df_all["Coin"] == coin].copy()
            if dfc.empty:
                results.append(out_rec); continue
            out = run_ma_for_coin(dfc, short=short, long=long, fee=fee)
            out_rec["FinalValue"] = float(out.get("final_value", np.nan))
            out_rec["Trades"]     = int(out.get("trades", 0))
            out_rec["Sharpe"]     = float(out.get("sharpe", np.nan))
            eq = out.get("equity")
            if eq is not None and hasattr(eq, "empty") and not eq.empty:
                equity_curves[coin] = eq
        except Exception as e:
            errors.append({"coin": coin, "error": str(e), "trace": traceback.format_exc()})
        results.append(out_rec)

    results_df = pd.DataFrame(results).sort_values("FinalValue", ascending=False, na_position="last").reset_index(drop=True)
    results_df.index += 1
    return results_df, equity_curves, errors



# Linear Regression signals + batch & merge

def make_rolling_features(prices, window):
    p = np.asarray(prices, dtype=float)
    if len(p) <= window:
        return np.empty((0, window)), np.empty((0,))
    X, y = [], []
    for i in range(len(p) - window):
        X.append(p[i:i+window])
        y.append(p[i+window])
    return np.array(X), np.array(y)

def run_lr_for_coin(df_coin, window=LR_WINDOW, train_frac=LR_TRAIN_FRAC, fee=FEE, min_samples=LR_MIN_SAMPLES):
    if not HAVE_SKLEARN:
        return {"final_value": np.nan, "trades": 0, "sharpe": np.nan, "equity": pd.Series(dtype=float)}
    dfc = df_coin.copy().sort_values("Date").reset_index(drop=True)
    try:
        price_col = detect_price_col(dfc)
    except Exception:
        return {"final_value": np.nan, "trades": 0, "sharpe": np.nan, "equity": pd.Series(dtype=float)}
    prices = dfc[price_col].values
    X, y = make_rolling_features(prices, window)
    if X.shape[0] < min_samples:
        return {"final_value": np.nan, "trades": 0, "sharpe": np.nan, "equity": pd.Series(dtype=float)}
    n = X.shape[0]
    split = max(int(n * train_frac), window)
    X_train, y_train = X[:split], y[:split]
    X_test,  y_test  = X[split:], y[split:]
    if len(X_test) < 1:
        return {"final_value": np.nan, "trades": 0, "sharpe": np.nan, "equity": pd.Series(dtype=float)}

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_s, y_train)
    preds = model.predict(X_test_s)

    test_start_idx = split
    idxs = np.arange(test_start_idx + LR_WINDOW, test_start_idx + LR_WINDOW + len(preds))
    df_test = dfc.iloc[idxs].copy().reset_index(drop=True)

    df_test["Predicted_Price_LR"] = preds
    df_test["signal_lr"] = 0
    df_test.loc[df_test["Predicted_Price_LR"] > df_test[price_col], "signal_lr"] = 1
    df_test.loc[df_test["Predicted_Price_LR"] < df_test[price_col], "signal_lr"] = -1

    try:
        res = backtest_minimal(df_test, signal_col="signal_lr", price_col=price_col, initial_cash=INITIAL_CASH, fee=fee)
    except Exception:
        return {"final_value": np.nan, "trades": 0, "sharpe": np.nan, "equity": pd.Series(dtype=float)}
    eq = res["equity"]
    sharpe = np.nan
    if len(eq) > 1:
        r = eq.pct_change().dropna()
        ann = 252.0
        sharpe = (r.mean() * ann) / (r.std() * math.sqrt(ann) + 1e-12)
    return {"final_value": res["final_value"], "trades": len(res["trades"]), "sharpe": sharpe, "equity": eq}

def batch_run_lr_all(df_all, coins=None, window=LR_WINDOW, train_frac=LR_TRAIN_FRAC, fee=FEE, min_samples=LR_MIN_SAMPLES):
    coins = sorted(df_all["Coin"].unique().tolist()) if coins is None else coins
    lr_results = []
    lr_equity_curves = {}
    for coin in tqdm(coins, desc="LR batch"):
        try:
            dfc = df_all[df_all["Coin"] == coin].copy()
            if dfc.empty:
                lr_results.append({"Coin": coin, "FinalValue_LR": np.nan, "Trades_LR": 0, "Sharpe_LR": np.nan})
                continue
            out = run_lr_for_coin(dfc, window=window, train_frac=train_frac, fee=fee, min_samples=min_samples)
            lr_results.append({"Coin": coin,
                               "FinalValue_LR": out.get("final_value", np.nan),
                               "Trades_LR": out.get("trades", 0),
                               "Sharpe_LR": out.get("sharpe", np.nan)})
            if out.get("equity") is not None and hasattr(out["equity"], "empty") and not out["equity"].empty:
                lr_equity_curves[coin] = out["equity"]
        except Exception as e:
            print(f"LR error for {coin}: {e}")
            lr_results.append({"Coin": coin, "FinalValue_LR": np.nan, "Trades_LR": 0, "Sharpe_LR": np.nan})

    results_lr_df = pd.DataFrame(lr_results).sort_values("FinalValue_LR", ascending=False).reset_index(drop=True)
    results_lr_df.index += 1
    out_csv = os.path.join(RESULTS_DIR, "results_lr_all_coins.csv")
    results_lr_df.to_csv(out_csv, index=True)
    print("Saved LR results to", out_csv)
    return results_lr_df, lr_equity_curves

def compare_ma_and_lr(results_ma_df, results_lr_df, out_csv=os.path.join(RESULTS_DIR, "results_compare.csv")):
    ma = results_ma_df.copy()
    if "Coin" not in ma.columns and "coin" in ma.columns:
        ma.rename(columns={"coin":"Coin"}, inplace=True)
    lr = results_lr_df.copy()
    if "Coin" not in lr.columns and "coin" in lr.columns:
        lr.rename(columns={"coin":"Coin"}, inplace=True)

    merged = pd.merge(ma, lr, on="Coin", how="outer")
    merged["Pct_Change_FinalValue_LR_vs_MA"] = np.nan
    for i, row in merged.iterrows():
        mv = row.get("FinalValue", np.nan)
        lv = row.get("FinalValue_LR", np.nan)
        try:
            if not (np.isnan(mv) or np.isnan(lv)) and mv != 0:
                merged.at[i, "Pct_Change_FinalValue_LR_vs_MA"] = (lv - mv) / abs(mv)
        except Exception:
            merged.at[i, "Pct_Change_FinalValue_LR_vs_MA"] = np.nan

    merged.to_csv(out_csv, index=False)
    print("Saved MA vs LR comparison to", out_csv)
    return merged



# Plotting conveniences

def plot_top_equity_curves(results_df, equity_curves, top_n=TOP_N_PLOT, savepath=os.path.join(RESULTS_DIR, "top_equity_curves_fixed.png")):
    top_coins = results_df["Coin"].dropna().tolist()[:top_n]
    plt.figure(figsize=(12,6))
    plotted = 0
    for coin in top_coins:
        if coin in equity_curves and not equity_curves[coin].empty:
            series = equity_curves[coin].dropna()
            norm = series / series.iloc[0]
            plt.plot(norm.index, norm.values, label=coin)
            plotted += 1
    if plotted == 0:
        print("No valid equity curves to plot.")
        return
    plt.legend()
    plt.title("Top Coins Equity Curves (Normalized)")
    plt.xlabel("Date"); plt.ylabel("Normalized equity")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(savepath, dpi=200)
    print("Saved to:", savepath)
    # show on screen if running interactively
    try:
        plt.show()
    except Exception:
        pass



# Walk-forward & portfolio

def walk_forward_ma(df_coin, price_col=None, short=SHORT, long=LONG, fee=FEE, n_splits=5):
    dfc = df_coin.copy().sort_values("Date").reset_index(drop=True)
    if price_col is None:
        price_col = detect_price_col(dfc)
    n = len(dfc)
    if n < 200:
        return None
    fold = n // (n_splits + 1)
    eqs = []
    for k in range(n_splits):
        train_end = (k+1)*fold
        test_end = min((k+2)*fold, n)
        test = dfc.iloc[train_end:test_end].copy()
        if test.empty:
            continue
        test["short_ma"] = test[price_col].rolling(short).mean()
        test["long_ma"]  = test[price_col].rolling(long).mean()
        test = test.dropna(subset=["short_ma","long_ma"])
        if test.empty:
            continue
        test["signal"] = 0
        test.loc[test["short_ma"]>test["long_ma"],"signal"] = 1
        test.loc[test["short_ma"]<test["long_ma"],"signal"] = -1
        res = backtest_minimal(test, signal_col="signal", price_col=price_col, initial_cash=INITIAL_CASH, fee=fee)
        eqs.append(res["equity"])
    if not eqs:
        return None
    return pd.concat(eqs)

def build_equal_weight_portfolio_fixed(df_all, coins_list, price_col=None, rebalance=REBALANCE,
                                       fee=FEE, initial_cash=INITIAL_CASH, min_history_days=MIN_HISTORY_DAYS,
                                       debug_first_rebalances=6):
    if price_col is None:
        price_col = detect_price_col(df_all)

    dfp = (df_all[df_all["Coin"].isin(coins_list)]
           .pivot_table(index="Date", columns="Coin", values=price_col)
           .sort_index())

    dfp.index = pd.to_datetime(dfp.index)
    dfp = dfp.ffill()

    valid_cols = [c for c in dfp.columns if dfp[c].dropna().shape[0] >= min_history_days]
    dfp = dfp[valid_cols].dropna(how="all")
    if dfp.empty or len(valid_cols) == 0:
        raise ValueError("No overlapping data with sufficient history across selected coins.")

    rebal_dt_index    = dfp.resample(rebalance).last().index
    rebal_dates_norm  = set(pd.Timestamp(d).normalize() for d in rebal_dt_index)

    cash = float(initial_cash)
    holdings = pd.Series(0.0, index=dfp.columns)
    equity = []
    last_px = None
    rebalance_count = 0

    for dt, row in dfp.iterrows():
        px = row
        cur_vals = (holdings * px).fillna(0.0)
        port_val = cash + cur_vals.sum()

        if pd.Timestamp(dt).normalize() in rebal_dates_norm:
            rebalance_count += 1
            if last_px is not None:
                sell_price = last_px.reindex_like(holdings).fillna(px)
                cash += (holdings * sell_price * (1 - fee)).sum()
                holdings[:] = 0.0

            available = px.dropna().index.tolist()
            if len(available) > 0:
                weights    = pd.Series(1.0 / len(available), index=available)
                target_val = cash * weights
                qty = pd.Series(0.0, index=dfp.columns)
                for c in available:
                    price_c = px[c]
                    if price_c is None or price_c <= 0 or np.isnan(price_c):
                        qty[c] = 0.0
                    else:
                        qty[c] = target_val[c] / (price_c * (1 + fee))
                spend = (qty * px * (1 + fee)).replace([np.nan, np.inf], 0.0).sum()
                if spend > cash and spend > 0:
                    scale = cash / spend
                    qty *= scale
                    spend = (qty * px * (1 + fee)).replace([np.nan, np.inf], 0.0).sum()
                cash -= spend
                holdings = holdings.add(qty, fill_value=0.0)

            if rebalance_count <= debug_first_rebalances:
                print(f"Rebalance #{rebalance_count} on {pd.Timestamp(dt).date()} - available: {available}")

        equity.append(port_val)
        last_px = px

    return pd.Series(equity, index=dfp.index)



# Main runner

def main():
    parser = argparse.ArgumentParser(description="Crypto research pipeline (VS Code)")
    parser.add_argument("--skip-garch", action="store_true", help="Skip GARCH forecasting step")
    parser.add_argument("--coins-limit", type=int, default=None, help="Limit number of coins processed")
    args = parser.parse_args()

    ensure_dirs()
    print(f"DATA_DIR  = {DATA_DIR}")
    print(f"RESULTS_DIR = {RESULTS_DIR}")

    # 1) Load & prepare
    df_all = combine_csvs_to_df(DATA_DIR)
    df_all = prepare_df_all(df_all)
    print("Combined shape:", df_all.shape)
    print("Columns:", df_all.columns.tolist())

    # 2) Volatility + (optional) GARCH
    try:
        price_col = detect_price_col(df_all)
        # pick first coin for sample volatility plot
        sample_coin = df_all["Coin"].unique()[0]
        dfc = df_all[df_all["Coin"] == sample_coin].copy()
        dfc = compute_volatility(dfc)
        # save a small sample
        dfc.head(20).to_csv(os.path.join(RESULTS_DIR, f"sample_vol_{sample_coin}.csv"), index=False)
        print(f"Saved sample volatility rows for {sample_coin}")
    except Exception as e:
        print("Volatility precheck error:", e)

    if not args.skip_garch:
        try:
            vol_results = run_volatility_all(df_all)
            vol_path = os.path.join(RESULTS_DIR, "volatility_forecasts.csv")
            vol_results.to_csv(vol_path, index=False)
            print("Saved:", vol_path)
        except Exception as e:
            print("GARCH step failed/skipped:", e)

    # 3) MA batch
    results_df, equity_curves, errors = batch_run_ma_all(df_all, short=SHORT, long=LONG, fee=FEE)
    out_csv = os.path.join(RESULTS_DIR, "results_all_coins.csv")
    results_df.to_csv(out_csv, index=True)
    print("Saved MA results:", out_csv)
    if errors:
        err_csv = os.path.join(RESULTS_DIR, "errors_ma.json")
        pd.DataFrame(errors).to_json(err_csv, orient="records", indent=2)
        print(f"Encountered errors for {len(errors)} coins. Saved to {err_csv}")

    # 4) Plot top equity curves
    plot_top_equity_curves(results_df, equity_curves, top_n=TOP_N_PLOT)

    # 5) LR batch + compare
    results_lr_df, lr_equity_curves = batch_run_lr_all(df_all, window=LR_WINDOW, train_frac=LR_TRAIN_FRAC)
    merged = compare_ma_and_lr(results_df, results_lr_df)
    merged.to_csv(os.path.join(RESULTS_DIR, "results_compare.csv"), index=False)

    # 6) Walk-forward on top coins
    top_coins = [c for c in results_df["Coin"].dropna().tolist()][:WF_TOP_COINS]
    wf_folder = os.path.join(RESULTS_DIR, "walkforward")
    os.makedirs(wf_folder, exist_ok=True)
    wf_rows = []
    for coin in top_coins:
        print(f"\n-- Walk-forward for {coin} --")
        coin_df = df_all[df_all["Coin"] == coin].copy()
        if coin_df.empty or len(coin_df) < 50:
            print(f"Skipping {coin}: not enough data ({len(coin_df)} rows).")
            continue
        wf_eq = walk_forward_ma(coin_df, short=SHORT, long=LONG, fee=FEE, n_splits=WF_SPLITS)
        if wf_eq is None or len(wf_eq) < 2:
            print(f"No valid walk-forward equity for {coin}.")
            continue

        metrics = risk_report(wf_eq)
        wf_rows.append({
            "Coin": coin,
            "WF_points": len(wf_eq),
            "CAGR": metrics.get("CAGR"),
            "Sharpe": metrics.get("Sharpe"),
            "Sortino": metrics.get("Sortino"),
            "MaxDD": metrics.get("MaxDD"),
            "Calmar": metrics.get("Calmar")
        })

        # save equity & plot
        safe = coin.replace(" ", "_")
        csv_path = os.path.join(wf_folder, f"walkforward_{safe}.csv")
        wf_eq.to_frame(name="equity").to_csv(csv_path)
        fig = plt.figure(figsize=(10,4))
        plt.plot(wf_eq.index, wf_eq.values)
        plt.title(f"{coin} — Walk-forward equity")
        plt.grid(True)
        img_path = os.path.join(wf_folder, f"walkforward_{safe}.png")
        fig.savefig(img_path, dpi=160)
        plt.close(fig)
        print(f"Saved walk-forward CSV: {csv_path}")
        print(f"Saved walk-forward PNG: {img_path}")

    if wf_rows:
        wf_summary = pd.DataFrame(wf_rows).sort_values("Sharpe", ascending=False).reset_index(drop=True)
        wf_summary.to_csv(os.path.join(wf_folder, "walkforward_summary.csv"), index=False)
        print("Walk-forward summary saved.")

    # 7) Equal-weight monthly rebalanced portfolio on top 5
    top5 = results_df["Coin"].head(5).tolist() if not results_df.empty else []
    if top5:
        print("Building equal-weight portfolio for:", top5)
        try:
            portfolio_eq = build_equal_weight_portfolio_fixed(
                df_all, top5, price_col=detect_price_col(df_all),
                rebalance=REBALANCE, fee=FEE, initial_cash=INITIAL_CASH,
                min_history_days=MIN_HISTORY_DAYS, debug_first_rebalances=6
            )
            metrics = risk_report(portfolio_eq)
            print("Portfolio metrics:", metrics)

            # Save CSV
            csv_path = os.path.join(RESULTS_DIR, "portfolio_eq_fixed.csv")
            portfolio_eq.to_csv(csv_path)
            print("Saved portfolio CSV:", csv_path)

            # Plot absolute / normalized
            plt.figure(figsize=(10,4))
            plt.plot(portfolio_eq.index, portfolio_eq.values)
            plt.title("Equal-weight monthly rebalanced portfolio (abs)")
            plt.grid(True)
            png_abs = os.path.join(RESULTS_DIR, "portfolio_eq_fixed.png")
            plt.savefig(png_abs, dpi=160); plt.close()
            print("Saved:", png_abs)

            plt.figure(figsize=(10,4))
            plt.plot(portfolio_eq.index, (portfolio_eq / portfolio_eq.iloc[0]).values)
            plt.title("Portfolio equity (normalized)")
            plt.grid(True)
            png_norm = os.path.join(RESULTS_DIR, "portfolio_eq_fixed_normalized.png")
            plt.savefig(png_norm, dpi=160); plt.close()
            print("Saved:", png_norm)

            # Drawdown quick stats
            roll_max = portfolio_eq.cummax()
            dd = portfolio_eq/roll_max - 1.0
            dd.to_csv(os.path.join(RESULTS_DIR, "portfolio_drawdown_series.csv"))
            print("Max drawdown:", dd.min())

        except Exception as e:
            print("Portfolio build error:", e)
    else:
        print("No top coins available; skipped portfolio step.")

    # 8) Final ranked results save
    results_df.to_csv(os.path.join(RESULTS_DIR, "results_all_coins_ranked.csv"), index=True)
    print("\nSummary:")
    print(" - n_coins:", df_all["Coin"].nunique())
    print(" - top_coins:", results_df["Coin"].head(10).tolist())


if __name__ == "__main__":
    main()
