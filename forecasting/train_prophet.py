import argparse
import os
import glob
import json
import pickle
from datetime import datetime, timedelta, date
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train Prophet models on generated CSVs")
    parser.add_argument("--mode", choices=["category", "product"], default="category",
                        help="Forecast by category or by product within each category")
    parser.add_argument("--data-dir", default=".", help="Directory containing generated CSV files")
    parser.add_argument("--horizon", type=int, default=30, help="Forecast horizon in days")
    parser.add_argument("--output-dir", default="forecasting/output", help="Directory to save outputs")
    parser.add_argument("--grid", action="store_true", help="Run small hyperparameter grid search")
    parser.add_argument("--include-groups", default=None,
                        help="Optional comma-separated list of group names to train (e.g., 'Makeup,Hair Care')")
    return parser.parse_args()


def load_generated_csvs(data_dir: str) -> pd.DataFrame:
    patterns = [
        os.path.join(data_dir, "beauty_skincare_sales_*.csv"),
        os.path.join(data_dir, "clothing_accessories_sales_*.csv"),
    ]
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    if not files:
        # fallback to any csv in dir
        files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not files:
        raise FileNotFoundError("No CSV files found in data directory.")

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            print(f"Skipping {f}: {e}")
    if not dfs:
        raise RuntimeError("Unable to read any CSV files.")
    df = pd.concat(dfs, ignore_index=True)
    # normalize column names
    df.columns = [c.strip() for c in df.columns]
    return df


def ensure_columns(df: pd.DataFrame):
    """Ensure minimal required columns exist; create sane defaults for optional ones.

    Accepts legacy CSVs that may not include the new fields and fills them.
    """
    minimal_required = ["date", "quantitySold", "category", "productName"]
    missing_min = [c for c in minimal_required if c not in df.columns]
    if missing_min:
        raise ValueError(f"Missing core columns: {missing_min}. The dataset must include date, quantitySold, category, productName.")

    # Parse date
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])  # drop bad dates
    # Clean types
    df["quantitySold"] = pd.to_numeric(df["quantitySold"], errors="coerce")
    df = df.dropna(subset=["quantitySold"])  # drop rows without target

    # Optional columns
    # Map legacy 'seasonality' to 'inSeason' if available
    if "inSeason" not in df.columns and "seasonality" in df.columns:
        df["inSeason"] = pd.to_numeric(df["seasonality"], errors="coerce").fillna(0).astype(int)
    for opt_int in ["inSeason", "promotional"]:
        if opt_int not in df.columns:
            df[opt_int] = 0
        df[opt_int] = pd.to_numeric(df[opt_int], errors="coerce").fillna(0).astype(int)

    if "discountRate" not in df.columns:
        df["discountRate"] = 0.0
    else:
        df["discountRate"] = pd.to_numeric(df["discountRate"], errors="coerce").fillna(0.0)

    if "season" not in df.columns:
        df["season"] = ""

    # Numeric parsing for price if present
    if "unitPrice" in df.columns:
        df["unitPrice"] = pd.to_numeric(df["unitPrice"], errors="coerce")
        df["unitPrice"] = df["unitPrice"].fillna(df["unitPrice"].median() if df["unitPrice"].notna().any() else 0.0)

    return df


def build_sale_event_holidays(start: date, end: date) -> pd.DataFrame:
    # Generate payday (15th and last day of month), and double-digit sale days (m/m)
    rows = []
    cur = date(start.year, start.month, 1)
    while cur <= end:
        last_day = (date(cur.year, cur.month + 1, 1) - timedelta(days=1)) if cur.month < 12 else date(cur.year, 12, 31)
        # payday 15th
        payday15 = date(cur.year, cur.month, 15)
        rows.append({"ds": payday15, "holiday": "payday"})
        # payday last day
        rows.append({"ds": last_day, "holiday": "payday"})
        # double-digit day (month == day)
        if cur.month <= 12:
            dd = date(cur.year, cur.month, cur.month)
            rows.append({"ds": dd, "holiday": "double_day"})
        # next month
        if cur.month == 12:
            cur = date(cur.year + 1, 1, 1)
        else:
            cur = date(cur.year, cur.month + 1, 1)

    # Christmas season anchor with window
    # Prophet supports lower/upper windows via additional columns
    christmas_rows = []
    for y in range(start.year, end.year + 1):
        christmas = date(y, 12, 25)
        christmas_rows.append({"ds": christmas, "holiday": "christmas", "lower_window": -30, "upper_window": 10})

    all_rows = rows + christmas_rows
    holidays_df = pd.DataFrame(all_rows)
    return holidays_df


def add_inseason_regressor(df_daily: pd.DataFrame, category_name: str) -> pd.DataFrame:
    # Deterministic by month and category; mirrors generator logic roughly
    df_daily["month"] = df_daily["ds"].dt.month
    def inseason(month: int) -> int:
        # Extract category if group contains product (format: "Category__Product")
        base_category = category_name.split("__")[0] if "__" in category_name else category_name
        if base_category == "Skincare":
            return 1 if month in [3,4,5,8,9,10] else 0
        if base_category == "Makeup":
            return 1 if month in [11,12,1,2] else 0
        if base_category == "Hair Care":
            return 1 if month in [3,4,5] else 0
        if base_category == "Tops":
            return 1 if month in [3,4,5,6,7] else 0
        if base_category == "Bottoms":
            return 1 if month in [6,7,8,9,10] else 0
        if base_category == "Dresses":
            return 1 if month in [3,4,5,11,12,1] else 0
        if base_category == "Accessories":
            return 1 if month in [3,4,5,11,12,1] else 0
        return 0
    df_daily["inSeasonDeterministic"] = df_daily["month"].apply(inseason)
    df_daily = df_daily.drop(columns=["month"])  # clean up
    return df_daily


def aggregate_series(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    g = df.groupby(group_cols + ["date"], as_index=False)["quantitySold"].sum()
    g = g.rename(columns={"date": "ds", "quantitySold": "y"})
    return g


def prepare_group_daily(df: pd.DataFrame, group_name: str, mode: str) -> pd.DataFrame:
    """Build a daily dataframe with target y and available regressors for a given group."""
    if mode == "category":
        df_g = df[df["category"] == group_name].copy()
    else:
        cat, prod = group_name.split("__", 1)
        df_g = df[(df["category"] == cat) & (df["productName"] == prod)].copy()

    # Derive calendar features if missing
    df_g["date"] = pd.to_datetime(df_g["date"])
    df_g["isWeekend"] = df_g["date"].dt.weekday.isin([5, 6]).astype(int)
    # Payday: 15th and last day of month
    last_day = df_g["date"].dt.to_period('M').dt.to_timestamp('M')
    df_g["isPayday"] = ((df_g["date"].dt.day == 15) | (df_g["date"].dt.date == last_day.dt.date)).astype(int)
    # Double day: month == day
    df_g["isDoubleDay"] = (df_g["date"].dt.month == df_g["date"].dt.day).astype(int)

    # One-hot encode promoType if present
    if "promoType" in df_g.columns:
        dummies = pd.get_dummies(df_g["promoType"], prefix="promoType")
        df_g = pd.concat([df_g.drop(columns=["promoType"]), dummies], axis=1)

    # One-hot encode season label if present
    if "season" in df_g.columns:
        season_dummies = pd.get_dummies(df_g["season"], prefix="season")
        df_g = pd.concat([df_g.drop(columns=["season"]), season_dummies], axis=1)

    # Aggregate target and regressors daily
    agg = {
        "quantitySold": "sum",
        "promotional": "max",
        "discountRate": "mean",
        "unitPrice": "mean",
        "isWeekend": "max",
        "isPayday": "max",
        "isDoubleDay": "max",
    }
    # Include promoType and season dummies if created
    promo_cols = [c for c in df_g.columns if c.startswith("promoType_")]
    season_cols = [c for c in df_g.columns if c.startswith("season_")]
    for c in promo_cols:
        agg[c] = "max"
    for c in season_cols:
        agg[c] = "max"
    cols_present = [c for c in agg.keys() if c in df_g.columns]
    g = df_g.groupby("date", as_index=False).agg({c: agg[c] for c in cols_present})
    g = g.rename(columns={"quantitySold": "y", "date": "ds"})
    # Fill missing columns with defaults
    for c in ["promotional", "discountRate", "unitPrice", "isWeekend", "isPayday", "isDoubleDay"] + promo_cols + season_cols:
        if c not in g.columns:
            g[c] = 0
    # Ensure daily continuity
    fills = {"y": 0, "discountRate": 0}
    if "unitPrice" in g.columns:
        fills["unitPrice"] = g["unitPrice"].median() if g["unitPrice"].notna().any() else 0.0
    g = g.set_index("ds").asfreq("D").fillna(fills).fillna(0).reset_index()
    return g


def fit_prophet(series_df: pd.DataFrame, holidays_df: pd.DataFrame, category_name: str,
                grid_params: Dict = None) -> Prophet:
    # Base model
    params = {
        "weekly_seasonality": True,
        "yearly_seasonality": True,
        "daily_seasonality": False,
        "holidays": holidays_df,
        "holidays_prior_scale": 15.0,
        "seasonality_mode": "additive",
    }
    if grid_params:
        params.update(grid_params)

    m = Prophet(**params)
    # Add deterministic in-season regressor
    m.add_regressor("inSeasonDeterministic", mode="additive", prior_scale=10.0)
    # Season dummies if present
    for c in [c for c in series_df.columns if c.startswith("season_")]:
        m.add_regressor(c, mode="additive")
    m.fit(series_df)
    return m


def make_future_df(m: Prophet, horizon_days: int, series_df: pd.DataFrame) -> pd.DataFrame:
    future = m.make_future_dataframe(periods=horizon_days)
    # Add deterministic regressor for future
    future["month"] = future["ds"].dt.month
    def inseason(month: int) -> int:
        return 0  # placeholder; will be replaced by caller
    # Caller will overwrite the column with actual deterministic values
    future["inSeasonDeterministic"] = 0
    future = future.drop(columns=["month"])
    return future


def run_cv(m: Prophet, horizon_days: int) -> pd.DataFrame:
    try:
        df_cv = cross_validation(m, horizon=f"{horizon_days} days", period=f"{max(7, horizon_days//2)} days")
        df_p = performance_metrics(df_cv)
        return df_p
    except Exception as e:
        print(f"Cross-validation failed: {e}")
        return pd.DataFrame()


def train_for_group(df: pd.DataFrame, group_name: str, horizon: int, output_dir: str,
                    grid: bool, holidays_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    df_group = df[df["group"] == group_name].copy()
    # Ensure daily continuity
    df_group_daily = df_group.set_index("ds").asfreq("D").fillna(0).reset_index()
    # Add deterministic in-season
    df_group_daily = add_inseason_regressor(df_group_daily, group_name)

    # Fit model (optionally grid)
    best_params = None
    best_mape = np.inf
    best_model = None
    param_grid = [
        {"changepoint_prior_scale": cps, "seasonality_prior_scale": sps}
        for cps in [0.05, 0.1, 0.2]
        for sps in [5.0, 10.0, 15.0]
    ] if grid else [{}]

    for params in param_grid:
        m = fit_prophet(df_group_daily, holidays_df, group_name, params if grid else None)
        metrics = run_cv(m, horizon)
        if not metrics.empty:
            mape = metrics["mape"].mean()
        else:
            mape = np.inf
        if mape < best_mape:
            best_mape = mape
            best_model = m
            best_params = params

    # Forecast
    future = best_model.make_future_dataframe(periods=horizon)
    # Add deterministic regressor for future
    future = add_inseason_regressor(future, group_name)
    forecast = best_model.predict(future)

    # Save outputs
    os.makedirs(output_dir, exist_ok=True)
    forecast_out = os.path.join(output_dir, f"forecast_{group_name}.csv")
    forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_csv(forecast_out, index=False)

    model_out = os.path.join(output_dir, f"model_{group_name}.pkl")
    with open(model_out, "wb") as f:
        pickle.dump(best_model, f)

    return forecast, {"group": group_name, "best_mape": best_mape, "best_params": best_params}


def main():
    args = parse_args()
    # Enforce horizon bounds: minimum 7, maximum 30
    args.horizon = max(7, min(30, int(args.horizon)))
    df = load_generated_csvs(args.data_dir)
    df = ensure_columns(df)

    # Aggregate
    if args.mode == "category":
        series = aggregate_series(df, ["category"])  # ds, y, category
        series["group"] = series["category"]
    else:
        series = aggregate_series(df, ["category", "productName"])  # ds, y, category, productName
        series["group"] = series["category"].astype(str) + "__" + series["productName"].astype(str)

    # Build holidays within available data + horizon
    start = series["ds"].min().date()
    end = (series["ds"].max() + pd.Timedelta(days=args.horizon)).date()
    holidays_df = build_sale_event_holidays(start, end)

    # Train each group
    results = []
    forecast_dir = os.path.join(args.output_dir, args.mode)
    os.makedirs(forecast_dir, exist_ok=True)
    groups = sorted(series["group"].unique())
    if args.include_groups:
        requested = [s.strip() for s in args.include_groups.split(",") if s.strip()]
        groups = [g for g in groups if g in requested]
        if not groups:
            raise ValueError(f"No matching groups found for include-groups={requested}")
    for g in groups:
        # Prepare df for Prophet with regressors from original data
        df_daily = prepare_group_daily(df, g, args.mode)
        df_daily = add_inseason_regressor(df_daily, g.split("__")[0] if args.mode == "product" else g)

        # Small hyperparameter grid to improve fit
        grid_params = [
            {
                "changepoint_prior_scale": cps,
                "seasonality_mode": mode,
                "seasonality_prior_scale": sps,
                "holidays_prior_scale": hps,
                "monthly_fourier": mf,
                "n_changepoints": ncp
            }
            for cps in [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
            for mode in ["additive", "multiplicative"]
            for sps in [5.0, 10.0, 15.0, 20.0, 30.0]
            for hps in [5.0, 10.0, 15.0, 20.0, 30.0]
            for mf in [0, 5, 10, 15]
            for ncp in [15, 25, 50]
        ] if args.grid else [{
            "changepoint_prior_scale": 0.1,
            "seasonality_mode": "additive",
            "seasonality_prior_scale": 10.0,
            "holidays_prior_scale": 15.0,
            "monthly_fourier": 0,
            "n_changepoints": 25
        }]
        # If focusing on Makeup with grid, use a smaller, targeted grid to speed up
        if args.grid and g == "Makeup":
            grid_params = [
                {
                    "changepoint_prior_scale": cps,
                    "seasonality_mode": mode,
                    "seasonality_prior_scale": sps,
                    "holidays_prior_scale": hps,
                    "monthly_fourier": mf,
                    "n_changepoints": ncp
                }
                for cps in [0.005, 0.01, 0.05, 0.1]
                for mode in ["additive", "multiplicative"]
                for sps in [20.0, 30.0]
                for hps in [5.0, 10.0]
                for mf in [0, 5, 10]
                for ncp in [25, 50]
            ]

        best_m = None
        best_mape_holdout = np.inf
        best_wmape_holdout = np.inf
        best_params = None
        for gp in grid_params:
            m = Prophet(weekly_seasonality=True, yearly_seasonality=True, daily_seasonality=False,
                        holidays=holidays_df,
                        holidays_prior_scale=gp["holidays_prior_scale"],
                        seasonality_mode=gp["seasonality_mode"],
                        changepoint_prior_scale=gp["changepoint_prior_scale"],
                        seasonality_prior_scale=gp["seasonality_prior_scale"],
                        n_changepoints=int(gp["n_changepoints"]))
            m.add_regressor("inSeasonDeterministic", mode="additive", prior_scale=10.0)
            # Optional monthly seasonality to capture month-by-month effects
            # Optional monthly seasonality to capture month-by-month effects
            if int(gp["monthly_fourier"]) > 0:
                m.add_seasonality(name="monthly", period=30.5, fourier_order=int(gp["monthly_fourier"]))
            # Optional regressors if present
            for reg in ["promotional", "isWeekend", "isPayday", "isDoubleDay"] + [c for c in df_daily.columns if c.startswith("promoType_")] + [c for c in df_daily.columns if c.startswith("season_")]:
                if reg in df_daily.columns:
                    m.add_regressor(reg, mode="additive")
            # Treat unitPrice multiplicatively to capture elasticity
            if "unitPrice" in df_daily.columns:
                m.add_regressor("unitPrice", mode="multiplicative")
            # Treat discountRate multiplicatively to capture lift from discounts
            if "discountRate" in df_daily.columns:
                m.add_regressor("discountRate", mode="multiplicative")
            m.fit(df_daily)

            # Holdout evaluation
            # Use a stable 14-day holdout (or 20% of history, min 7) to avoid volatile metrics
            test_len = max(14, min(args.horizon, max(7, int(len(df_daily) * 0.2))))
            if len(df_daily) > test_len + 30:
                df_train = df_daily.iloc[:-test_len].copy()
                df_test = df_daily.iloc[-test_len:].copy()
                m_bt = Prophet(weekly_seasonality=True, yearly_seasonality=True, daily_seasonality=False,
                               holidays=holidays_df,
                               holidays_prior_scale=gp["holidays_prior_scale"],
                               seasonality_mode=gp["seasonality_mode"],
                               changepoint_prior_scale=gp["changepoint_prior_scale"],
                               seasonality_prior_scale=gp["seasonality_prior_scale"],
                               n_changepoints=int(gp["n_changepoints"]))
                m_bt.add_regressor("inSeasonDeterministic", mode="additive", prior_scale=10.0)
                if int(gp["monthly_fourier"]) > 0:
                    m_bt.add_seasonality(name="monthly", period=30.5, fourier_order=int(gp["monthly_fourier"]))
                # Add base, promoType, and season regressors for backtest model
                for reg in ["promotional", "isWeekend", "isPayday", "isDoubleDay"] + [c for c in df_train.columns if c.startswith("promoType_")] + [c for c in df_train.columns if c.startswith("season_")]:
                    if reg in df_train.columns:
                        m_bt.add_regressor(reg, mode="additive")
                if "unitPrice" in df_train.columns:
                    m_bt.add_regressor("unitPrice", mode="multiplicative")
                if "discountRate" in df_train.columns:
                    m_bt.add_regressor("discountRate", mode="multiplicative")
                m_bt.fit(df_train)
                promo_future_cols = [c for c in df_test.columns if c.startswith("promoType_")]
                season_future_cols = [c for c in df_test.columns if c.startswith("season_")]
                base_future_cols = ["ds", "inSeasonDeterministic", "promotional", "discountRate", "unitPrice", "isWeekend", "isPayday", "isDoubleDay"]
                future_bt = df_test[base_future_cols + promo_future_cols + season_future_cols].copy()
                # Ensure all columns exist
                for c in ["promotional", "discountRate", "unitPrice", "isWeekend", "isPayday", "isDoubleDay"] + promo_future_cols + season_future_cols:
                    if c not in future_bt.columns:
                        future_bt[c] = 0
                fc_bt = m_bt.predict(future_bt)
                df_eval = df_test.merge(fc_bt[["ds", "yhat"]], on="ds", how="left")
                denom = df_eval["y"].replace(0, np.nan)
                mape_vals = (df_eval["yhat"] - df_eval["y"]).abs() / denom
                mape_hold = mape_vals.mean() if mape_vals.notna().any() else np.inf
                wmape_hold = (np.abs(df_eval["yhat"] - df_eval["y"]).sum()) / (df_eval["y"].sum() + 1e-9)
                # Select best model by weighted MAPE (more robust with low-volume days)
                if wmape_hold < best_wmape_holdout:
                    best_wmape_holdout = wmape_hold
                    best_mape_holdout = mape_hold
                    best_m = m
                    best_params = gp
            else:
                # If not enough data, choose first model
                if best_m is None:
                    best_m = m
                    best_params = gp

        m = best_m

        # CV & metrics
        metrics = run_cv(m, args.horizon)
        mape = metrics["mape"].mean() if not metrics.empty else np.nan

        # Holdout backtest if CV failed or insufficient
        if np.isnan(mape):
            # Use last N days as test
            test_len = max(14, min(args.horizon, max(7, int(len(df_daily) * 0.2))))
            if len(df_daily) > test_len + 30:
                df_train = df_daily.iloc[:-test_len].copy()
                df_test = df_daily.iloc[-test_len:].copy()
                m_bt = Prophet(weekly_seasonality=True, yearly_seasonality=True, daily_seasonality=False,
                               holidays=holidays_df, holidays_prior_scale=15.0, seasonality_mode="additive",
                               changepoint_prior_scale=0.1, seasonality_prior_scale=10.0)
                m_bt.add_regressor("inSeasonDeterministic", mode="additive", prior_scale=10.0)
                m_bt.fit(df_train)
                # Predict on test period
                future_bt = df_test[["ds", "inSeasonDeterministic"]].copy()
                fc_bt = m_bt.predict(future_bt)
                df_eval = df_test.merge(fc_bt[["ds", "yhat"]], on="ds", how="left")
                # MAPE avoiding division by zero; fall back to sMAPE
                denom = df_eval["y"].replace(0, np.nan)
                mape_vals = (df_eval["yhat"] - df_eval["y"]).abs() / denom
                if mape_vals.notna().any():
                    mape = mape_vals.mean()
                else:
                    smape = 2.0 * (df_eval["yhat"] - df_eval["y"]).abs() / (df_eval["yhat"].abs() + df_eval["y"].abs() + 1e-9)
                    mape = smape.mean()

        # Forecast
        future = m.make_future_dataframe(periods=args.horizon)
        future = add_inseason_regressor(future,
                                        g.split("__")[0] if args.mode == "product" else g)
        # Carry forward known calendar regressors for future
        future["isWeekend"] = future["ds"].dt.weekday.isin([5, 6]).astype(int)
        # Payday future
        last_day = future["ds"].dt.to_period('M').dt.to_timestamp('M')
        future["isPayday"] = ((future["ds"].dt.day == 15) | (future["ds"].dt.date == last_day.dt.date)).astype(int)
        future["isDoubleDay"] = (future["ds"].dt.month == future["ds"].dt.day).astype(int)
        # Assume no promotions ahead unless provided externally
        if "promotional" in df_daily.columns:
            future["promotional"] = 0
        if "discountRate" in df_daily.columns:
            future["discountRate"] = 0.0
        if "unitPrice" in df_daily.columns:
            # Default future price to historical median (replace with planned pricing if available)
            future["unitPrice"] = df_daily["unitPrice"].median()
        # Ensure promoType and season dummies exist in future if model used them
        promo_future_cols = [c for c in df_daily.columns if c.startswith("promoType_")]
        season_future_cols = [c for c in df_daily.columns if c.startswith("season_")]
        for c in promo_future_cols + season_future_cols:
            future[c] = 0
        forecast = m.predict(future)

        out_f = os.path.join(forecast_dir, f"forecast_{g}.csv")
        forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_csv(out_f, index=False)

        out_m = os.path.join(forecast_dir, f"model_{g}.pkl")
        with open(out_m, "wb") as f:
            pickle.dump(m, f)

        # Accuracy computed from weighted MAPE
        holdout_accuracy = None
        try:
            holdout_accuracy = max(0.0, 1.0 - float(best_wmape_holdout))
        except Exception:
            holdout_accuracy = None
        results.append({
            "group": g,
            "mape": mape,
            "holdout_mape": best_mape_holdout,
            "holdout_wmape": best_wmape_holdout,
            "accuracy": holdout_accuracy,
            "params": json.dumps(best_params)
        })

    # Save metrics
    metrics_df = pd.DataFrame(results)
    metrics_out = os.path.join(forecast_dir, "metrics.csv")
    metrics_df.to_csv(metrics_out, index=False)
    print("Saved metrics to", metrics_out)
    print(metrics_df.sort_values("mape"))


if __name__ == "__main__":
    main()