import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATA_PATH = Path("france_load_weather.parquet")
OUT_DIR = Path("data_audit")
OUT_DIR.mkdir(exist_ok=True)


def fmt_pct(x):
    return f"{100 * x:.2f}%"


def robust_outlier_mask(s: pd.Series, k: float = 3.5):
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) == 0:
        return pd.Series(dtype=bool)
    med = s.median()
    mad = np.median(np.abs(s - med))
    if mad == 0:
        return pd.Series(False, index=s.index)
    z = 0.6745 * (s - med) / mad
    return np.abs(z) > k


def iqr_outlier_mask(s: pd.Series, k: float = 1.5):
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) == 0:
        return pd.Series(dtype=bool)
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    lo = q1 - k * iqr
    hi = q3 + k * iqr
    return (s < lo) | (s > hi)


def safe_corr(a, b):
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    mask = a.notna() & b.notna()
    if mask.sum() < 3:
        return np.nan
    return a[mask].corr(b[mask])


def baseline_metrics(y_true, y_pred):
    mask = pd.notna(y_true) & pd.notna(y_pred)
    y_true = y_true[mask].astype(float)
    y_pred = y_pred[mask].astype(float)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape_mask = y_true > 0
    mape = np.mean(np.abs((y_true[mape_mask] - y_pred[mape_mask]) / y_true[mape_mask])) * 100
    return {"mae": float(mae), "rmse": float(rmse), "mape": float(mape), "n": int(len(y_true))}


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"missing file: {DATA_PATH.resolve()}")

    df = pd.read_parquet(DATA_PATH)

    report = {
        "file": str(DATA_PATH.resolve()),
        "rows": int(len(df)),
        "cols": int(df.shape[1]),
        "issues": [],
        "warnings": [],
        "summary": {},
    }

    print("\n=== LOADING ===")
    print(f"shape: {df.shape}")
    print(f"columns: {list(df.columns)}")

    # index handling
    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            df = df.set_index("timestamp").sort_index()
            report["warnings"].append("index was not DatetimeIndex; used timestamp column instead")
        else:
            report["issues"].append("index is not DatetimeIndex and no timestamp column exists")
            print("fatal: no datetime index")
            Path(OUT_DIR / "audit_report.json").write_text(json.dumps(report, indent=2))
            return

    print("\n=== INDEX CHECK ===")
    print(f"date range: {df.index.min()} -> {df.index.max()}")
    print(f"timezone: {df.index.tz}")
    print(f"is_monotonic_increasing: {df.index.is_monotonic_increasing}")
    print(f"duplicate timestamps: {int(df.index.duplicated().sum())}")

    if not df.index.is_monotonic_increasing:
        report["issues"].append("datetime index is not sorted ascending")
        df = df.sort_index()

    dup_count = int(df.index.duplicated().sum())
    if dup_count > 0:
        report["issues"].append(f"{dup_count} duplicate timestamps found")

    if df.index.tz is None:
        report["warnings"].append("datetime index is naive (no timezone)")
    else:
        report["summary"]["timezone"] = str(df.index.tz)

    # expected hourly continuity
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="h", tz=df.index.tz)
    missing_ts = full_idx.difference(df.index)
    expected_rows = len(full_idx)
    missing_count = len(missing_ts)

    print("\n=== TIMELINE CONTINUITY ===")
    print(f"expected hourly rows: {expected_rows}")
    print(f"actual rows:          {len(df)}")
    print(f"missing timestamps:   {missing_count}")

    report["summary"]["expected_hourly_rows"] = int(expected_rows)
    report["summary"]["missing_timestamps"] = int(missing_count)

    if missing_count > 0:
        report["issues"].append(f"{missing_count} hourly timestamps are missing")
        missing_preview = [str(x) for x in missing_ts[:25]]
        report["summary"]["missing_preview"] = missing_preview
        print("first missing timestamps:")
        for ts in missing_preview[:10]:
            print(" ", ts)

    gaps = df.index.to_series().diff().dropna()
    gap_counts = gaps.value_counts().sort_index()
    gap_table = pd.DataFrame({"gap": gap_counts.index.astype(str), "count": gap_counts.values})
    gap_table.to_csv(OUT_DIR / "gap_counts.csv", index=False)

    print("\nmost common time gaps:")
    print(gap_table.head(10).to_string(index=False))

    non_1h_gaps = gap_counts[gap_counts.index != pd.Timedelta(hours=1)]
    if len(non_1h_gaps) > 0:
        report["warnings"].append("there are non-1h gaps in the index")

    # missing values
    print("\n=== MISSING VALUES ===")
    missing_by_col = df.isna().sum().sort_values(ascending=False)
    print(missing_by_col.to_string())
    missing_by_col.to_csv(OUT_DIR / "missing_by_col.csv")

    nonzero_missing = missing_by_col[missing_by_col > 0]
    if len(nonzero_missing) > 0:
        report["issues"].append("one or more columns contain NaNs")

    # duplicated full rows
    dup_rows = int(df.duplicated().sum())
    print(f"\nduplicate rows: {dup_rows}")
    if dup_rows > 0:
        report["warnings"].append(f"{dup_rows} duplicated rows found")

    # dtypes
    print("\n=== DTYPES ===")
    print(df.dtypes.to_string())
    pd.DataFrame({"dtype": df.dtypes.astype(str)}).to_csv(OUT_DIR / "dtypes.csv")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_cols = [c for c in df.columns if c not in numeric_cols]

    # basic stats
    print("\n=== DESCRIPTIVE STATS ===")
    desc = df[numeric_cols].describe().T
    desc["missing_pct"] = df[numeric_cols].isna().mean() * 100
    desc.to_csv(OUT_DIR / "describe_numeric.csv")
    print(desc.round(3).to_string())

    # constant / near-constant columns
    print("\n=== CONSTANT / LOW-VARIANCE CHECK ===")
    nunique = df.nunique(dropna=False).sort_values()
    low_var = nunique[nunique <= 2]
    print(low_var.to_string())
    pd.DataFrame({"nunique": nunique}).to_csv(OUT_DIR / "nunique.csv")

    # required columns sanity
    expected = [
        "load_mw", "temperature_2m", "relative_humidity_2m", "wind_speed_10m",
        "shortwave_radiation", "cloud_cover", "hour", "day_of_week", "month",
        "is_weekend", "is_holiday"
    ]
    missing_expected = [c for c in expected if c not in df.columns]
    if missing_expected:
        report["issues"].append(f"missing expected columns: {missing_expected}")

    # domain checks
    print("\n=== DOMAIN CHECKS ===")
    domain_checks = {}

    def add_check(name, ok, detail):
        domain_checks[name] = {"ok": bool(ok), "detail": detail}
        status = "ok" if ok else "FAIL"
        print(f"{status:>4} | {name}: {detail}")
        if not ok:
            report["issues"].append(f"{name}: {detail}")

    if "load_mw" in df:
        add_check("load_non_negative", (df["load_mw"] >= 0).all(), f"min={df['load_mw'].min():.2f}")
    if "relative_humidity_2m" in df:
        rh_ok = df["relative_humidity_2m"].between(0, 100).all()
        add_check("humidity_in_0_100", rh_ok, f"min={df['relative_humidity_2m'].min():.2f}, max={df['relative_humidity_2m'].max():.2f}")
    if "cloud_cover" in df:
        cc_ok = df["cloud_cover"].between(0, 100).all()
        add_check("cloud_cover_in_0_100", cc_ok, f"min={df['cloud_cover'].min():.2f}, max={df['cloud_cover'].max():.2f}")
    if "hour" in df:
        add_check("hour_in_0_23", df["hour"].between(0, 23).all(), f"min={df['hour'].min()}, max={df['hour'].max()}")
    if "day_of_week" in df:
        add_check("day_of_week_in_0_6", df["day_of_week"].between(0, 6).all(), f"min={df['day_of_week'].min()}, max={df['day_of_week'].max()}")
    if "month" in df:
        add_check("month_in_1_12", df["month"].between(1, 12).all(), f"min={df['month'].min()}, max={df['month'].max()}")
    if "is_weekend" in df:
        add_check("is_weekend_binary", set(df["is_weekend"].dropna().unique()).issubset({0, 1, 0.0, 1.0}), f"values={sorted(df['is_weekend'].dropna().unique().tolist())[:10]}")
    if "is_holiday" in df:
        add_check("is_holiday_binary", set(df["is_holiday"].dropna().unique()).issubset({0, 1, 0.0, 1.0}), f"values={sorted(df['is_holiday'].dropna().unique().tolist())[:10]}")

    # frequency consistency of derived calendar columns
    print("\n=== CALENDAR CONSISTENCY ===")
    if "hour" in df:
        hour_match = (df.index.hour == df["hour"].astype(int)).mean()
        print(f"hour matches index:       {fmt_pct(hour_match)}")
        if hour_match < 1.0:
            report["issues"].append(f"hour column mismatches datetime index in {fmt_pct(1 - hour_match)} of rows")
    if "day_of_week" in df:
        dow_match = (df.index.dayofweek == df["day_of_week"].astype(int)).mean()
        print(f"day_of_week matches idx:  {fmt_pct(dow_match)}")
        if dow_match < 1.0:
            report["issues"].append(f"day_of_week column mismatches datetime index in {fmt_pct(1 - dow_match)} of rows")
    if "month" in df:
        month_match = (df.index.month == df["month"].astype(int)).mean()
        print(f"month matches index:      {fmt_pct(month_match)}")
        if month_match < 1.0:
            report["issues"].append(f"month column mismatches datetime index in {fmt_pct(1 - month_match)} of rows")
    if "is_weekend" in df:
        wk_match = ((df.index.dayofweek >= 5).astype(int) == df["is_weekend"].astype(int)).mean()
        print(f"is_weekend matches idx:   {fmt_pct(wk_match)}")
        if wk_match < 1.0:
            report["issues"].append(f"is_weekend column mismatches datetime index in {fmt_pct(1 - wk_match)} of rows")

    # suspicious exact repeats
    print("\n=== EXACT REPEATS CHECK ===")
    for col in ["load_mw", "temperature_2m", "relative_humidity_2m", "wind_speed_10m", "shortwave_radiation", "cloud_cover"]:
        if col in df:
            same_as_prev = (df[col] == df[col].shift(1)).mean()
            print(f"{col:<22} repeated vs prev row: {fmt_pct(same_as_prev)}")

    # outliers
    print("\n=== OUTLIERS ===")
    outlier_summary = []
    for col in numeric_cols:
        s = df[col]
        iqr_mask = iqr_outlier_mask(s)
        mad_mask = robust_outlier_mask(s)
        outlier_summary.append({
            "column": col,
            "iqr_outliers": int(iqr_mask.sum()) if len(iqr_mask) else 0,
            "iqr_outlier_pct": float(iqr_mask.mean() * 100) if len(iqr_mask) else 0.0,
            "mad_outliers": int(mad_mask.sum()) if len(mad_mask) else 0,
            "mad_outlier_pct": float(mad_mask.mean() * 100) if len(mad_mask) else 0.0,
        })
    outlier_df = pd.DataFrame(outlier_summary).sort_values("mad_outlier_pct", ascending=False)
    outlier_df.to_csv(OUT_DIR / "outliers.csv", index=False)
    print(outlier_df.round(3).to_string(index=False))

    # correlations
    print("\n=== CORRELATIONS WITH LOAD ===")
    corr_rows = []
    if "load_mw" in df:
        for col in numeric_cols:
            if col == "load_mw":
                continue
            corr_rows.append({
                "feature": col,
                "pearson_corr_with_load": safe_corr(df["load_mw"], df[col])
            })
    corr_df = pd.DataFrame(corr_rows).sort_values("pearson_corr_with_load", key=lambda s: s.abs(), ascending=False)
    corr_df.to_csv(OUT_DIR / "corr_with_load.csv", index=False)
    print(corr_df.round(4).to_string(index=False))

    # target memory / naive baselines
    print("\n=== NAIVE BASELINES (ON RAW SERIES) ===")
    if "load_mw" in df:
        y = df["load_mw"]
        baselines = {}
        for lag, name in [(1, "last_hour"), (24, "same_hour_yesterday"), (168, "same_hour_last_week")]:
            baselines[name] = baseline_metrics(y, y.shift(lag))
        base_df = pd.DataFrame(baselines).T
        base_df.to_csv(OUT_DIR / "naive_baselines.csv")
        print(base_df.round(3).to_string())

    # seasonal sanity
    print("\n=== SEASONAL SANITY ===")
    if "load_mw" in df and "hour" in df:
        hourly = df.groupby("hour")["load_mw"].mean()
        weekday = df.groupby("day_of_week")["load_mw"].mean() if "day_of_week" in df else None
        monthly = df.groupby("month")["load_mw"].mean() if "month" in df else None

        print("avg load by hour:")
        print(hourly.round(1).to_string())

        if weekday is not None:
            print("\navg load by weekday:")
            print(weekday.round(1).to_string())

        if monthly is not None:
            print("\navg load by month:")
            print(monthly.round(1).to_string())

    # holiday sanity
    print("\n=== HOLIDAY / WEEKEND DISTRIBUTION ===")
    if "is_holiday" in df:
        holiday_rate = df["is_holiday"].mean()
        print(f"is_holiday rate: {fmt_pct(holiday_rate)}")
        if holiday_rate < 0.01:
            report["warnings"].append("holiday rate is very low; holiday feature may be incomplete")
    if "is_weekend" in df:
        weekend_rate = df["is_weekend"].mean()
        print(f"is_weekend rate: {fmt_pct(weekend_rate)}")

    # leakage red flags
    print("\n=== LEAKAGE RED FLAGS ===")
    leak_flags = []
    bad_names = [c for c in df.columns if any(x in c.lower() for x in ["future", "lead", "target_", "t+"]) and c != "target"]
    if bad_names:
        leak_flags.extend(bad_names)
    if leak_flags:
        print("potential leakage-named columns:", leak_flags)
        report["warnings"].append(f"potential leakage-named columns: {leak_flags}")
    else:
        print("no obvious leakage column names found")

    # plots
    print("\n=== SAVING PLOTS ===")
    fig, axes = plt.subplots(4, 3, figsize=(20, 20))

    # 1
    axes[0, 0].plot(df.index, df["load_mw"], linewidth=0.25)
    axes[0, 0].set_title("Load - full history")
    axes[0, 0].set_ylabel("MW")

    # 2
    last_week = df["load_mw"].iloc[-168:]
    axes[0, 1].plot(last_week.index, last_week.values, linewidth=1.2)
    axes[0, 1].set_title("Load - last 168 hours")
    axes[0, 1].tick_params(axis="x", rotation=30)

    # 3
    if len(missing_ts) > 0:
        miss_s = pd.Series(1, index=missing_ts)
        miss_s.resample("MS").sum().plot(ax=axes[0, 2], kind="bar")
        axes[0, 2].set_title("Missing timestamps by month")
    else:
        axes[0, 2].text(0.5, 0.5, "No missing timestamps", ha="center", va="center")
        axes[0, 2].set_title("Missing timestamps by month")
        axes[0, 2].set_axis_off()

    # 4
    if "hour" in df:
        df.groupby("hour")["load_mw"].mean().plot(ax=axes[1, 0], kind="bar", color="darkorange")
        axes[1, 0].set_title("Average load by hour")

    # 5
    if "day_of_week" in df:
        tmp = df.groupby("day_of_week")["load_mw"].mean()
        axes[1, 1].bar(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], tmp.values, color="mediumseagreen")
        axes[1, 1].set_title("Average load by weekday")

    # 6
    if "month" in df:
        tmp = df.groupby("month")["load_mw"].mean()
        axes[1, 2].bar(tmp.index.astype(str), tmp.values, color="orchid")
        axes[1, 2].set_title("Average load by month")

    # 7
    if "temperature_2m" in df:
        axes[2, 0].hexbin(df["temperature_2m"], df["load_mw"], gridsize=50, mincnt=1)
        axes[2, 0].set_title("Temperature vs load")
        axes[2, 0].set_xlabel("Temperature (C)")
        axes[2, 0].set_ylabel("MW")

    # 8
    if "load_mw" in df:
        axes[2, 1].hist(df["load_mw"], bins=100)
        axes[2, 1].set_title("Load distribution")

    # 9
    if len(gaps) > 0:
        gap_hours = gaps.dt.total_seconds() / 3600
        gap_hours = gap_hours[gap_hours <= 48]
        axes[2, 2].hist(gap_hours, bins=48)
        axes[2, 2].set_title("Gap size distribution (<=48h)")
        axes[2, 2].set_xlabel("hours")

    # 10
    if {"month", "hour", "load_mw"}.issubset(df.columns):
        pivot = df.pivot_table(index="month", columns="hour", values="load_mw", aggfunc="mean")
        im = axes[3, 0].imshow(pivot.values, aspect="auto")
        axes[3, 0].set_title("Mean load heatmap (month x hour)")
        axes[3, 0].set_yticks(range(len(pivot.index)))
        axes[3, 0].set_yticklabels(pivot.index.tolist())
        axes[3, 0].set_xticks(range(0, 24, 2))
        axes[3, 0].set_xticklabels(list(range(0, 24, 2)))
        fig.colorbar(im, ax=axes[3, 0], fraction=0.046, pad=0.04)

    # 11
    if len(corr_df) > 0:
        top_corr = corr_df.head(10).iloc[::-1]
        axes[3, 1].barh(top_corr["feature"], top_corr["pearson_corr_with_load"])
        axes[3, 1].set_title("Top correlations with load")

    # 12
    if "load_mw" in df:
        rolling = df["load_mw"].rolling(24 * 30).mean()
        axes[3, 2].plot(rolling.index, rolling.values, linewidth=1.0)
        axes[3, 2].set_title("30-day rolling mean load")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "data_audit_plots.png", dpi=160)
    plt.close()

    # suspicious periods export
    if len(missing_ts) > 0:
        pd.DataFrame({"missing_timestamp": missing_ts}).to_csv(OUT_DIR / "missing_timestamps.csv", index=False)

    # final summary
    print("\n=== FINAL VERDICT ===")
    if not report["issues"]:
        print("No hard data-quality failures found.")
    else:
        print("Hard issues:")
        for x in report["issues"]:
            print("-", x)

    if report["warnings"]:
        print("\nWarnings:")
        for x in report["warnings"]:
            print("-", x)

    report["summary"]["rows_after_load"] = int(len(df))
    report["summary"]["numeric_cols"] = numeric_cols
    report["summary"]["non_numeric_cols"] = non_numeric_cols

    with open(OUT_DIR / "audit_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\nsaved:")
    print(f"- {OUT_DIR / 'audit_report.json'}")
    print(f"- {OUT_DIR / 'describe_numeric.csv'}")
    print(f"- {OUT_DIR / 'missing_by_col.csv'}")
    print(f"- {OUT_DIR / 'gap_counts.csv'}")
    print(f"- {OUT_DIR / 'outliers.csv'}")
    print(f"- {OUT_DIR / 'corr_with_load.csv'}")
    print(f"- {OUT_DIR / 'naive_baselines.csv'}")
    print(f"- {OUT_DIR / 'data_audit_plots.png'}")


if __name__ == "__main__":
    main()