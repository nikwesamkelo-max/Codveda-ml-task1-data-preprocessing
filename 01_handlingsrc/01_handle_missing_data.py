"""Step 1 - Handle missing data in the stock prices dataset."""
import pandas as pd

os.makedirs("data/processed", exist_ok=True)

RAW_PATH = "data/raw/stock_prices.csv"
CLEAN_PATH = "data/processed/stock_prices_step1_missing_handled.csv"
PRICE_COLS = ["open", "high", "low"]


def missing_report(df):
    report = pd.DataFrame({
        "missing": df.isna().sum(),
        "percent": (df.isna().mean() * 100).round(4),
    })
    return report[report["missing"] > 0]


def fill_missing_prices(df):
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    # open: yesterday's close (fall back to today's close)
    prev_close = df.groupby("symbol")["close"].shift()
    df["open"] = df["open"].fillna(prev_close).fillna(df["close"])

    # high / low: today's close
    df["high"] = df["high"].fillna(df["close"])
    df["low"] = df["low"].fillna(df["close"])

    # keep low <= open/close <= high
    df["high"] = df[["high", "open", "close"]].max(axis=1)
    df["low"] = df[["low", "open", "close"]].min(axis=1)
    return df


def main():
    df = pd.read_csv(RAW_PATH, parse_dates=["date"])
    print(f"Loaded {df.shape[0]:,} rows\n")
    print("Missing BEFORE:\n", missing_report(df), "\n")

    clean = fill_missing_prices(df)

    assert clean[PRICE_COLS].isna().sum().sum() == 0
    assert len(clean) == len(df)
    print("Missing AFTER: 0 | Rows kept:", f"{len(clean):,}")

    clean.to_csv(CLEAN_PATH, index=False)


if __name__ == "__main__":
    main()
