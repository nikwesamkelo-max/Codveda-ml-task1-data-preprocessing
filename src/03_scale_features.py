"""Step 3 - Standardize the numerical features of the stock prices dataset."""
import os
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

INPUT_PATH = "data/processed/stock_prices_step2_encoded.csv"
OUTPUT_PATH = "data/processed/stock_prices_step3_scaled.csv"
SCALER_PATH = "artifacts/scaler.joblib"

# Only true measurements get scaled. 'symbol_encoded' is an ID, not a quantity.
NUMERIC_COLS = ["open", "high", "low", "close", "volume"]
TRAIN_FRACTION = 0.8


def get_cutoff_date(df, train_fraction=TRAIN_FRACTION):
    """Last date of the training period (first 80% of trading days)."""
    dates = df["date"].drop_duplicates().sort_values().reset_index(drop=True)
    return dates.iloc[int(len(dates) * train_fraction) - 1]


def standardize(df, cols, cutoff):
    """Rescale each column to mean 0 and std 1.

    The scaler learns its mean/std from the TRAINING period only, then is
    applied to every row. This stops information from the future (test
    period) leaking into training.
    """
    scaler = StandardScaler()
    train_mask = df["date"] <= cutoff
    scaler.fit(df.loc[train_mask, cols])
    df[cols] = scaler.transform(df[cols])
    return df, scaler


def main():
    df = pd.read_csv(INPUT_PATH, parse_dates=["date"])
    cutoff = get_cutoff_date(df)
    print(f"Training period ends: {cutoff.date()}\n")

    print("BEFORE scaling:")
    print(df[NUMERIC_COLS].describe().loc[["mean", "std", "min", "max"]].round(2), "\n")

    df, scaler = standardize(df, NUMERIC_COLS, cutoff)

    train = df[df["date"] <= cutoff]
    print("AFTER scaling (training period):")
    print(train[NUMERIC_COLS].describe().loc[["mean", "std"]].round(3), "\n")

    # Checks
    assert train[NUMERIC_COLS].mean().abs().max() < 1e-6, "Mean is not ~0"
    assert (train[NUMERIC_COLS].std() - 1).abs().max() < 1e-3, "Std is not ~1"
    assert df.isna().sum().sum() == 0
    print("Scaling verified")

    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Saved -> {OUTPUT_PATH} and {SCALER_PATH}")


if __name__ == "__main__":
    main()
