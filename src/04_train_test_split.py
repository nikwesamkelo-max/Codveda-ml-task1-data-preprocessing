"""Step 4 - Split the stock prices dataset into training and testing sets."""
import os
import pandas as pd

INPUT_PATH = "data/processed/stock_prices_step3_scaled.csv"
TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"
TRAIN_FRACTION = 0.8  # must match step 3, where the scaler was fitted


def split_by_date(df, train_fraction=TRAIN_FRACTION):
    """Train = earliest 80% of trading days, test = the latest 20%.

    Stock prices are a time series, so we never train on the future.
    A random split would let the model 'peek' at days that come after
    the ones it is tested on.
    """
    dates = df["date"].drop_duplicates().sort_values().reset_index(drop=True)
    cutoff = dates.iloc[int(len(dates) * train_fraction) - 1]
    train = df[df["date"] <= cutoff].copy()
    test = df[df["date"] > cutoff].copy()
    return train, test, cutoff


def main():
    df = pd.read_csv(INPUT_PATH, parse_dates=["date"])
    train, test, cutoff = split_by_date(df)

    print(f"Cutoff date: {cutoff.date()}")
    print(f"Train: {len(train):,} rows ({len(train)/len(df):.1%}) "
          f"| {train['date'].min().date()} to {train['date'].max().date()}")
    print(f"Test : {len(test):,} rows ({len(test)/len(df):.1%}) "
          f"| {test['date'].min().date()} to {test['date'].max().date()}")

    # Checks
    assert len(train) + len(test) == len(df), "Rows were lost"
    assert train["date"].max() < test["date"].min(), "Train and test overlap in time"
    # Matches step 3: scaled training data should have mean ~0, std ~1
    num = ["open", "high", "low", "close", "volume"]
    assert train[num].mean().abs().max() < 1e-6, "Split does not match scaler"

    new_in_test = set(test["symbol"]) - set(train["symbol"])
    print(f"Symbols only in test set: {len(new_in_test)}")
    print("Split verified")

    os.makedirs("data/processed", exist_ok=True)
    train.to_csv(TRAIN_PATH, index=False)
    test.to_csv(TEST_PATH, index=False)
    print(f"Saved -> {TRAIN_PATH} and {TEST_PATH}")


if __name__ == "__main__":
    main()
