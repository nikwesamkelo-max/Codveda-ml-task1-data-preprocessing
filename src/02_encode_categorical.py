"""Step 2 - Encode categorical variables in the stock prices dataset."""
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

INPUT_PATH = "data/processed/stock_prices_step1_missing_handled.csv"
OUTPUT_PATH = "data/processed/stock_prices_step2_encoded.csv"
MAPPING_PATH = "data/processed/symbol_mapping.csv"


def find_categorical_columns(df):
    """Columns holding text (categories) rather than numbers or dates."""
    return df.select_dtypes(include=["object", "string"]).columns.tolist()


def label_encode(df, col):
    """Replace each category with an integer. Returns the data and a lookup table."""
    encoder = LabelEncoder()
    df[f"{col}_encoded"] = encoder.fit_transform(df[col])
    mapping = pd.DataFrame({col: encoder.classes_,
                            f"{col}_encoded": range(len(encoder.classes_))})
    return df, mapping


def main():
    df = pd.read_csv(INPUT_PATH, parse_dates=["date"])
    cats = find_categorical_columns(df)
    print("Categorical columns:", cats)
    print(f"'symbol' has {df['symbol'].nunique()} unique values\n")

    # Why not one-hot for ALL symbols? Compare the sizes:
    top5 = df["symbol"].value_counts().head(5).index
    demo = pd.get_dummies(df[df["symbol"].isin(top5)], columns=["symbol"], dtype="int8")
    print(f"One-hot on 5 symbols   -> {demo.filter(like='symbol_').shape[1]} new columns")
    print(f"One-hot on all symbols -> {df['symbol'].nunique()} new columns "
          f"x {len(df):,} rows (huge)\n")

    # So we use label encoding for 'symbol'
    df, mapping = label_encode(df, "symbol")

    # Checks
    assert df["symbol_encoded"].isna().sum() == 0
    assert df["symbol_encoded"].nunique() == df["symbol"].nunique()
    back = df["symbol_encoded"].map(dict(zip(mapping["symbol_encoded"], mapping["symbol"])))
    assert (back == df["symbol"]).all(), "Decoding does not match original"
    print(mapping.head())
    print("\nEncoding verified: every code maps back to its symbol")

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    mapping.to_csv(MAPPING_PATH, index=False)
    print(f"Saved -> {OUTPUT_PATH} and {MAPPING_PATH}")


if __name__ == "__main__":
    main()
