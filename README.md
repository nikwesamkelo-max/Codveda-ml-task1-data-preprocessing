# Codveda-ml-task1-data-preprocessing
Internship Task
# Data Preprocessing for Machine Learning: Stock Prices

Codveda Technology Machine Learning Internship · Level 1, Task 1

A reproducible pipeline that turns a raw stock price dataset into
machine-learning-ready training and testing sets.

## Dataset

Daily prices for 505 stocks (2014-01-02 to 2017-12-29), 497,472 rows with the
columns `symbol, date, open, high, low, close, volume`.

## Pipeline

| Step | Script | What it does |
|------|--------|--------------|
| 1 | `01_handle_missing_data.py` | Detects and fills missing `open`/`high`/`low` values |
| 2 | `02_encode_categorical.py` | Label-encodes `symbol` and saves a lookup table |
| 3 | `03_scale_features.py` | Standardizes price and volume columns |
| 4 | `04_train_test_split.py` | Splits into train/test sets by date |

## Key decisions

**Missing data.** Only 11 rows (0.002%) had gaps. Instead of dropping them, I
tested fill strategies on 5,000 known values that I hid:

| Method | Average error |
|--------|---------------|
| Global mean | ~96% |
| Median per stock | ~15% |
| Close price (same or previous day) | under 1% |

Open is filled from the previous day's close; high and low from the same day's
close. Prices are then checked so that low <= open/close <= high.

**Encoding.** `symbol` has 505 categories. One-hot encoding would add 505
columns across ~500k rows, so I used label encoding. This suits tree-based
models; linear models would need one-hot encoding instead.

**Scaling.** Standardization (mean 0, std 1) was chosen over min-max because of
extreme values (prices up to about $2,049; volume up to 618M). The scaler is
fitted on the training period only to prevent data leakage. `symbol_encoded` is
an ID, so it is not scaled.

**Split.** Prices are a time series, so the data is split by date: the first
80% of trading days for training (up to 2017-03-14), the latest 20% for
testing. A random split would let the model train on the future.

## Results

| | Rows | Period |
|---|---|---|
| Train | 395,939 | 2014-01-02 to 2017-03-14 |
| Test | 101,533 | 2017-03-15 to 2017-12-29 |

Each step includes automated checks (no rows lost, no missing values, codes
decode correctly, scaled mean ~0 / std ~1, no train/test overlap).

## Setup and run

```bash
pip install -r requirements.txt
python run_all.py
```

The full pipeline takes about 20 seconds. Outputs are written to
`data/processed/` and `artifacts/` (not tracked in Git).

To run a single step: `python src/03_scale_features.py`
(run from the repository root).

### Run in Google Colab

```python
!git clone https://github.com/nikwesamkelo-max/codveda-ml-task1-data-preprocessing.git
%cd codveda-ml-task1-data-preprocessing
!pip install -r requirements.txt
!python run_all.py
```

## Tools

Python, pandas, scikit-learn, joblib

## Author

Samkelo · [GitHub](https://github.com/nikwesamkelo-max) ·
[LinkedIn](https://linkedin.com/in/samkelo-nikwe-93226b344)
