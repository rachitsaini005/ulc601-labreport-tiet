"""Run linear regression on one or more CSV/TXT datasets and export regression metrics.
Assumption: the final column is the target; edit TARGET_COLUMN if that is not true.
"""
from pathlib import Path
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_table(path):
    path = Path(path)
    if path.suffix.lower() in {".txt", ".data"}:
        return pd.read_csv(path, header=None, sep=None, engine="python")
    return pd.read_csv(path)


def evaluate(path):
    df = load_table(path)
    df = pd.get_dummies(df, drop_first=True)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")
    if df.shape[1] < 2:
        raise ValueError("Need at least one feature and one target column.")
    df = df.fillna(df.median(numeric_only=True)).dropna()
    X = df.iloc[:, :-1].select_dtypes(include=np.number)
    y = pd.to_numeric(df.iloc[:, -1], errors="coerce")
    valid = y.notna()
    X, y = X.loc[valid], y.loc[valid]
    if X.empty or len(X) < 5:
        raise ValueError("Not enough numeric rows/features for regression.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    start = time.perf_counter()
    model = LinearRegression().fit(X_train, y_train)
    predictions = model.predict(X_test)
    duration = time.perf_counter() - start
    mse = mean_squared_error(y_test, predictions)
    return {"dataset": Path(path).name, "rows_used": len(X), "training_and_prediction_seconds": round(duration, 4),
            "MAE": mean_absolute_error(y_test, predictions), "MSE": mse,
            "RMSE": np.sqrt(mse), "R2": r2_score(y_test, predictions)}


def main():
    raw = input("Enter CSV/TXT file paths separated by commas: ").strip()
    paths = [p.strip().strip("\"") for p in raw.split(",") if p.strip()]
    if not paths:
        print("No file paths provided."); return
    rows = []
    for path in paths:
        try:
            result = evaluate(path); rows.append(result); print(result)
        except Exception as exc:
            print(f"Could not process {path}: {exc}")
    if rows:
        out = Path("linear_regression_metrics.xlsx")
        pd.DataFrame(rows).to_excel(out, index=False)
        print(f"Saved metrics to {out.resolve()}")


if __name__ == "__main__":
    main()
