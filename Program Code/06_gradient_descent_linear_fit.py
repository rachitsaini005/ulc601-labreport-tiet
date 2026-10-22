"""Fit a straight line to two-column data using batch gradient descent.
By default, the first column is X and the second column is y.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def gradient_descent(x, y, learning_rate=0.05, iterations=2000):
    m = len(y)
    weight, bias = 0.0, float(np.mean(y))
    costs = []
    for _ in range(iterations):
        prediction = weight * x + bias
        error = prediction - y
        costs.append(float(np.mean(error ** 2) / 2))
        weight -= learning_rate * np.mean(error * x)
        bias -= learning_rate * np.mean(error)
    return weight, bias, costs


def main():
    path = input("Path to a CSV file (first column X, second column y): ").strip().strip("\"")
    if not Path(path).exists():
        raise SystemExit(f"File not found: {path}")
    df = pd.read_csv(path, header=None if Path(path).suffix.lower() == ".txt" else "infer", sep=None, engine="python")
    numeric = df.apply(pd.to_numeric, errors="coerce").dropna()
    if numeric.shape[1] < 2 or len(numeric) < 2:
        raise SystemExit("Dataset must contain at least two numeric columns and two rows.")
    x_raw = numeric.iloc[:, 0].to_numpy(dtype=float)
    y = numeric.iloc[:, 1].to_numpy(dtype=float)
    x_mean, x_std = x_raw.mean(), x_raw.std()
    if x_std == 0:
        raise SystemExit("The first column has no variation.")
    x = (x_raw - x_mean) / x_std
    weight, bias, costs = gradient_descent(x, y)
    print(f"Scaled-space weight={weight:.5f}, bias={bias:.5f}")
    order = np.argsort(x)
    plt.figure(); plt.scatter(x, y, alpha=0.6); plt.plot(x[order], (weight*x+bias)[order])
    plt.xlabel("Standardized X"); plt.ylabel("y"); plt.title("Linear fit using gradient descent")
    plt.tight_layout(); plt.savefig("gradient_descent_fit.png", dpi=150); plt.show()
    plt.figure(); plt.plot(costs); plt.xlabel("Iteration"); plt.ylabel("Half mean squared error")
    plt.title("Gradient descent cost history"); plt.tight_layout(); plt.savefig("gradient_descent_cost.png", dpi=150); plt.show()
    pd.DataFrame({"iteration": np.arange(1, len(costs)+1), "cost": costs}).to_csv("gradient_descent_cost.csv", index=False)


if __name__ == "__main__":
    main()
