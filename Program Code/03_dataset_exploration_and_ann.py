"""Explore Iris/Wine CSV data, visualize distributions/correlations, remove IQR outliers,
then evaluate an ANN on Iris in binary and multiclass settings.

If Iris.csv or Wine.csv are absent, the script uses the corresponding scikit-learn datasets.
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             ConfusionMatrixDisplay, roc_curve, auc,
                             precision_recall_curve, average_precision_score)

warnings.filterwarnings("ignore", category=UserWarning)
OUT = Path("dataset_analysis_results")
OUT.mkdir(exist_ok=True)


def get_dataset(filename, loader, target_name):
    path = Path(filename)
    if path.exists():
        df = pd.read_csv(path, header=None)
        return df
    bunch = loader()
    df = pd.DataFrame(bunch.data, columns=bunch.feature_names)
    df[target_name] = bunch.target
    return df


def numeric_features(df):
    return df.select_dtypes(include=np.number)


def remove_iqr_outliers(df, columns):
    clean = df.copy()
    for col in columns:
        q1, q3 = clean[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        clean = clean[clean[col].between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)]
    return clean


def explore(name, df):
    print(f"\n--- {name} ---\nShape: {df.shape}\nTypes:\n{df.dtypes}\nLast five rows:\n{df.tail()}")
    features = numeric_features(df)
    if not features.empty:
        plt.figure(figsize=(10, 5))
        sns.boxplot(data=features)
        plt.xticks(rotation=45, ha="right")
        plt.title(f"{name}: feature boxplots")
        plt.tight_layout()
        plt.savefig(OUT / f"{name.lower()}_boxplot_before.png", dpi=150)
        plt.close()
        plt.figure(figsize=(9, 7))
        sns.heatmap(features.corr(), annot=True, fmt=".2f", cmap="coolwarm")
        plt.title(f"{name}: correlation matrix")
        plt.tight_layout()
        plt.savefig(OUT / f"{name.lower()}_correlation.png", dpi=150)
        plt.close()
        cleaned = remove_iqr_outliers(df, features.columns)
        print(f"Rows after sequential IQR filtering: {len(cleaned)} (before: {len(df)})")
        plt.figure(figsize=(10, 5))
        sns.boxplot(data=numeric_features(cleaned))
        plt.xticks(rotation=45, ha="right")
        plt.title(f"{name}: feature boxplots after IQR filtering")
        plt.tight_layout()
        plt.savefig(OUT / f"{name.lower()}_boxplot_after.png", dpi=150)
        plt.close()


def evaluate_ann(X, y, label, prefix):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=700,
                          early_stopping=True, random_state=42)
    model.fit(X_train, y_train)
    pred, probs = model.predict(X_test), model.predict_proba(X_test)
    average = "binary" if len(np.unique(y)) == 2 else "weighted"
    metrics = {
        "task": label,
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, average=average, zero_division=0),
        "recall": recall_score(y_test, pred, average=average, zero_division=0),
        "f1": f1_score(y_test, pred, average=average, zero_division=0),
    }
    print(metrics)
    ConfusionMatrixDisplay.from_predictions(y_test, pred)
    plt.title(f"{label}: confusion matrix")
    plt.tight_layout(); plt.savefig(OUT / f"{prefix}_confusion_matrix.png", dpi=150); plt.close()
    if len(np.unique(y)) == 2:
        fpr, tpr, _ = roc_curve(y_test, probs[:, 1])
        precision, recall, _ = precision_recall_curve(y_test, probs[:, 1])
        plt.figure(); plt.plot(fpr, tpr, label=f"AUC={auc(fpr,tpr):.3f}"); plt.plot([0,1],[0,1],"k--")
        plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.legend(); plt.title(label + " ROC")
        plt.tight_layout(); plt.savefig(OUT / f"{prefix}_roc.png", dpi=150); plt.close()
        plt.figure(); plt.plot(recall, precision, label=f"AP={average_precision_score(y_test, probs[:,1]):.3f}")
        plt.xlabel("Recall"); plt.ylabel("Precision"); plt.legend(); plt.title(label + " precision-recall")
        plt.tight_layout(); plt.savefig(OUT / f"{prefix}_pr.png", dpi=150); plt.close()
    else:
        classes = np.unique(y)
        y_bin = label_binarize(y_test, classes=classes)
        plt.figure()
        for i, cls in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
            plt.plot(fpr, tpr, label=f"Class {cls}, AUC={auc(fpr,tpr):.2f}")
        plt.plot([0,1],[0,1],"k--"); plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.legend()
        plt.title(label + " one-vs-rest ROC"); plt.tight_layout(); plt.savefig(OUT / f"{prefix}_roc.png", dpi=150); plt.close()
        plt.figure()
        for i, cls in enumerate(classes):
            precision, recall, _ = precision_recall_curve(y_bin[:, i], probs[:, i])
            plt.plot(recall, precision, label=f"Class {cls}")
        plt.xlabel("Recall"); plt.ylabel("Precision"); plt.legend(); plt.title(label + " precision-recall")
        plt.tight_layout(); plt.savefig(OUT / f"{prefix}_pr.png", dpi=150); plt.close()
    return metrics


def main():
    iris = get_dataset("Iris.csv", load_iris, "target")
    wine = get_dataset("Wine.csv", load_wine, "target")
    explore("Iris", iris)
    explore("Wine", wine)

    # Treat the final column as the class label; UCI Iris files often store species as text.
    X_df = iris.iloc[:, :-1].apply(pd.to_numeric, errors="coerce")
    y_series = iris.iloc[:, -1].astype(str)
    valid = X_df.notna().all(axis=1) & y_series.notna()
    X = X_df.loc[valid].to_numpy()
    y = LabelEncoder().fit_transform(y_series.loc[valid])
    results = []
    mask = np.isin(y, np.unique(y)[:2])
    results.append(evaluate_ann(X[mask], y[mask], "Iris binary ANN", "iris_binary"))
    results.append(evaluate_ann(X, y, "Iris multiclass ANN", "iris_multiclass"))
    pd.DataFrame(results).to_csv(OUT / "ann_metrics.csv", index=False)
    print(f"Results saved in: {OUT.resolve()}")


if __name__ == "__main__":
    main()
