"""Train an ANN on selected OpenML datasets and save metrics/plots to a folder."""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             ConfusionMatrixDisplay, roc_curve, auc,
                             precision_recall_curve, average_precision_score)

try:
    import openml
except ImportError as exc:
    raise SystemExit("Install OpenML first: pip install openml") from exc

OUT = Path("openml_ann_results")
OUT.mkdir(exist_ok=True)
DATASET_IDS = [61, 31, 1464, 151, 1461, 40945, 182, 37, 3, 334]


def main():
    summaries = []
    for dataset_id in DATASET_IDS:
        try:
            dataset = openml.datasets.get_dataset(dataset_id)
            X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute,
                                          dataset_format="dataframe")
            name = dataset.name.replace(" ", "_").replace("/", "_")
            X = pd.get_dummies(X, dummy_na=True)
            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.fillna(X.median(numeric_only=True)).fillna(0)
            y = pd.Series(LabelEncoder().fit_transform(pd.Series(y).astype(str)))
            # Keep stratification feasible for tiny classes.
            counts = y.value_counts()
            stratify = y if counts.min() >= 2 else None
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.30, random_state=42, stratify=stratify)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500,
                                  early_stopping=True, random_state=42)
            model.fit(X_train, y_train)
            pred, prob = model.predict(X_test), model.predict_proba(X_test)
            classes = np.unique(y_test)
            row = {"dataset_id": dataset_id, "dataset": name,
                   "accuracy": accuracy_score(y_test, pred),
                   "precision_weighted": precision_score(y_test, pred, average="weighted", zero_division=0),
                   "recall_weighted": recall_score(y_test, pred, average="weighted", zero_division=0),
                   "f1_weighted": f1_score(y_test, pred, average="weighted", zero_division=0)}
            summaries.append(row)
            ConfusionMatrixDisplay.from_predictions(y_test, pred)
            plt.title(f"{name}: confusion matrix"); plt.tight_layout()
            plt.savefig(OUT / f"{name}_confusion_matrix.png", dpi=150); plt.close()
            plt.figure(); plt.plot(model.loss_curve_)
            plt.xlabel("Iteration"); plt.ylabel("Loss"); plt.title(f"{name}: training loss")
            plt.tight_layout(); plt.savefig(OUT / f"{name}_loss_curve.png", dpi=150); plt.close()
            if len(np.unique(y)) == 2:
                fpr, tpr, _ = roc_curve(y_test, prob[:, 1])
                precision, recall, _ = precision_recall_curve(y_test, prob[:, 1])
                plt.figure(); plt.plot(fpr, tpr, label=f"AUC={auc(fpr,tpr):.3f}"); plt.plot([0,1],[0,1],"k--")
                plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend(); plt.title(f"{name}: ROC")
                plt.tight_layout(); plt.savefig(OUT / f"{name}_roc.png", dpi=150); plt.close()
                plt.figure(); plt.plot(recall, precision, label=f"AP={average_precision_score(y_test, prob[:,1]):.3f}")
                plt.xlabel("Recall"); plt.ylabel("Precision"); plt.legend(); plt.title(f"{name}: PR")
                plt.tight_layout(); plt.savefig(OUT / f"{name}_pr.png", dpi=150); plt.close()
            print(f"Completed {name}: accuracy={row['accuracy']:.3f}")
        except Exception as exc:
            print(f"Skipping OpenML dataset {dataset_id}: {exc}")
    results = pd.DataFrame(summaries)
    results.to_csv(OUT / "ann_metrics.csv", index=False)
    if not results.empty:
        results.set_index("dataset")["accuracy"].plot(kind="bar", figsize=(12, 5))
        plt.ylabel("Accuracy"); plt.tight_layout(); plt.savefig(OUT / "accuracy_by_dataset.png", dpi=150); plt.close()
        results[["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"]].plot(kind="box")
        plt.tight_layout(); plt.savefig(OUT / "metric_boxplot.png", dpi=150); plt.close()
    print(f"Results saved in {OUT.resolve()}")


if __name__ == "__main__":
    main()
