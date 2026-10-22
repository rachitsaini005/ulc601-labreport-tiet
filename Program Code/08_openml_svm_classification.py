"""Evaluate an RBF SVM on selected OpenML datasets with stratified cross-validation.
Metrics are exported to Excel; confusion matrices and ROC/PR curves are saved as PNGs.
"""
from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             ConfusionMatrixDisplay, roc_curve, auc,
                             precision_recall_curve)
try:
    import openml
except ImportError as exc:
    raise SystemExit("Install OpenML first: pip install openml") from exc

OUT = Path("openml_svm_results")
OUT.mkdir(exist_ok=True)
DATASET_IDS = [61, 37, 1494, 151, 40668, 40945, 1464, 4534, 40498, 41021]


def main():
    summaries = []
    with pd.ExcelWriter(OUT / "svm_cross_validation.xlsx", engine="openpyxl") as writer:
        for dataset_id in DATASET_IDS:
            try:
                ds = openml.datasets.get_dataset(dataset_id)
                X, y, _, _ = ds.get_data(target=ds.default_target_attribute, dataset_format="dataframe")
                X = pd.get_dummies(X, dummy_na=True).replace([np.inf, -np.inf], np.nan)
                X = X.fillna(X.median(numeric_only=True)).fillna(0)
                y = LabelEncoder().fit_transform(pd.Series(y).astype(str))
                counts = pd.Series(y).value_counts()
                n_splits = min(5, int(counts.min()))
                if n_splits < 2:
                    raise ValueError("At least one class has fewer than two samples")
                cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                model = make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True, random_state=42))
                cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                                    random_state=42, stratify=y)
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                probs = model.predict_proba(X_test)
                name = ds.name.replace(" ", "_")[:40]
                row = {"dataset_id": dataset_id, "dataset": ds.name,
                       "cv_accuracy_mean": cv_scores.mean(), "cv_accuracy_std": cv_scores.std(),
                       "test_accuracy": accuracy_score(y_test, pred),
                       "precision_weighted": precision_score(y_test, pred, average="weighted", zero_division=0),
                       "recall_weighted": recall_score(y_test, pred, average="weighted", zero_division=0),
                       "f1_weighted": f1_score(y_test, pred, average="weighted", zero_division=0)}
                summaries.append(row)
                ConfusionMatrixDisplay.from_predictions(y_test, pred)
                plt.title(f"{name}: SVM confusion matrix"); plt.tight_layout()
                plt.savefig(OUT / f"{dataset_id}_confusion_matrix.png", dpi=150); plt.close()
                classes = np.unique(y)
                if len(classes) == 2:
                    fpr, tpr, _ = roc_curve(y_test, probs[:, 1])
                    precision, recall, _ = precision_recall_curve(y_test, probs[:, 1])
                    plt.figure(); plt.plot(fpr, tpr, label=f"AUC={auc(fpr,tpr):.3f}"); plt.plot([0,1],[0,1],"k--")
                    plt.legend(); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"{name}: ROC")
                    plt.tight_layout(); plt.savefig(OUT / f"{dataset_id}_roc.png", dpi=150); plt.close()
                    plt.figure(); plt.plot(recall, precision); plt.xlabel("Recall"); plt.ylabel("Precision")
                    plt.title(f"{name}: precision-recall"); plt.tight_layout()
                    plt.savefig(OUT / f"{dataset_id}_pr.png", dpi=150); plt.close()
                pd.DataFrame({"fold": np.arange(1, len(cv_scores)+1), "accuracy": cv_scores}).to_excel(
                    writer, sheet_name=f"dataset_{dataset_id}"[:31], index=False)
                print(f"{ds.name}: CV accuracy {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
            except Exception as exc:
                print(f"Skipping dataset {dataset_id}: {exc}")
        pd.DataFrame(summaries).to_excel(writer, sheet_name="Summary", index=False)
    print(f"Saved results in {OUT.resolve()}")


if __name__ == "__main__":
    main()
