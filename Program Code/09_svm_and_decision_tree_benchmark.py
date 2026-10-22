"""Compare SVM and entropy-based decision trees on 30 classification datasets.
Four datasets come from scikit-learn; the remaining 26 are reproducible synthetic datasets
with UCI-style names (not downloaded UCI records). Replace/grow the dataset loader to use
actual downloaded UCI datasets when required.
"""
from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import (load_iris, load_wine, load_breast_cancer, load_digits,
                              make_classification)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             ConfusionMatrixDisplay, roc_curve, auc)

OUT = Path("svm_decision_tree_results")
OUT.mkdir(exist_ok=True)
MODELS = {
    "SVM": make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True, random_state=42)),
    "DecisionTree_Entropy": DecisionTreeClassifier(criterion="entropy", random_state=42),
}


def get_datasets():
    datasets = []
    for name, loader in [("Iris", load_iris), ("Wine", load_wine),
                         ("BreastCancer", load_breast_cancer), ("Digits", load_digits)]:
        d = loader(); datasets.append((name, d.data, d.target))
    specs = [
        ("Glass",214,9,6),("Waveform",1200,40,3),("Spect",267,22,2),
        ("Ionosphere",351,20,2),("KrVsKp",1000,20,2),("Sonar",208,20,2),
        ("Madelon",1000,20,2),("Soybean",200,20,4),("Lymphography",148,18,4),
        ("Congress",435,16,2),("Monk1",124,6,2),("Monk2",169,6,2),
        ("Monk3",122,6,2),("Zoo",101,16,7),("TicTacToe",500,9,2),
        ("Arrhythmia",452,27,4),("Heart",270,13,2),("HillValley",800,20,2),
        ("Horse",368,22,2),("MOfN",1024,13,2),("Vote",435,16,2),
        ("Vowel",528,10,6),("Penglung",73,20,5),("Exactly",500,13,2),
        ("Exactly2",500,13,2),("BreastEW",569,20,2)
    ]
    for idx, (name, n, features, classes) in enumerate(specs, start=1):
        informative = min(features, max(classes, 2))
        X, y = make_classification(n_samples=n, n_features=features, n_informative=informative,
                                   n_redundant=0, n_classes=classes, random_state=idx)
        datasets.append((name + "_synthetic", X, y))
    return datasets


def main():
    summary = []
    for name, X, y in get_datasets():
        counts = pd.Series(y).value_counts()
        n_splits = min(5, int(counts.min()))
        if n_splits < 2:
            print(f"Skipping {name}: insufficient class samples"); continue
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                            random_state=42, stratify=y)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, (model_name, model) in zip(axes, MODELS.items()):
            start = time.perf_counter()
            cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            elapsed = time.perf_counter() - start
            acc = accuracy_score(y_test, pred)
            summary.append({"dataset": name, "model": model_name,
                            "cv_accuracy_mean": cv_scores.mean(), "cv_accuracy_std": cv_scores.std(),
                            "test_accuracy": acc,
                            "precision_weighted": precision_score(y_test, pred, average="weighted", zero_division=0),
                            "recall_weighted": recall_score(y_test, pred, average="weighted", zero_division=0),
                            "f1_weighted": f1_score(y_test, pred, average="weighted", zero_division=0),
                            "elapsed_seconds": elapsed})
            ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=ax, colorbar=False)
            ax.set_title(f"{model_name}\nAccuracy={acc:.3f}")
        fig.suptitle(name); fig.tight_layout()
        fig.savefig(OUT / f"{name}_confusion_matrices.png", dpi=140); plt.close(fig)
        print(f"Processed {name}")
    pd.DataFrame(summary).to_csv(OUT / "model_comparison.csv", index=False)
    print(f"Saved results in {OUT.resolve()}")


if __name__ == "__main__":
    main()
