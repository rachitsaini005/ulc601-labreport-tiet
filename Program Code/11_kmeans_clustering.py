"""K-Means clustering with elbow and silhouette analysis, PCA visualization and CSV export."""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score

warnings.filterwarnings("ignore", category=UserWarning)
OUT = Path("kmeans_results")
OUT.mkdir(exist_ok=True)


def main():
    path = input("Path to clustering CSV (e.g. Cluster.csv): ").strip().strip('"')
    if not Path(path).exists():
        raise SystemExit(f"File not found: {path}")
    original = pd.read_csv(path)
    if original.empty or original.shape[0] < 3:
        raise SystemExit("Dataset must contain at least three rows.")
    data = original.copy()
    for col in data.select_dtypes(include="object").columns:
        data[col] = LabelEncoder().fit_transform(data[col].astype(str))
    id_cols = [c for c in data.columns if "id" in c.lower()]
    data = data.drop(columns=id_cols, errors="ignore")
    data = data.select_dtypes(include=np.number).replace([np.inf, -np.inf], np.nan)
    data = data.fillna(data.median()).dropna(axis=1, how="all")
    if data.shape[1] == 0:
        raise SystemExit("No usable numeric features found.")
    X = StandardScaler().fit_transform(data)
    max_k = min(10, len(X) - 1)
    ks = range(1, max_k + 1)
    inertias = []
    for k in ks:
        model = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
        inertias.append(model.inertia_)
    plt.figure(figsize=(8, 4)); plt.plot(list(ks), inertias, "o-")
    plt.xlabel("k"); plt.ylabel("Inertia (WCSS)"); plt.title("Elbow method")
    plt.tight_layout(); plt.savefig(OUT / "elbow_method.png", dpi=150); plt.close()

    silhouette_scores = {}
    for k in range(2, min(9, len(X))):
        labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)
        if len(np.unique(labels)) > 1:
            silhouette_scores[k] = silhouette_score(X, labels)
    if not silhouette_scores:
        raise SystemExit("Could not compute silhouette scores for this dataset.")
    best_k = max(silhouette_scores, key=silhouette_scores.get)
    print("Silhouette scores:", silhouette_scores, "\nSelected k:", best_k)
    plt.figure(figsize=(7, 4)); plt.bar(silhouette_scores.keys(), silhouette_scores.values())
    plt.xlabel("k"); plt.ylabel("Silhouette score"); plt.title("Silhouette score by k")
    plt.tight_layout(); plt.savefig(OUT / "silhouette_scores.png", dpi=150); plt.close()

    model = KMeans(n_clusters=best_k, n_init=10, random_state=42)
    labels = model.fit_predict(X)
    result = original.copy(); result["Cluster"] = labels
    result.to_csv(OUT / "clustered_output.csv", index=False)
    print("Cluster sizes:\n", result["Cluster"].value_counts().sort_index())
    print("Cluster means (numeric columns):\n", result.groupby("Cluster").mean(numeric_only=True).round(2))

    pca = PCA(n_components=2, random_state=42)
    projected = pca.fit_transform(X)
    centers = pca.transform(model.cluster_centers_)
    plt.figure(figsize=(8, 6))
    for cluster in range(best_k):
        mask = labels == cluster
        plt.scatter(projected[mask, 0], projected[mask, 1], label=f"Cluster {cluster}", alpha=0.7)
    plt.scatter(centers[:, 0], centers[:, 1], marker="X", s=180, label="Centroids")
    plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title("K-Means clusters in PCA space"); plt.legend()
    plt.tight_layout(); plt.savefig(OUT / "pca_clusters.png", dpi=150); plt.close()

    if X.shape[0] > best_k:
        samples = silhouette_samples(X, labels)
        plt.figure(figsize=(8, 5))
        y_lower = 10
        for cluster in range(best_k):
            vals = np.sort(samples[labels == cluster])
            y_upper = y_lower + len(vals)
            plt.fill_betweenx(np.arange(y_lower, y_upper), 0, vals, alpha=0.7, label=f"Cluster {cluster}")
            y_lower = y_upper + 10
        plt.axvline(np.mean(samples), linestyle="--", label=f"Mean={np.mean(samples):.3f}")
        plt.xlabel("Silhouette coefficient"); plt.ylabel("Samples by cluster"); plt.legend()
        plt.tight_layout(); plt.savefig(OUT / "silhouette_plot.png", dpi=150); plt.close()

    numeric_cols = list(data.columns[:min(6, data.shape[1])])
    plot_data = result[numeric_cols + ["Cluster"]].copy()
    plot_data["Cluster"] = plot_data["Cluster"].astype(str)
    if len(numeric_cols) >= 2:
        sns.pairplot(plot_data, hue="Cluster", diag_kind="hist", corner=True)
        plt.savefig(OUT / "pairplot_clusters.png", dpi=150, bbox_inches="tight"); plt.close("all")
    for col in numeric_cols:
        plt.figure(figsize=(7, 4)); sns.boxplot(data=plot_data, x="Cluster", y=col)
        plt.title(f"{col} distribution by cluster"); plt.tight_layout()
        plt.savefig(OUT / f"boxplot_{col[:35]}.png", dpi=140); plt.close()
    print(f"Outputs saved in {OUT.resolve()}")


if __name__ == "__main__":
    main()
