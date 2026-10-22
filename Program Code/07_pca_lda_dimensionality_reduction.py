"""Standardize Iris features and compare PCA and LDA projections."""
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def main():
    data = load_iris()
    X = StandardScaler().fit_transform(data.data)
    y = data.target
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    # Iris has three classes, so LDA supports at most min(features, classes - 1) = 2 components.
    lda = LinearDiscriminantAnalysis(n_components=2)
    X_lda = lda.fit_transform(X, y)
    print("Original shape:", data.data.shape)
    print("PCA shape:", X_pca.shape, "explained variance:", pca.explained_variance_ratio_)
    print("LDA shape:", X_lda.shape)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for cls, name in enumerate(data.target_names):
        axes[0].scatter(X_pca[y == cls, 0], X_pca[y == cls, 1], label=name)
        axes[1].scatter(X_lda[y == cls, 0], X_lda[y == cls, 1], label=name)
    axes[0].set_title("PCA projection"); axes[1].set_title("LDA projection")
    for ax in axes:
        ax.set_xlabel("Component 1"); ax.set_ylabel("Component 2"); ax.legend()
    plt.tight_layout(); plt.savefig("pca_lda_iris.png", dpi=150); plt.show()


if __name__ == "__main__":
    main()
