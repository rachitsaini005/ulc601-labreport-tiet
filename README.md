# Machine Learning Techniques — Python Laboratory Programs

A collection of Python implementations covering numerical computing, exploratory data analysis, supervised learning, dimensionality reduction, regression, and unsupervised clustering. The repository organizes the lab work by **what each program does**, using a numeric prefix for a predictable reading order rather than naming files “Experiment 1”, “Experiment 2”, etc.

## Contents

| File | What it does |
|---|---|
| `01_python_environment_setup.md` | Install Python, create a virtual environment, and install dependencies. The original installation activity did not contain a standalone Python program. |
| `02_numpy_fundamentals.py` | Demonstrates NumPy arrays, numerical tests, comparisons, array creation, random values, and iteration. |
| `03_dataset_exploration_and_ann.py` | Explores Iris/Wine data, creates boxplots and correlation heatmaps, applies IQR outlier filtering, and evaluates binary/multiclass ANN models. |
| `04_openml_ann_classification.py` | Trains multilayer perceptrons on selected OpenML datasets; exports classification metrics and plots. |
| `05_linear_regression_metrics.py` | Runs linear regression on supplied CSV/TXT files and reports MAE, MSE, RMSE, and R² in an Excel workbook. Assumes the last column is the target. |
| `06_gradient_descent_linear_fit.py` | Fits a line using batch gradient descent and saves the fitted-line plot and cost history. Assumes the first two numeric columns are X and y. |
| `07_pca_lda_dimensionality_reduction.py` | Standardizes Iris features and visualizes two-component PCA and LDA projections. |
| `08_openml_svm_classification.py` | Evaluates an RBF Support Vector Machine across selected OpenML datasets using stratified cross-validation. |
| `09_svm_and_decision_tree_benchmark.py` | Compares SVM and entropy-based decision trees on 30 datasets. Four are real scikit-learn datasets; the other 26 are synthetic benchmark data. |
| `10_svm_naive_bayes_decision_tree_benchmark.py` | Compares SVM, Gaussian Naive Bayes, and entropy-based decision trees using the same benchmark design as file 09. |
| `11_kmeans_clustering.py` | Runs K-Means clustering with elbow and silhouette analysis, PCA visualization, cluster summaries, and CSV export. |

## Setup

Python 3.10+ is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate       # macOS/Linux
# .venv\\Scripts\\activate    # Windows PowerShell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

OpenML-based scripts need internet access. On macOS/Linux, use `python script_name.py` to run a script. Some programs prompt for a local dataset path; keep the file path handy. The NumPy, PCA/LDA, and most benchmark scripts can run without supplying a dataset.

## Dataset notes

- `03_dataset_exploration_and_ann.py` uses local `Iris.csv` and `Wine.csv` when available; otherwise it falls back to the versions bundled with scikit-learn.
- `04_openml_ann_classification.py` and `08_openml_svm_classification.py` retrieve datasets by OpenML ID. Dataset availability, default target attributes, and network access can affect which datasets complete successfully.
- `05_linear_regression_metrics.py` expects the final column after one-hot encoding to be a numeric regression target. If a dataset uses another target column, edit the script before running it.
- `06_gradient_descent_linear_fit.py` uses the first two numeric columns as input and target. Adjust the column selection for datasets with a different layout.
- Files 09 and 10 intentionally label generated benchmark data with a `_synthetic` suffix. They do **not** claim those records are the original UCI datasets. To benchmark actual UCI datasets, replace the synthetic dataset generator with downloaded/licensed dataset files and document their sources.
- `11_kmeans_clustering.py` accepts a CSV path and creates an output directory containing plots and the clustered dataset.

## Output

Programs write generated charts, CSV summaries, or Excel workbooks into clearly named output directories or files in the current working directory. These generated results are not committed to the repository; run the scripts to reproduce them.

## Dependencies

See [`requirements.txt`](requirements.txt). Core libraries include NumPy, pandas, Matplotlib, Seaborn, scikit-learn, OpenML, and openpyxl.

## Scope and reproducibility

This repository is a cleaned, runnable study collection based on the topics in a Machine Learning Techniques laboratory report. It preserves the methods and intended analyses while making scripts more portable than notebook-only code. Results can vary with package versions, dataset versions, and train/test splits. The scripts use fixed random seeds where practical for repeatability.

## Suggested citation

If you reuse or adapt these programs, cite the repository owner and the original dataset sources where applicable. Confirm dataset licenses and attribution requirements before redistributing datasets.
