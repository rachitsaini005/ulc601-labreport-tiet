# Python Environment Setup

The original lab's first exercise is an installation/setup activity rather than a Python program. Install Python 3.10 or newer, then create a virtual environment and install the packages listed in `requirements.txt`.

```bash
python3 -m venv .venv
source .venv/bin/activate       # macOS/Linux
# .venv\\Scripts\\activate    # Windows PowerShell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For notebook-based workflows, also install Jupyter with `pip install jupyter` or run the notebook-compatible scripts in Google Colab. The scripts that use OpenML require internet access. Scripts that request CSV files need the relevant dataset file in the current directory or a path supplied when prompted.
