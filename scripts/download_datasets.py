"""
Download the two educational data sources used by AI StudyMate.

1) Hugging Face: allenai/sciq
2) Kaggle: allenai/ai2-science-questions

Kaggle authentication:
- Install/configure Kaggle API credentials.
- Current Kaggle documentation supports kaggle auth login or an API key.
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

def download_huggingface():
    from datasets import load_dataset
    ds = load_dataset("allenai/sciq", split="train")
    df = ds.to_pandas()
    df.to_csv(DATA / "sciq.csv", index=False)
    print(f"SciQ downloaded: {len(df):,} rows")

def download_kaggle():
    import kagglehub
    path = kagglehub.dataset_download("allenai/ai2-science-questions")
    print("Kaggle dataset downloaded to:", path)
    print("Copy the CSV files into:", DATA / "kaggle_ai2")

if __name__ == "__main__":
    try:
        download_huggingface()
    except Exception as e:
        print("Hugging Face download failed:", e)

    try:
        download_kaggle()
    except Exception as e:
        print("Kaggle download failed:", e)
        print("Configure Kaggle authentication and run this script again.")
