#!/usr/bin/env python3
"""
Usage:
  python scripts/split_datasets.py \
      --in data/processed/erisk25_clean.parquet \
      --outdir data/processed \
      --train 0.7 --test 0.3 \
      --random-state 42 \
      --compression zstd
"""

from __future__ import annotations
import argparse
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


def load_dataset(path: Path) -> pd.DataFrame:
    print(f"Loading dataset: {path}")
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    print(f"  → Loaded {len(df):,} rows, columns = {list(df.columns)}")
    return df

def split_dataset(
    df: pd.DataFrame,
    train_ratio: float,
    test_ratio: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if abs(train_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Train + test ratio must sum to 1.0")

    df_train, df_test = train_test_split(
        df,
        train_size=train_ratio,
        random_state=random_state,
        stratify=df["label"],
    )
    print(f"Split complete: train={len(df_train):,}, test={len(df_test):,}")
    return df_train, df_test

def save_split(df: pd.DataFrame, path: Path, compression: str = "zstd") -> None:
    print(f"Saving {len(df):,} rows → {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False, compression=compression)
    else:
        df.to_csv(path, index=False)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="input", required=True, help="Input cleaned dataset (.parquet or .csv)")
    parser.add_argument("--outdir", required=True, help="Output directory for splits")
    parser.add_argument("--train", type=float, default=0.7, help="Train set ratio (default 0.7)")
    parser.add_argument("--test", type=float, default=0.3, help="Test set ratio (default 0.3)")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--compression", default="zstd", help="Parquet compression codec")
    args = parser.parse_args()

    df = load_dataset(Path(args.input))
    df_train, df_test = split_dataset(df, args.train, args.test, args.random_state)

    out_dir = Path(args.outdir)
    save_split(df_train, out_dir / "train.parquet", args.compression)
    save_split(df_test, out_dir / "test.parquet", args.compression)

    print("Done")


if __name__ == "__main__":
    main()
