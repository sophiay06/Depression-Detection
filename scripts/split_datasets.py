#!/usr/bin/env python3
"""
Group-aware train/test split for eRisk.

Usage:
  python scripts/split_datasets.py \
      --in data/processed/erisk25_clean.parquet \
      --outdir data/processed \
      --train 0.7 --test 0.3 \
      --group-col pre \
      --label-col label \
      --random-state 42 \
      --compression zstd
"""

from __future__ import annotations
import argparse
from pathlib import Path
import sys
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

def load_dataset(path: Path) -> pd.DataFrame:
    print(f"Loading {path}")
    if not path.exists():
        print(f"ERROR: file not found: {path}", file=sys.stderr)
        sys.exit(1)
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        print("ERROR: input must be .parquet or .csv", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded {len(df):,} rows; columns: {list(df.columns)}")
    return df

def basic_checks(df: pd.DataFrame, group_col: str, label_col: str) -> None:
    missing = [c for c in (group_col, label_col) if c not in df.columns]
    if missing:
        print(f"ERROR: missing required columns: {missing}", file=sys.stderr)
        sys.exit(1)
    if df[group_col].isna().any():
        n = int(df[group_col].isna().sum())
        print(f"ERROR: {n} rows have NaN in group column '{group_col}'", file=sys.stderr)
        sys.exit(1)

def do_group_split(
    df: pd.DataFrame,
    group_col: str,
    train_ratio: float,
    test_ratio: float,
    random_state: int,
):
    if abs(train_ratio + test_ratio - 1.0) > 1e-6:
        print("ERROR: --train + --test must equal 1.0", file=sys.stderr)
        sys.exit(1)

    groups = df[group_col].values
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_ratio, random_state=random_state)
    train_idx, test_idx = next(splitter.split(df, groups=groups))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df  = df.iloc[test_idx].reset_index(drop=True)

    # Hard assertions: no leakage
    train_users = set(train_df[group_col].unique())
    test_users  = set(test_df[group_col].unique())
    overlap = train_users & test_users
    if overlap:
        print(f"ERROR: user leakage detected: {len(overlap)} users appear in both splits", file=sys.stderr)
        sys.exit(1)

    return train_df, test_df

def describe_split(name: str, df: pd.DataFrame, group_col: str, label_col: str) -> None:
    n_rows = len(df)
    n_users = df[group_col].nunique()
    print(f"\n=== {name} split ===")
    print(f"Rows:  {n_rows:,}")
    print(f"Users: {n_users:,}")

    # Per-post label distribution
    if label_col in df.columns:
        vc_posts = df[label_col].value_counts(dropna=False).sort_index()
        total_posts = vc_posts.sum()
        print("Label distribution (posts):")
        for k, v in vc_posts.items():
            pct = 100.0 * v / total_posts if total_posts else 0.0
            print(f"  {k}: {v:,} ({pct:.2f}%)")

        # Per-user label distribution (majority label per user)
        user_majority = (
            df.groupby(group_col)[label_col]
              .apply(lambda s: s.value_counts().idxmax())
              .value_counts()
              .sort_index()
        )
        total_users = user_majority.sum()
        print("Label distribution (users; majority label per user):")
        for k, v in user_majority.items():
            pct = 100.0 * v / total_users if total_users else 0.0
            print(f"  {k}: {v:,} ({pct:.2f}%)")

def save_split(df: pd.DataFrame, path: Path, compression: str = "zstd") -> None:
    print(f"Saving {len(df):,} rows → {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False, compression=compression)
    elif path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
    else:
        # default to parquet if no extension
        path = path.with_suffix(".parquet")
        df.to_parquet(path, index=False, compression=compression)
        print(f"    (No extension provided; wrote {path.name})")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="input", required=True, help="Input cleaned dataset (.parquet or .csv)")
    parser.add_argument("--outdir", required=True, help="Output directory for splits")
    parser.add_argument("--train", type=float, default=0.7, help="Train ratio (default 0.7)")
    parser.add_argument("--test", type=float, default=0.3, help="Test ratio (default 0.3)")
    parser.add_argument("--group-col", default="pre", help="User/profile id column (default: pre)")
    parser.add_argument("--label-col", default="label", help="Label/target column (default: label)")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed (default 42)")
    parser.add_argument("--compression", default="zstd", help="Parquet compression (default zstd)")
    parser.add_argument("--prefix", default="", help="Optional filename prefix (e.g., 'erisk_')")
    args = parser.parse_args()

    df = load_dataset(Path(args.input))
    basic_checks(df, args.group_col, args.label_col)

    # Report global dataset stats
    print("\n=== Global dataset stats ===")
    print(f"Total rows:  {len(df):,}")
    print(f"Total users: {df[args.group_col].nunique():,}")

    train_df, test_df = do_group_split(
        df=df,
        group_col=args.group_col,
        train_ratio=args.train,
        test_ratio=args.test,
        random_state=args.random_state,
    )

    # Summaries
    describe_split("Train", train_df, args.group_col, args.label_col)
    describe_split("Test",  test_df,  args.group_col, args.label_col)

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_split(train_df, out_dir / f"{args.prefix}train.parquet", args.compression)
    save_split(test_df,  out_dir / f"{args.prefix}test.parquet",  args.compression)

    print("\nDone")

if __name__ == "__main__":
    main()
