"""
Split a post-level dataset into TRAIN / (optional VAL) / TEST **by user_id**,
ensuring there is no user overlap across splits.

Assumptions
----------
- The dataset has a string column `doc_id` formatted like: "<user_id>_<post_index>"
  e.g., "PgZVTC_177_1" where the user_id is "PgZVTC_177" (the prefix before the last underscore).
- The dataset has at least: doc_id, text, label (others are kept and passed through).

Outputs
-------
- <outdir>/train_user_split.parquet
- <outdir>/test_user_split.parquet
- (optional) <outdir>/val_user_split.parquet  (if --val-size > 0)
- <outdir>/split_users_train.txt, split_users_val.txt (if applicable), split_users_test.txt
- A printed summary with sanity checks (row counts, unique users, overlap = 0).

Usage
--------
# Simple 80/20 train/test split
python scripts/split_datasets_by_users.py \
  --input data/processed/eRisk25_clean.parquet \
  --outdir data/processed \
  --test-size 0.3 \
  --val-size 0.0 \
  --seed 42
"""

import argparse
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def extract_user_id_from_doc_id(doc_id: str) -> str:
    """
    Extract user_id from a doc_id of the form "<user_id>_<post_index>".
    We take everything before the LAST underscore.

    Examples
    --------
    "PgZVTC_177_1" -> "PgZVTC_177"
    "abc_def_99_42" -> "abc_def_99"
    """
    if not isinstance(doc_id, str):
        raise ValueError(f"doc_id must be a string, got {type(doc_id)}: {doc_id}")
    if "_" not in doc_id:
        # Fall back to entire doc_id if no underscore found, but warn
        return doc_id
    return doc_id.rsplit("_", 1)[0]


def add_user_id_column(df: pd.DataFrame) -> pd.DataFrame:
    if "doc_id" not in df.columns:
        raise KeyError("Input DataFrame must contain a 'doc_id' column.")
    df = df.copy()
    df["user_id"] = df["doc_id"].apply(extract_user_id_from_doc_id)
    return df


def split_users(
    users: np.ndarray,
    test_size: float,
    val_size: float,
    seed: int,
    stratify_labels_for_users: Optional[pd.Series] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """
    Split the array of unique users into train/(optional val)/test user lists.

    If val_size == 0, returns (train_users, None, test_users).
    If stratify_labels_for_users is provided (Series indexed by user_id),
    we will attempt user-level stratification on the coarse label attached to each user.
    """
    # Handle (train + val) vs test first
    if stratify_labels_for_users is not None:
        # Align to `users` order
        y = stratify_labels_for_users.reindex(users).values
    else:
        y = None

    # First, hold out TEST
    users_trainval, users_test = train_test_split(
        users,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
        stratify=y if y is not None else None,
    )

    if val_size and val_size > 0:
        # Compute relative val size within the remaining (train+val) pool
        rel_val = val_size / (1.0 - test_size)
        y_trainval = None
        if y is not None:
            # Recompute y for the remaining pool
            series = pd.Series(y, index=users)
            y_trainval = series.reindex(users_trainval).values

        users_train, users_val = train_test_split(
            users_trainval,
            test_size=rel_val,
            random_state=seed,
            shuffle=True,
            stratify=y_trainval if y_trainval is not None else None,
        )
        return users_train, users_val, users_test
    else:
        return users_trainval, None, users_test


def optionally_build_user_labels_for_stratification(
    df: pd.DataFrame,
    label_col: Optional[str],
) -> Optional[pd.Series]:
    """
    Build a user-level label for stratification, if label_col exists.
    Strategy: use the MAJORITY class label per user (ties broken by smallest label value).
    Returns a pd.Series indexed by user_id with a single label per user.
    """
    if label_col is None or label_col not in df.columns:
        return None

    # Ensure categorical/integer class labels work cleanly
    # We'll treat labels as strings for grouping, then try to cast back if possible.
    g = df.groupby("user_id")[label_col]
    # majority vote per user
    user_majority = (
        g.apply(lambda s: s.value_counts().sort_values(ascending=False).index[0])
        .astype(str)
    )

    # Try to cast back to int if possible
    try:
        user_majority = user_majority.astype(int)
    except Exception:
        pass

    return user_majority


def main():
    parser = argparse.ArgumentParser(description="Split dataset by user_id with no overlap.")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input parquet (e.g., data/processed/eRisk25_clean.parquet)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="data/processed",
        help="Directory to write outputs (parquet + user lists).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of USERS to allocate to test (default: 0.2).",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.0,
        help="Fraction of USERS to allocate to validation (taken from the non-test pool). Set 0 to skip.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--label-col",
        type=str,
        default="label",
        help="Optional label column for user-level stratification, if present (default: 'label').",
    )
    parser.add_argument(
        "--no-stratify",
        action="store_true",
        help="Disable user-level stratification even if label column exists.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ---------- Load ----------
    print(f"[INFO] Loading: {input_path}")
    df = pd.read_parquet(input_path)

    # ---------- Derive user_id ----------
    print("[INFO] Deriving user_id from doc_id ...")
    df = add_user_id_column(df)

    # ---------- Build optional user-level stratification labels ----------
    user_level_labels = None
    if not args.no_stratify and args.label_col in df.columns:
        print("[INFO] Building user-level labels for stratification (majority label per user) ...")
        user_level_labels = optionally_build_user_labels_for_stratification(df, args.label_col)
    else:
        if args.no_stratify:
            print("[INFO] Stratification disabled by flag.")
        else:
            print(f"[INFO] Label column '{args.label_col}' not found. Proceeding without stratification.")

    # ---------- Split users ----------
    unique_users = df["user_id"].dropna().unique()
    unique_users = np.array(sorted(unique_users))

    print(
        f"[INFO] Splitting USERS -> test_size={args.test_size:.3f}, "
        f"val_size={args.val_size:.3f}, seed={args.seed}"
    )

    train_users, val_users, test_users = split_users(
        unique_users,
        test_size=args.test_size,
        val_size=args.val_size,
        seed=args.seed,
        stratify_labels_for_users=user_level_labels if user_level_labels is not None else None,
    )

    # ---------- Build dataframes ----------
    train_df = df[df["user_id"].isin(train_users)].reset_index(drop=True)
    test_df = df[df["user_id"].isin(test_users)].reset_index(drop=True)
    val_df = None
    if val_users is not None:
        val_df = df[df["user_id"].isin(val_users)].reset_index(drop=True)

    # ---------- Sanity checks ----------
    overlap_train_test = set(train_df["user_id"]).intersection(set(test_df["user_id"]))
    if overlap_train_test:
        raise RuntimeError(f"User overlap found between TRAIN and TEST: {len(overlap_train_test)} users")

    if val_df is not None:
        overlap_train_val = set(train_df["user_id"]).intersection(set(val_df["user_id"]))
        overlap_val_test = set(val_df["user_id"]).intersection(set(test_df["user_id"]))
        if overlap_train_val or overlap_val_test:
            raise RuntimeError(
                f"User overlap in validation split: train∩val={len(overlap_train_val)}, val∩test={len(overlap_val_test)}"
            )

    # ---------- Save ----------
    train_out = outdir / "train_user_split.parquet"
    test_out = outdir / "test_user_split.parquet"
    val_out = outdir / "val_user_split.parquet" if val_df is not None else None

    print(f"[INFO] Saving train -> {train_out}")
    train_df.to_parquet(train_out, index=False)

    if val_df is not None:
        print(f"[INFO] Saving val   -> {val_out}")
        val_df.to_parquet(val_out, index=False)

    print(f"[INFO] Saving test  -> {test_out}")
    test_df.to_parquet(test_out, index=False)

    # Save user lists for reproducibility/auditing
    (outdir / "split_users_train.txt").write_text("\n".join(map(str, sorted(set(train_users)))), encoding="utf-8")
    (outdir / "split_users_test.txt").write_text("\n".join(map(str, sorted(set(test_users)))), encoding="utf-8")
    if val_users is not None:
        (outdir / "split_users_val.txt").write_text("\n".join(map(str, sorted(set(val_users)))), encoding="utf-8")

    # ---------- Summary ----------
    def fmt_counts(name: str, part_df: pd.DataFrame):
        n_rows = len(part_df)
        n_users = part_df["user_id"].nunique()
        labels = part_df[args.label_col].value_counts(dropna=False).to_dict() if args.label_col in part_df.columns else {}
        return f"{name:>5}: rows={n_rows:6d} | users={n_users:5d} | label_dist={labels}"

    print("\n=== Split Summary (by posts) ===")
    print(fmt_counts("TRAIN", train_df))
    if val_df is not None:
        print(fmt_counts("  VAL", val_df))
    print(fmt_counts(" TEST", test_df))

    # Final overlap assert (paranoid)
    assert set(train_df["user_id"]).isdisjoint(set(test_df["user_id"]))
    if val_df is not None:
        assert set(train_df["user_id"]).isdisjoint(set(val_df["user_id"]))
        assert set(val_df["user_id"]).isdisjoint(set(test_df["user_id"]))

    print("\n[OK] Done. No user overlap across splits.")

if __name__ == "__main__":
    main()
