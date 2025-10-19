#!/usr/bin/env python3
"""
Usage:
  python clean_eRisk.py \
      --in data/raw/erisk25-t1-dataset \
      --qrels data/raw/qrels_consensus_merged.csv \
      --out data/processed/erisk25_clean.parquet
"""

from __future__ import annotations
import re
import argparse
import sys
from pathlib import Path
import pandas as pd

# .trec parsing
DOC_BLOCK_RE = re.compile(r"<DOC>(.*?)</DOC>", flags=re.S)

def _extract_tag(tag: str, block: str) -> str:
    m = re.search(rf"<{tag}>(.*?)</{tag}>", block, flags=re.S)
    return m.group(1).strip() if m else ""

def parse_trec_file(path: Path) -> pd.DataFrame:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    docs = DOC_BLOCK_RE.findall(text)

    rows = []
    for block in docs:
        doc_id = _extract_tag("DOCNO", block)
        pre = _extract_tag("PRE", block)
        main_text = _extract_tag("TEXT", block)
        if not doc_id:
            continue
        rows.append({
            "doc_id": doc_id,
            "pre": pre,
            "text": main_text,
        })

    return pd.DataFrame(rows)

def load_all_trec(folder: Path) -> pd.DataFrame:
    all_files = sorted(Path(folder).glob("*.trec"))
    if not all_files:
        print(f"[ERROR] No .trec files found in: {folder}", file=sys.stderr)
        sys.exit(2)
    dfs = [parse_trec_file(f) for f in all_files]
    return pd.concat(dfs, ignore_index=True)

# qrels loading/merging
def load_qrels(qrels_path: Path) -> pd.DataFrame:
    qrels = pd.read_csv(qrels_path)

    if "label" in qrels.columns:
        pass
    elif "relevant" in qrels.columns:
        qrels["label"] = qrels["relevant"].map({True: 1, False: 0, 1: 1, 0: 0}).astype(int)
    else:
        raise ValueError("qrels must contain either a 'label' column or a 'relevant' column")

    if "doc_id" not in qrels.columns:
        if "DOCNO" in qrels.columns:
            qrels = qrels.rename(columns={"DOCNO": "doc_id"})
        else:
            raise ValueError("qrels must contain a 'doc_id' (or 'DOCNO') column to merge with posts")

    return qrels

def merge_with_qrels(df_text: pd.DataFrame, qrels: pd.DataFrame) -> pd.DataFrame:
    merged = df_text.merge(qrels[["doc_id", "label"]], on="doc_id", how="inner")
    return merged

# minimal post-processing
def finalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["text"]).drop_duplicates(subset=["doc_id"])
    df["num_tokens"] = df["text"].apply(lambda t: len(str(t).split()))
    return df[["doc_id", "pre", "text", "label", "num_tokens"]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="input", required=True, help="Folder containing .trec files")
    parser.add_argument("--qrels", required=True, help="Path to qrels CSV file")
    parser.add_argument("--out", dest="output", required=True, help="Output path (.csv or .parquet)")
    args = parser.parse_args()

    trec_dir = Path(args.input)
    qrels_path = Path(args.qrels)

    print(f"Loading .trec files from {trec_dir}")
    df_text = load_all_trec(trec_dir)
    print(f"  → {len(df_text):,} documents loaded")

    print(f"Loading qrels from {qrels_path}")
    qrels = load_qrels(qrels_path)
    print(f"  → {len(qrels):,} qrels rows")

    print("Merging posts with labels…")
    df_merged = merge_with_qrels(df_text, qrels)
    print(f"  → {len(df_merged):,} labeled documents after merge")

    print("Finalizing (keep PRE, drop POST, keep raw text, add num_tokens)…")
    df_final = finalize(df_merged)

    out_path = Path(args.output)
    print(f"Saving to {out_path}")
    if out_path.suffix.lower() == ".parquet":
        df_final.to_parquet(out_path, index=False)
    else:
        df_final.to_csv(out_path, index=False)

    print("Done.")

if __name__ == "__main__":
    main()
