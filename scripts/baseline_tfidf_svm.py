# scripts/baseline_tfidf_svm.py

import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, f1_score, accuracy_score

def parse_args():
    ap = argparse.ArgumentParser(description="Char TF-IDF + LinearSVC (message-level)")
    ap.add_argument("--csv", required=True, help="Path to CSV or Parquet")
    ap.add_argument("--text_col", default="text", help="Text column")
    ap.add_argument("--label_col", default="label", help="Binary label column (0/1)")
    ap.add_argument("--test_size", type=float, default=0.30, help="Test split size")
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--min_df", type=int, default=5)
    ap.add_argument("--max_df", type=float, default=0.9)
    ap.add_argument("--results_dir", default="results")
    return ap.parse_args()

def load_df(path: Path):
    suf = path.suffix.lower()
    if suf == ".csv":
        return pd.read_csv(path)
    elif suf == ".parquet":
        return pd.read_parquet(path)
    else:
        raise SystemExit(f"Unsupported file type: {suf} (use .csv or .parquet)")

def main():
    args = parse_args()
    path = Path(args.csv)
    if not path.exists():
        raise SystemExit(f"File not found: {path.resolve()}")

    df = load_df(path)

    for c in [args.text_col, args.label_col]:
        if c not in df.columns:
            raise SystemExit(f"Missing column '{c}'. Found: {list(df.columns)}")

    # Drop empty rows
    df = df.dropna(subset=[args.text_col, args.label_col])
    df = df[df[args.text_col].astype(str).str.strip() != ""]

    X = df[args.text_col].astype(str).values
    y = df[args.label_col].values

    # Stratified split (keeps label balance), no user grouping
    sss = StratifiedShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.random_state)
    train_idx, test_idx = next(sss.split(X, y))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Character TF-IDF (great for short/noisy chat)
    vect = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=args.min_df,
        max_df=args.max_df,
        sublinear_tf=True,
        smooth_idf=True,
    )

    clf = LinearSVC(class_weight="balanced", C=1.0)  # strong linear baseline on sparse TF-IDF

    Xtr = vect.fit_transform(X_train)
    Xte = vect.transform(X_test)
    clf.fit(Xtr, y_train)

    y_pred = clf.predict(Xte)
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    report = classification_report(y_test, y_pred, digits=3)

    print("\n=== CHAR TF-IDF + LinearSVC (message-level) ===")
    print(f"Accuracy : {acc:.3f}")
    print(f"Macro-F1 : {macro_f1:.3f}")
    print(report)

    # Save outputs
    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "svm_char_report.txt").write_text(
        f"Accuracy: {acc:.4f}\nMacro-F1: {macro_f1:.4f}\n\n{report}",
        encoding="utf-8",
    )
    pd.DataFrame({"y_true": y_test, "y_pred": y_pred}).to_csv(out_dir / "svm_char_predictions.csv", index=False)

if __name__ == "__main__":
    main()

