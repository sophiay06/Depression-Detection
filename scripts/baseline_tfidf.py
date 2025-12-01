# TF-IDF + Logistic Regression baseline

import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, accuracy_score


def parse_args():
    ap = argparse.ArgumentParser(description="TF-IDF + Logistic Regression (message-level)")
    ap.add_argument("--csv", required=True, help="Processed CSV or Parquet file")
    ap.add_argument("--text_col", default="text", help="Column containing message text")
    ap.add_argument("--label_col", default="label", help="Column containing binary label (0/1)")
    ap.add_argument("--test_size", type=float, default=0.30)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--min_df", type=int, default=5)
    ap.add_argument("--max_df", type=float, default=0.9)
    ap.add_argument(
        "--ngrams",
        type=int,
        default=2,
        help="max ngram length (1=unigrams, 2=uni+bi, etc.)",
    )
    ap.add_argument("--results_dir", default="results", help="Where to save results")
    return ap.parse_args()


def load_dataframe(path: Path):
    """Load CSV or Parquet automatically based on extension."""
    suf = path.suffix.lower()
    if suf == ".csv":
        return pd.read_csv(path)
    elif suf == ".parquet":
        return pd.read_parquet(path)
    else:
        raise SystemExit(f"Unsupported file type: {suf} (use .csv or .parquet)")


def main():
    args = parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"File not found: {csv_path.resolve()}")

    df = load_dataframe(csv_path)

    for col in [args.text_col, args.label_col]:
        if col not in df.columns:
            raise SystemExit(f"Missing column: '{col}'. Found columns: {list(df.columns)}")

    df = df.dropna(subset=[args.text_col, args.label_col])
    df = df[df[args.text_col].astype(str).str.strip() != ""]

    X = df[args.text_col].astype(str).values
    y = df[args.label_col].values

    # stratified split
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    train_idx, test_idx = next(splitter.split(X, y))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Word-level TF-IDF
    vect = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words="english",
        ngram_range=(1, args.ngrams),
        min_df=args.min_df,
        max_df=args.max_df,
        sublinear_tf=True,
        smooth_idf=True,
    )

    # Modeling
    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        n_jobs=None,
    )

    # Training
    Xtr = vect.fit_transform(X_train)
    Xte = vect.transform(X_test)
    clf.fit(Xtr, y_train)

    # Predicting
    y_pred = clf.predict(Xte)
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    report = classification_report(y_test, y_pred, digits=3)

    print("\n=== TF-IDF + Logistic Regression (message-level, non-author-aware) ===")
    print(f"Accuracy : {acc:.3f}")
    print(f"Macro-F1 : {macro_f1:.3f}")
    print(report)

    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tfidf_report.txt").write_text(
        f"Accuracy: {acc:.4f}\nMacro-F1: {macro_f1:.4f}\n\n{report}",
        encoding="utf-8",
    )

    pd.DataFrame({"y_true": y_test, "y_pred": y_pred}).to_csv(
        out_dir / "tfidf_predictions.csv", index=False
    )

    # Saving vectorizer and model
    try:
        import joblib

        joblib.dump(vect, out_dir / "tfidf_vectorizer.joblib")
        joblib.dump(clf, out_dir / "tfidf_logreg.joblib")
    except Exception as e:
        print(f"(Skipping model save) {e}")


if __name__ == "__main__":
    main()