"""Audit the held-out evaluation behind the reported test F1.

Checks, all on the fixed split in data/splits/split_indices.json:
  1. duplicates inside the dataset and overlap between train and val/test
  2. the TF-IDF + logistic-regression baseline (same settings as train_baseline.py)
  3. the deployed RoBERTa run: confusion matrix, confidence of its errors
  4. a small set of hand-written, school-style probe sentences (illustrative only)

Writes results/audit/audit.json and results/audit/test_errors.csv.

Usage (from research/):
  python scripts/audit_evaluation.py --data Suicide_Detection.csv \
      --model path/to/roberta-seed42/final_model
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]

# (expected label, text). Written for this audit, not drawn from any dataset
# or from student submissions. Too few to estimate a rate; they only show
# the kinds of input the Reddit test set doesn't cover.
PROBES = [
    (0, "This chemistry test is going to kill me, I have not studied at all."),
    (0, "In Romeo and Juliet, Juliet takes her own life after finding Romeo dead, which shows how the feud destroys both families."),
    (0, "My essay argues that schools should expand access to counselors because teen suicide rates have risen over the past decade."),
    (0, "I was so embarrassed when I tripped in front of the whole class that I wanted to disappear."),
    (0, "Over the summer I volunteered at a food bank and learned how much planning goes into feeding a community."),
    (1, "Lately I don't really see the point of anything. I sleep all day and I think everyone would be better off without me."),
    (1, "Sometimes I wish I could just go to sleep and not wake up."),
    (1, "I have been giving my things away to friends because I will not need them soon."),
    (1, "In this personal narrative I want to write about the night last spring when I decided I was done and wrote a goodbye letter."),
    (1, "honestly lol i kinda want to die haha anyway whats the homework"),
    (1, "I want to kill myself."),
    (1, "I don't want to be alive anymore."),
]


def rates(y, p):
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    return {
        "accuracy": (tp + tn) / len(y),
        "f1": 2 * prec * rec / (prec + rec),
        "fnr": fn / (fn + tp),
        "fpr": fp / (fp + tn),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


def load(path):
    # Same cleaning as src/dataset.py, so the split indices line up.
    df = pd.read_csv(path, dtype={"text": str, "class": str}).dropna()
    df["label"] = df["class"].map({"non-suicide": 0, "suicide": 1})
    df = df.dropna(subset=["label"]).reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", default="roberta-base")
    ap.add_argument("--out", default=str(ROOT / "results" / "audit"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = load(args.data)
    splits = json.loads((ROOT / "data" / "splits" / "split_indices.json").read_text())
    idx = {k: splits[f"{k}_indices"] for k in ("train", "val", "test")}
    tr, va, te = (df.loc[idx[k]] for k in ("train", "val", "test"))
    report = {"rows": len(df), "class_counts": df.label.value_counts().sort_index().tolist()}

    # 1. duplicates
    norm = df.text.map(lambda t: re.sub(r"\s+", " ", t.lower()).strip())
    train_norm = set(norm.loc[idx["train"]])
    groups = pd.DataFrame({"n": norm, "label": df.label}).groupby("n").label.nunique()
    report["duplicates"] = {
        "exact_duplicate_rows": int(df.text.duplicated().sum()),
        "normalized_duplicate_rows": int(norm.duplicated().sum()),
        "normalized_texts_with_conflicting_labels": int((groups > 1).sum()),
        "val_rows_also_in_train": int(norm.loc[idx["val"]].isin(train_norm).sum()),
        "test_rows_also_in_train": int(norm.loc[idx["test"]].isin(train_norm).sum()),
    }

    # Scraping artifact: line breaks dropped so sentences run together ("...meI").
    glued = df.text.str.contains(r"[a-z](?:I|I'm)\s", regex=True)
    report["glued_line_share_by_class"] = {
        "non_suicide": float(glued[df.label == 0].mean()),
        "suicide": float(glued[df.label == 1].mean()),
    }

    # 2. baseline
    vec = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=2, max_df=0.95, stop_words="english")
    clf = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced", solver="liblinear")
    clf.fit(vec.fit_transform(tr.text), tr.label)
    report["tfidf_logreg"] = {
        "val": rates(va.label.values, clf.predict(vec.transform(va.text))),
        "test": rates(te.label.values, clf.predict(vec.transform(te.text))),
    }
    order = np.argsort(clf.coef_[0])
    names = vec.get_feature_names_out()
    report["tfidf_logreg"]["top_features"] = {
        "non_suicide": names[order[:25]].tolist(),
        "suicide": names[order[-25:][::-1]].tolist(),
    }

    # 3. RoBERTa on the test set
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).eval().to(device)

    def predict(texts, bs=64):
        probs = []
        with torch.no_grad():
            for i in range(0, len(texts), bs):
                batch = tok(texts[i:i + bs], truncation=True, max_length=256, padding=True, return_tensors="pt").to(device)
                probs += torch.softmax(model(**batch).logits.float(), -1)[:, 1].cpu().tolist()
        return np.array(probs)

    p = predict(te.text.tolist())
    pred = (p > 0.5).astype(int)
    y = te.label.values
    err = (pred != y)
    words = te.text.str.split().str.len().values
    report["roberta_test"] = rates(y, pred) | {
        "false_negatives_with_p_below_0.05": int(((y == 1) & (pred == 0) & (p < 0.05)).sum()),
        "false_positives_with_p_above_0.95": int(((y == 0) & (pred == 1) & (p > 0.95)).sum()),
        "share_of_test_with_p_between_0.1_and_0.9": float(((p > 0.1) & (p < 0.9)).mean()),
        "median_words_all": float(np.median(words)),
        "median_words_errors": float(np.median(words[err])),
    }
    pd.DataFrame({"row": te.index[err], "label": y[err], "p_suicide": p[err], "text": te.text.values[err]}).to_csv(
        out / "test_errors.csv", index=False
    )

    # 4. probes
    probe_p = predict([t for _, t in PROBES])
    report["probes"] = [
        {"expected": e, "p_suicide": round(float(q), 3), "text": t} for (e, t), q in zip(PROBES, probe_p)
    ]

    (out / "audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "probes"}, indent=2))
    for r in report["probes"]:
        print(r["expected"], r["p_suicide"], r["text"])


if __name__ == "__main__":
    main()
