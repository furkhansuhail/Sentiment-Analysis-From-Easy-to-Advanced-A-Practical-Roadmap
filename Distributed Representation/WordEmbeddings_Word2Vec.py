#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
IMDB 50K Sentiment: Word2Vec / GloVe / FastText + SVM
- User-selectable embeddings via CLI flag --embedding or interactive prompt.
- Word2Vec: train on corpus OR load pretrained (e.g., GoogleNews .bin).
- GloVe: load .txt; auto-convert to word2vec format if needed.
- FastText: train on corpus OR load Facebook pretrained (.bin/.vec).
- TF-IDF-weighted or mean-pooled document embeddings.
- LinearSVC with optional GridSearch and calibrated probabilities.
- Pickling-safe TF-IDF (no lambda) and sklearn version-safe calibration.
- Robust pretrained path resolution with helpful errors.
- Built-in example reviews + optional external examples file.
"""

import os
import re
import json
import argparse
import logging
import random
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# Text cleaning
from bs4 import BeautifulSoup

import nltk
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# ML
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, roc_auc_score, average_precision_score
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import sklearn

# Gensim: embeddings
from gensim.models import Word2Vec, FastText
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

from pathlib import Path

# ---- Example reviews to score after training ----
DEF_EXAMPLES = [
    "An exhilarating and heartfelt film. The performances felt honest and the ending stayed with me.",
    "What a waste of time. The plot was a mess and the dialogue was painful.",
    "Surprisingly good! I went in with low expectations and left smiling. Great pacing and music.",
    "Dull, predictable, and way too long. I kept checking the clock.",
    "A beautifully shot movie with real emotional weight. I teared up more than once.",
    "The jokes never landed and the lead actor looked bored. I couldn’t finish it.",
    "Not perfect, but the characters were rich and the story had charm.",
    "This is easily the worst sequel in the series—empty fan service and nothing else.",
    "Inventive direction and a stellar score elevate a familiar story into something special.",
    "Mediocre at best. It’s not terrible, but there’s nothing memorable either."
]

# -------------------------------
# Helpers / Fixes
# -------------------------------

def identity(x):
    """Top-level identity function so it's picklable (used by TfidfVectorizer)."""
    return x

def sklearn_at_least(major: int, minor: int) -> bool:
    parts = sklearn.__version__.split(".")
    try:
        maj = int(parts[0]); minr = int(parts[1])
    except Exception:
        return True
    return (maj > major) or (maj == major and minr >= minor)

def set_all_seeds(seed: int = 42):
    random.seed(seed); np.random.seed(seed)

def safe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)

def strip_html(text: str) -> str:
    return BeautifulSoup(text, "html.parser").get_text(separator=" ")

def clean_text(
    text: str,
    lowercase: bool = True,
    remove_stopwords_flag: bool = True,
    lemmatize_flag: bool = True,
    min_token_len: int = 1
) -> List[str]:
    if text is None:
        return []
    text = strip_html(text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    if lowercase:
        text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9'.,!?;:\s-]", " ", text)
    tokens = word_tokenize(text)
    if remove_stopwords_flag:
        sw = set(stopwords.words("english"))
        tokens = [t for t in tokens if t not in sw]
    if lemmatize_flag:
        lemm = WordNetLemmatizer()
        tokens = [lemm.lemmatize(t) for t in tokens]
    tokens = [t for t in tokens if len(t) >= min_token_len]
    return tokens

def resolve_existing_path(candidate: Optional[str], results_dir: Optional[str] = None) -> Tuple[Optional[Path], List[Path]]:
    """
    Resolve a file path robustly:
    - Expands ~ and environment variables
    - If relative or not found, tries: CWD, script dir, script parent, results_dir
    Returns (found_path_or_None, tried_candidates_list).
    """
    tried: List[Path] = []
    if not candidate:
        return None, tried

    # Base expansions
    p = Path(os.path.expanduser(os.path.expandvars(candidate)))

    # Where are we?
    try:
        script_dir = Path(__file__).resolve().parent
    except Exception:
        script_dir = Path.cwd()

    # Candidate list
    candidates = [p]
    if not p.is_absolute() or not p.exists():
        candidates += [
            Path.cwd() / candidate,
            script_dir / candidate,
            script_dir.parent / candidate,
        ]
        if results_dir:
            candidates.append(Path(results_dir) / candidate)

    for c in candidates:
        tried.append(c)
        if c.exists():
            return c, tried

    return None, tried

def _truncate(s: str, n: int = 80) -> str:
    return (s[:n] + "…") if len(s) > n else s

def print_predictions(preds: List[Dict], max_preview: int = 80):
    """
    Pretty print model predictions returned by runner.predict_texts.
    Handles calibrated and non-calibrated models.
    """
    print("\n=== Example Predictions ===")
    header = "{:>2}  {:<8}  {:>7}  {}".format("#", "label", "prob+", "review_preview")
    print(header)
    print("-" * len(header))
    for i, p in enumerate(preds, 1):
        label = p.get("pred_sentiment", str(p.get("pred_label", "")))
        prob = p.get("prob_positive", None)
        prob_s = f"{prob:.3f}" if prob is not None else "-"
        print("{:>2}  {:<8}  {:>7}  {}".format(i, label, prob_s, _truncate(p["text"], max_preview)))

def load_reviews_from_file(path: str) -> List[str]:
    """
    Load example reviews from a file:
      - .txt: one review per line (blank lines ignored)
      - .json: a JSON array of strings
      - .csv: first column or column named 'review'
    """
    path = str(path)
    if path.lower().endswith(".txt"):
        with open(path, "r", encoding="utf-8") as f:
            return [ln.strip() for ln in f if ln.strip()]
    elif path.lower().endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("JSON must be an array of strings")
        return [str(x).strip() for x in data if str(x).strip()]
    elif path.lower().endswith(".csv"):
        df = pd.read_csv(path)
        if "review" in df.columns:
            col = df["review"].astype(str).tolist()
        else:
            col = df.iloc[:, 0].astype(str).tolist()
        return [x.strip() for x in col if x and x.strip()]
    else:
        raise ValueError("Unsupported file type. Use .txt, .json, or .csv")

# -------------------------------
# Config
# -------------------------------
@dataclass
class RunConfig:
    data_path: str = "IMDB Dataset.csv"
    text_col: str = "review"
    label_col: str = "sentiment"
    test_size: float = 0.2
    random_state: int = 42
    results_dir: str = "results_embed_svm"

    # Preprocessing
    lowercase: bool = True
    remove_stopwords: bool = True
    lemmatize: bool = True
    min_token_len: int = 1

    # Embedding choice
    embedding: str = "word2vec"            # word2vec | glove | fasttext

    # Word2Vec / FastText training
    vector_size: int = 300
    window: int = 5
    min_count: int = 2
    sg: int = 1
    negative: int = 10          # Word2Vec only
    epochs: int = 8

    # Pretrained paths
    use_pretrained: bool = False
    pretrained_path: Optional[str] = None   # GoogleNews .bin; GloVe .txt; FastText .bin/.vec

    # Embedding aggregation
    embedding_agg: str = "tfidf"   # tfidf | mean

    # Demo / examples
    run_examples: bool = False
    examples_path: Optional[str] = None

    # Classifier
    c_values: Tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)
    grid_search: bool = True
    cv_folds: int = 5
    calibrate: bool = True

    # Misc
    n_jobs: int = -1
    verbose: int = 1

# -------------------------------
# Main class
# -------------------------------
class IMDBEmbedSVM:
    def __init__(self, cfg: RunConfig):
        self.cfg = cfg
        safe_mkdir(cfg.results_dir)

        # TF-IDF bits
        self.tfidf_vec: Optional[TfidfVectorizer] = None
        self.tfidf_vocab_idf: Optional[Dict[str, float]] = None

        # Embedding store
        self.kv: Optional[KeyedVectors] = None
        self.dim: int = cfg.vector_size

        # If we trained models (for save)
        self.w2v_model: Optional[Word2Vec] = None
        self.ft_model: Optional[FastText] = None

        # Classifier
        self.clf = None
        self.label_map = {"negative": 0, "positive": 1}

        # Logging
        logging.basicConfig(
            level=logging.INFO if cfg.verbose else logging.WARNING,
            format="%(asctime)s | %(levelname)s | %(message)s"
        )
        self.logger = logging.getLogger("IMDBEmbedSVM")

    # ---------- Data ----------
    def load_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.cfg.data_path)
        assert self.cfg.text_col in df.columns, f"Missing text column '{self.cfg.text_col}'"
        assert self.cfg.label_col in df.columns, f"Missing label column '{self.cfg.label_col}'"
        df["label"] = df[self.cfg.label_col].map(self.label_map)
        return df[[self.cfg.text_col, "label"]].dropna().reset_index(drop=True)

    def tokenize_corpus(self, texts: List[str]) -> List[List[str]]:
        out = []
        for t in tqdm(texts, desc="Tokenizing", leave=False):
            out.append(
                clean_text(
                    t,
                    lowercase=self.cfg.lowercase,
                    remove_stopwords_flag=self.cfg.remove_stopwords,
                    lemmatize_flag=self.cfg.lemmatize,
                    min_token_len=self.cfg.min_token_len,
                )
            )
        return out

    # ---------- TF-IDF ----------
    def fit_tfidf(self, tokenized_docs: List[List[str]]):
        self.tfidf_vec = TfidfVectorizer(
            analyzer="word",
            tokenizer=identity,
            preprocessor=identity,
            token_pattern=None,
            lowercase=False,
            min_df=2
        )
        self.tfidf_vec.fit(tokenized_docs)
        vocab = self.tfidf_vec.get_feature_names_out()
        idf = self.tfidf_vec.idf_
        self.tfidf_vocab_idf = {w: float(i) for w, i in zip(vocab, idf)}

    # ---------- Embeddings loaders/trainers ----------
    def _load_pretrained_word2vec(self):
        path, tried = resolve_existing_path(self.cfg.pretrained_path, self.cfg.results_dir)
        assert path and path.exists(), (
            "Pretrained Word2Vec .bin not found.\n"
            f"Tried: {[str(p) for p in tried]}\n"
            "Pass --pretrained_path to a valid file (e.g., GoogleNews-vectors-negative300.bin)."
        )
        self.logger.info(f"Loading pretrained Word2Vec: {path}")
        self.kv = KeyedVectors.load_word2vec_format(str(path), binary=True)
        self.dim = self.kv.vector_size

    def _train_word2vec(self, tokenized_docs: List[List[str]]):
        self.logger.info("Training Word2Vec on corpus...")
        self.w2v_model = Word2Vec(
            sentences=tokenized_docs,
            vector_size=self.cfg.vector_size,
            window=self.cfg.window,
            min_count=self.cfg.min_count,
            workers=os.cpu_count(),
            sg=self.cfg.sg,
            negative=self.cfg.negative,
            epochs=self.cfg.epochs,
        )
        self.kv = self.w2v_model.wv
        self.dim = self.kv.vector_size

    def _load_pretrained_glove(self):
        # Accept raw GloVe .txt and convert to word2vec format once
        path, tried = resolve_existing_path(self.cfg.pretrained_path, self.cfg.results_dir)
        assert path and path.exists(), (
            "GloVe .txt not found.\n"
            f"Provided: {self.cfg.pretrained_path}\n"
            f"Tried: {[str(p) for p in tried]}\n"
            "Tips:\n"
            "  • Ensure the file is UNZIPPED (e.g., glove.6B.300d.txt)\n"
            r"  • Use an absolute path on Windows, e.g. --pretrained_path ""F:\Embeddings\glove.6B.300d.txt"""
        )

        self.logger.info(f"Using GloVe file: {path}")

        w2v_txt = Path(self.cfg.results_dir) / "glove_as_w2v.txt"
        if not w2v_txt.exists():
            self.logger.info("Converting GloVe .txt to word2vec format...")
            glove2word2vec(str(path), str(w2v_txt))

        self.logger.info("Loading converted GloVe as KeyedVectors...")
        self.kv = KeyedVectors.load_word2vec_format(str(w2v_txt), binary=False)
        self.dim = self.kv.vector_size

    def _train_fasttext(self, tokenized_docs: List[List[str]]):
        self.logger.info("Training FastText on corpus...")
        self.ft_model = FastText(
            sentences=tokenized_docs,
            vector_size=self.cfg.vector_size,
            window=self.cfg.window,
            min_count=self.cfg.min_count,
            workers=os.cpu_count(),
            sg=self.cfg.sg,         # FastText supports sg too
            epochs=self.cfg.epochs,
        )
        self.kv = self.ft_model.wv
        self.dim = self.kv.vector_size

    def _load_pretrained_fasttext(self):
        from gensim.models.fasttext import load_facebook_vectors
        path, tried = resolve_existing_path(self.cfg.pretrained_path, self.cfg.results_dir)
        assert path and path.exists(), (
            "FastText pretrained vectors not found.\n"
            f"Provided: {self.cfg.pretrained_path}\n"
            f"Tried: {[str(p) for p in tried]}\n"
            "Provide --pretrained_path to a .bin or .vec file."
        )
        self.logger.info(f"Loading pretrained FastText: {path}")
        if str(path).endswith(".bin"):
            self.kv = load_facebook_vectors(str(path))     # returns KeyedVectors
        else:
            self.kv = KeyedVectors.load_word2vec_format(str(path), binary=False)
        self.dim = self.kv.vector_size

    def build_embeddings(self, tokenized_docs: List[List[str]]):
        emb = self.cfg.embedding.lower()
        if emb == "word2vec":
            if self.cfg.use_pretrained:
                self._load_pretrained_word2vec()
            else:
                self._train_word2vec(tokenized_docs)
        elif emb == "glove":
            # GloVe is expected pretrained .txt
            self._load_pretrained_glove()
        elif emb == "fasttext":
            if self.cfg.use_pretrained:
                self._load_pretrained_fasttext()
            else:
                self._train_fasttext(tokenized_docs)
        else:
            raise ValueError("embedding must be one of: word2vec, glove, fasttext")
        self.logger.info(f"Embeddings ready: type={emb}, dim={self.dim:,}, vocab={len(self.kv.index_to_key):,}")

    # ---------- Doc vectors ----------
    def _doc_embedding_mean(self, tokens: List[str]) -> np.ndarray:
        vecs = [self.kv[w] for w in tokens if w in self.kv]
        if not vecs:
            return np.zeros(self.dim, dtype=np.float32)
        return np.mean(vecs, axis=0)

    def _doc_embedding_tfidf(self, tokens: List[str]) -> np.ndarray:
        if not self.tfidf_vocab_idf:
            return self._doc_embedding_mean(tokens)
        weighted = []; weights = []
        for w in tokens:
            if w in self.kv:
                weight = self.tfidf_vocab_idf.get(w, 1.0)
                weighted.append(self.kv[w] * weight)
                weights.append(weight)
        if not weighted:
            return np.zeros(self.dim, dtype=np.float32)
        return np.sum(weighted, axis=0) / (np.sum(weights) + 1e-8)

    def docs_to_matrix(self, tokenized_docs: List[List[str]]) -> np.ndarray:
        X = np.zeros((len(tokenized_docs), self.dim), dtype=np.float32)
        use_tfidf = (self.cfg.embedding_agg.lower() == "tfidf")
        for i, toks in enumerate(tqdm(tokenized_docs, desc="Building embeddings", leave=False)):
            X[i] = self._doc_embedding_tfidf(toks) if use_tfidf else self._doc_embedding_mean(toks)
        return X

    # ---------- Classifier ----------
    def fit_classifier(self, X_train: np.ndarray, y_train: np.ndarray):
        base = LinearSVC(C=1.0, class_weight="balanced", random_state=self.cfg.random_state)

        if self.cfg.grid_search:
            params = {"C": list(self.cfg.c_values)}
            skf = StratifiedKFold(n_splits=self.cfg.cv_folds, shuffle=True, random_state=self.cfg.random_state)
            gs = GridSearchCV(
                estimator=base,
                param_grid=params,
                scoring="f1_macro",
                cv=skf,
                n_jobs=self.cfg.n_jobs,
                verbose=self.cfg.verbose
            )
            gs.fit(X_train, y_train)
            best_c = gs.best_params_["C"]
            self._save_json(os.path.join(self.cfg.results_dir, "grid_search.json"),
                            {"best_params": gs.best_params_, "best_score": float(gs.best_score_)})
            self.logger.info(f"GridSearch best C={best_c}, best f1_macro={gs.best_score_:.4f}")
            base = LinearSVC(C=best_c, class_weight="balanced", random_state=self.cfg.random_state)

        if self.cfg.calibrate:
            if sklearn_at_least(1, 2):
                self.clf = CalibratedClassifierCV(estimator=base, method="sigmoid", cv=5)
            else:
                self.clf = CalibratedClassifierCV(base_estimator=base, method="sigmoid", cv=5)
        else:
            self.clf = base

        self.clf.fit(X_train, y_train)

    # ---------- Evaluation ----------
    def evaluate(self, X: np.ndarray, y: np.ndarray, split_name: str, save_prefix: str):
        y_pred = self.clf.predict(X)
        acc = accuracy_score(y, y_pred)
        f1m = f1_score(y, y_pred, average="macro")

        report = classification_report(y, y_pred, target_names=["negative", "positive"], digits=4)
        cm = confusion_matrix(y, y_pred).tolist()

        out = {
            "split": split_name,
            "accuracy": float(acc),
            "f1_macro": float(f1m),
            "confusion_matrix": cm,
            "report": report
        }
        if hasattr(self.clf, "predict_proba"):
            prob = self.clf.predict_proba(X)[:, 1]
            try:
                out["roc_auc"] = float(roc_auc_score(y, prob))
                out["pr_auc"] = float(average_precision_score(y, prob))
            except Exception:
                pass

        self._save_json(os.path.join(self.cfg.results_dir, f"{save_prefix}_metrics.json"), out)
        self.logger.info(f"[{split_name}] acc={acc:.4f} | f1_macro={f1m:.4f}")
        self.logger.info(f"Classification report:\n{report}")

    # ---------- Persistence ----------
    def save_artifacts(self):
        # Classifier
        joblib.dump(self.clf, os.path.join(self.cfg.results_dir, f"{self.cfg.embedding}_classifier.joblib"))

        # TF-IDF vectorizer + compact idf map
        if self.tfidf_vec is not None:
            joblib.dump(self.tfidf_vec, os.path.join(self.cfg.results_dir, f"{self.cfg.embedding}_tfidf.joblib"))
        if self.tfidf_vocab_idf is not None:
            joblib.dump(self.tfidf_vocab_idf, os.path.join(self.cfg.results_dir, f"{self.cfg.embedding}_tfidf_idf_map.joblib"))

        # Embeddings
        if self.cfg.embedding == "word2vec":
            if self.w2v_model is not None:
                self.w2v_model.wv.save(os.path.join(self.cfg.results_dir, "word2vec.kv"))
            elif self.kv is not None:
                self.kv.save(os.path.join(self.cfg.results_dir, "word2vec.kv"))
        elif self.cfg.embedding == "glove":
            if self.kv is not None:
                self.kv.save(os.path.join(self.cfg.results_dir, "glove.kv"))
        elif self.cfg.embedding == "fasttext":
            if self.ft_model is not None:
                self.ft_model.wv.save(os.path.join(self.cfg.results_dir, "fasttext.kv"))
            elif self.kv is not None:
                self.kv.save(os.path.join(self.cfg.results_dir, "fasttext.kv"))

        # Configs
        self._save_json(os.path.join(self.cfg.results_dir, f"{self.cfg.embedding}_config.json"), asdict(self.cfg))
        self._save_json(os.path.join(self.cfg.results_dir, f"{self.cfg.embedding}_label_map.json"), self.label_map)

        self.logger.info(f"Saved artifacts to: {self.cfg.results_dir}")

    def load_artifacts(self):
        # (Optional) Implement if you want reload by embedding type
        pass

    # ---------- Inference ----------
    def predict_texts(self, texts: List[str]) -> List[Dict]:
        toks = self.tokenize_corpus(texts)
        X = self.docs_to_matrix(toks)
        y_pred = self.clf.predict(X)
        out = []
        if hasattr(self.clf, "predict_proba"):
            probs = self.clf.predict_proba(X)[:, 1]
            for t, p, pr in zip(texts, y_pred, probs):
                out.append({
                    "text": t,
                    "pred_label": int(p),
                    "pred_sentiment": "positive" if p == 1 else "negative",
                    "prob_positive": float(pr)
                })
        else:
            for t, p in zip(texts, y_pred):
                out.append({
                    "text": t,
                    "pred_label": int(p),
                    "pred_sentiment": "positive" if p == 1 else "negative",
                })
        return out

    # ---------- Helpers ----------
    def _save_json(self, path: str, data: dict):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    # ---------- Orchestration ----------
    def run(self, demo_prompt: bool = True):
        set_all_seeds(self.cfg.random_state)
        df = self.load_data()

        X_train_txt, X_test_txt, y_train, y_test = train_test_split(
            df[self.cfg.text_col].tolist(),
            df["label"].values,
            test_size=self.cfg.test_size,
            random_state=self.cfg.random_state,
            stratify=df["label"].values
        )

        # Tokenize
        train_tok = self.tokenize_corpus(X_train_txt)
        test_tok  = self.tokenize_corpus(X_test_txt)

        # TF-IDF (for weighted embeddings)
        if self.cfg.embedding_agg.lower() == "tfidf":
            self.fit_tfidf(train_tok)

        # Build embeddings per user choice
        self.build_embeddings(train_tok)

        # Build doc vectors
        X_train = self.docs_to_matrix(train_tok)
        X_test  = self.docs_to_matrix(test_tok)

        # Train classifier
        self.fit_classifier(X_train, y_train)

        # Evaluate
        self.evaluate(X_train, y_train, "train", f"{self.cfg.embedding}_train")
        self.evaluate(X_test, y_test, "test", f"{self.cfg.embedding}_test")

        # Save
        self.save_artifacts()

        # ---- Batch examples (file or built-in) ----
        if self.cfg.examples_path:
            try:
                reviews = load_reviews_from_file(self.cfg.examples_path)
                if reviews:
                    preds = self.predict_texts(reviews)
                    print_predictions(preds)
                else:
                    print(f"[WARN] No reviews found in {self.cfg.examples_path}")
            except Exception as e:
                print(f"[ERROR] Could not load examples from file: {e}")
        elif self.cfg.run_examples:
            preds = self.predict_texts(DEF_EXAMPLES)
            print_predictions(preds)

        # Optional quick single-input demo
        if demo_prompt:
            print("\nTry a quick review (or press Enter to skip):")
            try:
                user_inp = input().strip()
            except EOFError:
                user_inp = ""
            if user_inp:
                preds = self.predict_texts([user_inp])
                print(json.dumps(preds[0], indent=2))


# -------------------------------
# CLI
# -------------------------------
def build_argparser():
    p = argparse.ArgumentParser(description="IMDB Sentiment with Word2Vec/GloVe/FastText + SVM")
    p.add_argument("--data_path", type=str, default="IMDB Dataset.csv")
    p.add_argument("--results_dir", type=str, default="results_embed_svm")
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--random_state", type=int, default=42)

    # Preproc
    p.add_argument("--no_lowercase", action="store_true")
    p.add_argument("--no_stop", action="store_true")
    p.add_argument("--no_lemmatize", action="store_true")
    p.add_argument("--min_token_len", type=int, default=1)

    # Embedding selection
    p.add_argument("--embedding", type=str, default=None, choices=["word2vec","glove","fasttext"],
                   help="If omitted, you will be prompted interactively.")
    p.add_argument("--embedding_agg", type=str, default="tfidf", choices=["tfidf", "mean"])

    # Train vs Pretrained
    p.add_argument("--use_pretrained", action="store_true",
                   help="For word2vec (GoogleNews) or fasttext (FB). GloVe is always pretrained.")
    p.add_argument("--pretrained_path", type=str, default=None,
                   help="Path to .bin/.vec/.txt depending on embedding type.")

    # Common training params
    p.add_argument("--vector_size", type=int, default=300)
    p.add_argument("--window", type=int, default=5)
    p.add_argument("--min_count", type=int, default=2)
    p.add_argument("--sg", type=int, default=1)
    p.add_argument("--negative", type=int, default=10)  # W2V only
    p.add_argument("--epochs", type=int, default=8)

    # Examples / demo
    p.add_argument("--run_examples", action="store_true",
                   help="After training, run 10 built-in example reviews (DEF_EXAMPLES).")
    p.add_argument("--examples_path", type=str, default=None,
                   help="Path to a txt/json/csv file with reviews to score (one per line or 'review' column).")

    # Classifier
    p.add_argument("--grid_search", action="store_true")
    p.add_argument("--no_grid_search", dest="grid_search", action="store_false")
    p.set_defaults(grid_search=True)

    p.add_argument("--calibrate", action="store_true")
    p.add_argument("--no_calibrate", dest="calibrate", action="store_false")
    p.set_defaults(calibrate=True)

    p.add_argument("--cv_folds", type=int, default=5)
    p.add_argument("--c_values", type=str, default="0.5,1.0,2.0,4.0")

    # Misc
    p.add_argument("--n_jobs", type=int, default=-1)
    p.add_argument("--verbose", type=int, default=1)
    p.add_argument("--no_demo", action="store_true")
    return p

def main():
    args = build_argparser().parse_args()

    # Interactive embedding choice if not provided
    embedding = args.embedding
    if embedding is None:
        print("Select embedding type:")
        print("  1) word2vec (train or pretrained GoogleNews)")
        print("  2) glove (pretrained .txt)")
        print("  3) fasttext (train or pretrained Facebook .bin/.vec)")
        choice = input("Enter 1/2/3: ").strip()
        mapping = {"1":"word2vec","2":"glove","3":"fasttext"}
        embedding = mapping.get(choice, "word2vec")

    # Default pretrained path if user chose GloVe and didn't pass one
    glove_default = "glove.6B.300d.txt" if embedding == "glove" else None
    pretrained_path = args.pretrained_path if args.pretrained_path is not None else glove_default

    cfg = RunConfig(
        data_path=args.data_path,
        results_dir=args.results_dir,
        test_size=args.test_size,
        random_state=args.random_state,

        lowercase=not args.no_lowercase,
        remove_stopwords=not args.no_stop,
        lemmatize=not args.no_lemmatize,
        min_token_len=args.min_token_len,

        embedding=embedding,

        vector_size=args.vector_size,
        window=args.window,
        min_count=args.min_count,
        sg=args.sg,
        negative=args.negative,
        epochs=args.epochs,

        use_pretrained=args.use_pretrained,
        pretrained_path=pretrained_path,

        embedding_agg=args.embedding_agg,

        run_examples=args.run_examples,
        examples_path=args.examples_path,

        grid_search=args.grid_search,
        calibrate=args.calibrate,
        cv_folds=args.cv_folds,
        c_values=tuple(float(x) for x in args.c_values.split(",")),

        n_jobs=args.n_jobs,
        verbose=args.verbose
    )

    logger = logging.getLogger("IMDBEmbedSVM")
    logger.info(f"Resolved embedding: {cfg.embedding}")
    logger.info(f"Pretrained path (raw): {cfg.pretrained_path}")
    logger.info(f"CWD: {os.getcwd()}")

    runner = IMDBEmbedSVM(cfg)
    runner.run(demo_prompt=not args.no_demo)


if __name__ == "__main__":
    main()
