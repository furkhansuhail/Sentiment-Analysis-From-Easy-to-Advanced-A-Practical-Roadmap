# sentiment_ml_imdb_oop.py
import os
import argparse
from typing import Dict, Tuple, List

import pandas as pd
import numpy as np
from tqdm.auto import tqdm

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay
)


class SentimentMLIMDB:
    def __init__(
        self,
        dataset_path: str = "IMDB Dataset.csv",
        results_dir: str = "results_ml",
        random_state: int = 42,
        tfidf_config: Dict = None
    ):
        self.dataset_path = dataset_path
        self.results_dir = results_dir
        self.random_state = random_state
        self.tfidf_config = tfidf_config or dict(
            stop_words="english",
            max_features=30000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.9,
        )

        os.makedirs(self.results_dir, exist_ok=True)

        # Will be set during run
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.tfidf: TfidfVectorizer = None
        self.X_train_tfidf = None
        self.X_test_tfidf = None
        self.models: Dict[str, object] = {}
        self.summary_rows: List[Dict] = []
        self.summary_df: pd.DataFrame = None

    # -----------------------------
    # Helpers (confusion matrix + top features)
    # -----------------------------
    def _show_and_save_confusion(self, y_true, y_pred, model_name: str) -> str:
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(
            y_true, y_pred, display_labels=["Negative", "Positive"], cmap=None, ax=ax
        )
        ax.set_title(f"{model_name} – Confusion Matrix")
        out_path = os.path.join(self.results_dir, f"confusion_{model_name}.png")
        fig.tight_layout()
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        return out_path

    def _top_features_linear(self, model, k: int = 20) -> Tuple[List[str], List[str]]:
        if not hasattr(model, "coef_"):
            return [], []
        feature_names = np.array(self.tfidf.get_feature_names_out())
        coefs = model.coef_.ravel()
        top_pos_idx = np.argsort(coefs)[-k:][::-1]
        top_neg_idx = np.argsort(coefs)[:k]
        return feature_names[top_pos_idx].tolist(), feature_names[top_neg_idx].tolist()

    def _top_features_nb(self, model, k: int = 20) -> Tuple[List[str], List[str]]:
        if not hasattr(model, "feature_log_prob_"):
            return [], []
        feature_names = np.array(self.tfidf.get_feature_names_out())
        # Assumes label order [0=negative, 1=positive]
        log_prob_neg = model.feature_log_prob_[0]
        log_prob_pos = model.feature_log_prob_[1]
        log_odds = log_prob_pos - log_prob_neg
        top_pos_idx = np.argsort(log_odds)[-k:][::-1]
        top_neg_idx = np.argsort(log_odds)[:k]
        return feature_names[top_pos_idx].tolist(), feature_names[top_neg_idx].tolist()

    def _save_top_features(self, model, model_name: str, k: int = 20) -> str:
        if isinstance(model, MultinomialNB):
            pos_terms, neg_terms = self._top_features_nb(model, k=k)
        else:
            pos_terms, neg_terms = self._top_features_linear(model, k=k)

        out_path = os.path.join(self.results_dir, f"top_features_{model_name}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"Model: {model_name}\n")
            f.write(f"Top {k} positive n-grams:\n")
            f.write(", ".join(pos_terms) + "\n\n")
            f.write(f"Top {k} negative n-grams:\n")
            f.write(", ".join(neg_terms) + "\n")
        return out_path

    # -----------------------------
    # Pipeline steps
    # -----------------------------
    def load_dataset(self):
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(
                f"Could not find '{self.dataset_path}'. "
                "Download from Kaggle and place it next to this script, or update dataset_path."
            )
        df = pd.read_csv(self.dataset_path)

        # Normalize column names and validate
        if set(df.columns) != {"review", "sentiment"}:
            df.columns = [c.strip().lower() for c in df.columns]
            if "review" not in df.columns or "sentiment" not in df.columns:
                raise ValueError("Expected columns: 'review' and 'sentiment'.")

        df = df.dropna(subset=["review", "sentiment"]).reset_index(drop=True)
        df["label"] = df["sentiment"].map({"positive": 1, "negative": 0})
        if df["label"].isna().any():
            raise ValueError("Found sentiments other than 'positive'/'negative'.")

        self.df = df

    def split(self, test_size: float = 0.2):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.df["review"],
            self.df["label"],
            test_size=test_size,
            stratify=self.df["label"],
            random_state=self.random_state
        )

    def vectorize(self):
        print("Building TF-IDF features...")
        self.tfidf = TfidfVectorizer(**self.tfidf_config)
        self.X_train_tfidf = self.tfidf.fit_transform(self.X_train)
        self.X_test_tfidf = self.tfidf.transform(self.X_test)
        print(f"TF-IDF shapes: train={self.X_train_tfidf.shape}, test={self.X_test_tfidf.shape}")

    def define_models(self):
        self.models = {
            "Naive_Bayes": MultinomialNB(),  # tweak alpha if desired
            "SVM": LinearSVC(C=1.0, random_state=self.random_state),
            "Logistic_Regression": LogisticRegression(
                max_iter=2000, solver="lbfgs", C=1.0, random_state=self.random_state
            ),
        }

    def train_evaluate_save(self):
        print("\nTraining and evaluating models...")
        outer_bar = tqdm(total=len(self.models), desc="Models", position=0)

        for model_name, model in self.models.items():
            inner_bar = tqdm(total=5, desc=f"{model_name}", position=1, leave=False)

            # (a) Train
            inner_bar.set_postfix_str("training")
            model.fit(self.X_train_tfidf, self.y_train)
            inner_bar.update(1)

            # (b) Predict
            inner_bar.set_postfix_str("predicting")
            y_pred = model.predict(self.X_test_tfidf)
            inner_bar.update(1)

            # (c) Metrics
            inner_bar.set_postfix_str("scoring")
            acc = accuracy_score(self.y_test, y_pred)
            f1_w = f1_score(self.y_test, y_pred, average="weighted")
            f1_m = f1_score(self.y_test, y_pred, average="macro")
            report = classification_report(
                self.y_test, y_pred, target_names=["Negative", "Positive"]
            )
            inner_bar.update(1)

            # (d) Save per-model predictions
            per_model_df = pd.DataFrame({
                "Review": self.X_test.values,
                "Actual": self.y_test.map({1: "positive", 0: "negative"}).values,
                "Predicted": pd.Series(y_pred).map({1: "positive", 0: "negative"}).values
            })
            per_model_path = os.path.join(self.results_dir, f"results_{model_name}.csv")
            per_model_df.to_csv(per_model_path, index=False)

            # (e) Confusion matrix + top features
            cm_path = self._show_and_save_confusion(self.y_test, y_pred, model_name)
            tf_path = self._save_top_features(model, model_name, k=20)

            # Save classification report
            with open(os.path.join(self.results_dir, f"classification_report_{model_name}.txt"),
                      "w", encoding="utf-8") as f:
                f.write(f"Model: {model_name}\n")
                f.write(f"Accuracy: {acc:.4f}\n")
                f.write(f"F1 (weighted): {f1_w:.4f}\n")
                f.write(f"F1 (macro): {f1_m:.4f}\n\n")
                f.write(report)
                f.write("\nArtifacts:\n")
                f.write(f"- Predictions CSV: {os.path.abspath(per_model_path)}\n")
                f.write(f"- Confusion Matrix PNG: {os.path.abspath(cm_path)}\n")
                f.write(f"- Top Features TXT: {os.path.abspath(tf_path)}\n")

            # Collect summary metrics
            self.summary_rows.append({
                "Model": model_name,
                "Accuracy": round(acc, 4),
                "F1_weighted": round(f1_w, 4),
                "F1_macro": round(f1_m, 4),
                "Predictions_CSV": per_model_path,
                "Confusion_PNG": cm_path,
                "Top_Features_TXT": tf_path,
            })

            inner_bar.update(1)
            inner_bar.close()
            outer_bar.update(1)

        outer_bar.close()

    def save_comparison(self) -> pd.DataFrame:
        self.summary_df = pd.DataFrame(self.summary_rows).sort_values(
            by="Accuracy", ascending=False
        )
        summary_csv = os.path.join(self.results_dir, "comparison_metrics.csv")
        self.summary_df.to_csv(summary_csv, index=False)

        print("\n================ SUMMARY ================")
        print(self.summary_df.to_string(index=False))
        print(f"\nSaved per-model predictions to: {os.path.abspath(self.results_dir)}")
        print(f"Comparison metrics saved to: {os.path.abspath(summary_csv)}")
        print("========================================")
        return self.summary_df

    def show_best_model_top_features(self, k: int = 20):
        if self.summary_df is None or self.summary_df.empty:
            print("Summary is empty; nothing to inspect.")
            return

        best_row = self.summary_df.iloc[0]
        best_model_name = best_row["Model"]
        print(f"\nBest model: {best_model_name}")

        # Refit best model on the full training set for feature inspection
        best_model = self.models[best_model_name]
        best_model.fit(self.X_train_tfidf, self.y_train)

        if best_model_name == "Naive_Bayes":
            pos_terms, neg_terms = self._top_features_nb(best_model, k=k)
        else:
            pos_terms, neg_terms = self._top_features_linear(best_model, k=k)

        print(f"\nTop {k} positive n-grams:")
        print(", ".join(pos_terms))
        print(f"\nTop {k} negative n-grams:")
        print(", ".join(neg_terms))

    # -----------------------------
    # Orchestrator
    # -----------------------------
    def run(self, test_size: float = 0.2, show_best: bool = True, k: int = 20):
        self.load_dataset()
        self.split(test_size=test_size)
        self.vectorize()
        self.define_models()
        self.train_evaluate_save()
        self.save_comparison()
        if show_best:
            self.show_best_model_top_features(k=k)


# -----------------------------
# main entry
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="IMDB Sentiment ML + TF-IDF (NB/SVM/LR)")
    parser.add_argument("--dataset", type=str, default="IMDB Dataset.csv",
                        help="Path to the IMDB dataset CSV")
    parser.add_argument("--results_dir", type=str, default="results_ml",
                        help="Directory to save outputs")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Test split size (0-1)")
    parser.add_argument("--no_show_best", action="store_true",
                        help="Disable printing top n-grams for best model")
    parser.add_argument("--k", type=int, default=20,
                        help="How many top n-grams to display/save")
    args = parser.parse_args()

    runner = SentimentMLIMDB(
        dataset_path=args.dataset,
        results_dir=args.results_dir,
        random_state=42,
        tfidf_config=dict(
            stop_words="english",
            max_features=30000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.9,
        )
    )
    runner.run(test_size=args.test_size, show_best=not args.no_show_best, k=args.k)


if __name__ == "__main__":
    main()
