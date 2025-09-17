# sentiment_charLSTM_attention_imdb.py
# One-file IMDB sentiment classifier:
# - Reads single CSV: IMDB Dataset.csv (columns: review, sentiment)
# - Splits into train/val/test
# - Char-level Bidirectional LSTM + Attention
# - Prints train/val/test metrics each epoch
# - Checkpoint save/load + interactive prediction loop

import os
import numpy as np
import pandas as pd
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report


# ---------------------------
# Utility
# ---------------------------
def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------
# Model: BiLSTM + Attention
# ---------------------------
class LSTMWithAttention(nn.Module):
    """
    Character-level sentiment model:
      Embedding -> BiLSTM -> Additive Attention over time -> FC logits
    """
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int, output_size: int, padding_idx: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        # simple additive attention
        self.att_q = nn.Linear(hidden_size * 2, hidden_size * 2, bias=False)
        self.att_k = nn.Linear(hidden_size * 2, hidden_size * 2, bias=False)
        self.att_v = nn.Linear(hidden_size * 2, 1, bias=False)

        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T]
        h, _ = self.lstm(self.embedding(x))      # h: [B, T, 2H]
        scores = self.att_v(torch.tanh(self.att_q(h) + self.att_k(h))).squeeze(-1)  # [B, T]
        weights = torch.softmax(scores, dim=-1)                                     # [B, T]
        ctx = torch.einsum("bt,btd->bd", weights, h)                                # [B, 2H]
        logits = self.fc(ctx)                                                       # [B, C]
        return logits


# ---------------------------
# Dataset & Collate
# ---------------------------
class CharDataset(Dataset):
    def __init__(self, texts: np.ndarray, labels: np.ndarray):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int):
        return self.texts[idx], int(self.labels[idx])


# ---------------------------
# Main Trainer
# ---------------------------
class IMDBCharLSTMTrainer:
    def __init__(
        self,
        csv_filename: str = "IMDB Dataset.csv",
        seq_len: int = 200,
        embedding_dim: int = 32,
        hidden_size: int = 64,
        batch_size: int = 64,
        num_epochs: int = 8,
        learning_rate: float = 1e-3,
        seed: int = 42,
    ):
        set_seed(seed)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Hyperparameters
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.output_size = 2
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate

        # Paths
        base_dir = os.path.dirname(__file__)
        self.csv_path = os.path.join(base_dir, csv_filename)
        self.ckpt_path = os.path.join(base_dir, "imdb_char_lstm.pt")

        # Vocab members
        self.char_to_index = None
        self.index_to_char = None
        self.vocab_size = None
        self.pad_id = None
        self.sos_id = None
        self.eos_id = None

        # Data splits
        self.train_texts = None
        self.train_labels = None
        self.val_texts = None
        self.val_labels = None
        self.test_texts = None
        self.test_labels = None

        # Dataloaders
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        # Model
        self.model = None

        # Load & prepare
        self._load_csv()
        self._build_vocab()
        self._split_data()
        self._build_loaders()

    # ---------------------------
    # Data prep
    # ---------------------------
    def _load_csv(self):
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV not found at {self.csv_path}. "
                                    f"Expected a file with columns: 'review' and 'sentiment'.")
        df = pd.read_csv(self.csv_path)
        if not {"review", "sentiment"}.issubset(df.columns):
            raise ValueError("CSV must contain columns: 'review' and 'sentiment'.")

        # Normalize/clean
        df["review"] = df["review"].astype(str)
        df["label"] = df["sentiment"].apply(self._normalize_sentiment).astype(int)

        self.df = df

        print("***** Data Overview *****")
        print(self.df.info())
        print("Rows:", len(self.df))
        print("Label distribution:", self.df["label"].value_counts().to_dict())
        print("***********************")

    @staticmethod
    def _normalize_sentiment(s) -> int:
        """
        Robust mapping to {0,1}.
        Accepts 0/1, 1/2, strings like positive/negative, yes/no, etc.
        """
        if pd.isna(s):
            return 0
        if isinstance(s, str):
            t = s.strip().lower()
            if t in {"positive", "pos", "good", "1", "true", "yes"}:
                return 1
            if t in {"negative", "neg", "bad", "0", "false", "no"}:
                return 0
            try:
                v = int(t)
                if v == 2:
                    return 1
                if v == 1:
                    return 1
                if v == 0:
                    return 0
                return 1 if v > 0 else 0
            except:
                return 0
        else:
            try:
                v = int(s)
                if v == 2:
                    return 1
                if v == 1:
                    return 1
                if v == 0:
                    return 0
                return 1 if v > 0 else 0
            except:
                return 0

    def _build_vocab(self):
        # Char inventory from all reviews + special tokens
        all_text = "".join(self.df["review"].to_numpy())
        base_chars = sorted(list(set(all_text)))
        vocab = base_chars + ["<SOS>", "<EOS>", "<PAD>"]

        self.char_to_index = {ch: i for i, ch in enumerate(vocab)}
        self.index_to_char = {i: ch for i, ch in enumerate(vocab)}
        self.vocab_size = len(vocab)
        self.pad_id = self.char_to_index["<PAD>"]
        self.sos_id = self.char_to_index["<SOS>"]
        self.eos_id = self.char_to_index["<EOS>"]

        print(f"Vocab size: {self.vocab_size}")

    def _encode_ids(self, text: str) -> List[int]:
        return [self.char_to_index.get(ch, self.pad_id) for ch in text]

    def _encode_with_sos_eos(self, text: str) -> List[int]:
        ids = [self.sos_id] + self._encode_ids(text) + [self.eos_id]
        if len(ids) > self.seq_len:
            # keep SOS/EOS and trim middle
            ids = [self.sos_id] + ids[1:self.seq_len - 1] + [self.eos_id]
        return ids

    def _split_data(self):
        X = self.df["review"].to_numpy()
        y = self.df["label"].to_numpy()

        # 80/10/10 split
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
        )

        self.train_texts, self.train_labels = X_train, y_train
        self.val_texts, self.val_labels = X_val, y_val
        self.test_texts, self.test_labels = X_test, y_test

        print(f"Splits -> train: {len(self.train_texts)} | val: {len(self.val_texts)} | test: {len(self.test_texts)}")

    def _collate(self, batch: List[Tuple[str, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        texts, labels = zip(*batch)
        encoded = [self._encode_with_sos_eos(t) for t in texts]
        max_len = min(max(len(seq) for seq in encoded), self.seq_len)
        X = np.full((len(encoded), max_len), self.pad_id, dtype=np.int64)
        for i, ids in enumerate(encoded):
            ids = ids[:max_len]
            X[i, :len(ids)] = ids
        xb = torch.from_numpy(X).long().to(self.device)
        yb = torch.tensor(labels, dtype=torch.long).to(self.device)
        return xb, yb

    def _build_loaders(self):
        train_ds = CharDataset(self.train_texts, self.train_labels)
        val_ds = CharDataset(self.val_texts, self.val_labels)
        test_ds = CharDataset(self.test_texts, self.test_labels)

        self.train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, collate_fn=self._collate)
        self.val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False, collate_fn=self._collate)
        self.test_loader = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False, collate_fn=self._collate)

        # quick sanity
        xb, yb = next(iter(self.train_loader))
        print("Sample batch:", xb.shape, yb.shape)

    # ---------------------------
    # Model & Checkpoint
    # ---------------------------
    def _build_model(self):
        model = LSTMWithAttention(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            hidden_size=self.hidden_size,
            output_size=self.output_size,
            padding_idx=self.pad_id
        ).to(self.device)
        return model

    def _save_ckpt(self):
        payload = {
            "model_state_dict": self.model.state_dict(),
            "char_to_index": self.char_to_index,
            "index_to_char": self.index_to_char,
            "vocab_size": self.vocab_size,
            "pad_id": self.pad_id,
            "sos_id": self.sos_id,
            "eos_id": self.eos_id,
            "seq_len": self.seq_len,
            "embedding_dim": self.embedding_dim,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
        }
        torch.save(payload, self.ckpt_path)
        print(f"[CKPT] Saved: {self.ckpt_path}")

    def _load_ckpt(self):
        print(f"[CKPT] Loading: {self.ckpt_path}")
        payload = torch.load(self.ckpt_path, map_location=self.device)

        self.char_to_index = payload["char_to_index"]
        self.index_to_char = payload["index_to_char"]
        self.vocab_size = payload["vocab_size"]
        self.pad_id = payload["pad_id"]
        self.sos_id = payload["sos_id"]
        self.eos_id = payload["eos_id"]
        self.seq_len = payload.get("seq_len", self.seq_len)
        self.embedding_dim = payload.get("embedding_dim", self.embedding_dim)
        self.hidden_size = payload.get("hidden_size", self.hidden_size)
        self.output_size = payload.get("output_size", self.output_size)

        self.model = self._build_model()
        self.model.load_state_dict(payload["model_state_dict"])
        self.model.eval()
        print("[CKPT] Loaded & model restored.")

    # ---------------------------
    # Train / Eval
    # ---------------------------
    @torch.no_grad()
    def _evaluate_loader(self, loader: DataLoader) -> Tuple[float, float]:
        self.model.eval()
        y_true, y_pred = [], []
        for xb, yb in loader:
            logits = self.model(xb)
            preds = logits.argmax(dim=-1)
            y_true.extend(yb.detach().cpu().tolist())
            y_pred.extend(preds.detach().cpu().tolist())
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro")
        return acc, f1

    def _run_epoch(self, optimizer: torch.optim.Optimizer, criterion: nn.Module) -> Tuple[float, float]:
        self.model.train()
        total_loss, n = 0.0, 0
        y_true, y_pred = [], []

        for xb, yb in self.train_loader:
            logits = self.model(xb)
            loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * yb.size(0)
            n += yb.size(0)
            preds = logits.argmax(dim=-1)
            y_true.extend(yb.detach().cpu().tolist())
            y_pred.extend(preds.detach().cpu().tolist())

        train_loss = total_loss / max(1, n)
        train_acc = accuracy_score(y_true, y_pred)
        return train_loss, train_acc

    def fit(self, eval_test_each_epoch: bool = True):
        if self.model is None:
            self.model = self._build_model()

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()

        best_val_acc = 0.0
        best_state = None

        for epoch in range(1, self.num_epochs + 1):
            train_loss, train_acc = self._run_epoch(optimizer, criterion)
            val_acc, val_f1 = self._evaluate_loader(self.val_loader)

            if eval_test_each_epoch:
                test_acc, test_f1 = self._evaluate_loader(self.test_loader)
                print(
                    f"Epoch {epoch:02d} | "
                    f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                    f"val_acc={val_acc:.4f} val_f1={val_f1:.4f} | "
                    f"test_acc={test_acc:.4f} test_f1={test_f1:.4f}"
                )
            else:
                print(
                    f"Epoch {epoch:02d} | "
                    f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                    f"val_acc={val_acc:.4f} val_f1={val_f1:.4f}"
                )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}

        if best_state is not None:
            self.model.load_state_dict(best_state)
            print("[INFO] Restored best validation weights.")

        # Final full evaluation
        val_acc, val_f1 = self._evaluate_loader(self.val_loader)
        test_acc, test_f1 = self._evaluate_loader(self.test_loader)
        print(f"[FINAL] val_acc={val_acc:.4f} val_f1={val_f1:.4f} | test_acc={test_acc:.4f} test_f1={test_f1:.4f}")

        self._save_ckpt()

    @torch.no_grad()
    def predict_text(self, text: str) -> Tuple[int, np.ndarray]:
        """
        Returns: (pred_class_idx, probs ndarray shape [2])
        """
        if self.model is None:
            raise RuntimeError("Model not initialized. Train or load first.")
        self.model.eval()
        ids = self._encode_with_sos_eos(str(text))
        X = np.full((1, min(len(ids), self.seq_len)), self.pad_id, dtype=np.int64)
        X[0, :len(ids)] = np.array(ids[: self.seq_len], dtype=np.int64)
        xb = torch.from_numpy(X).long().to(self.device)

        logits = self.model(xb)
        probs = torch.softmax(logits, dim=-1).squeeze(0).detach().cpu().numpy()
        pred_idx = int(np.argmax(probs))
        return pred_idx, probs


# ---------------------------
# CLI Entry
# ---------------------------
if __name__ == "__main__":
    trainer = IMDBCharLSTMTrainer(
        csv_filename="IMDB Dataset.csv",
        seq_len=200,
        embedding_dim=32,
        hidden_size=64,
        batch_size=64,
        num_epochs=8,           # increase for better accuracy
        learning_rate=1e-3,
        seed=42,
    )

    # Load existing checkpoint if available (optional)
    use_saved = False
    if os.path.exists(trainer.ckpt_path):
        try:
            resp = input(
                f"Found saved model at '{trainer.ckpt_path}'. Use it instead of retraining? [y/N]: "
            ).strip().lower()
            use_saved = (resp == "y")
        except Exception:
            use_saved = False

    if use_saved:
        trainer._load_ckpt()
    else:
        trainer.fit(eval_test_each_epoch=True)

    # Interactive predictions
    print("\nEnter a review to predict sentiment. Press ENTER on an empty line to quit.")
    while True:
        try:
            user_text = input("\nYour review: ").strip()
        except EOFError:
            break
        if not user_text:
            break
        pred_idx, probs = trainer.predict_text(user_text)
        label = "positive" if pred_idx == 1 else "negative"
        print(f"[{label.upper():8s}]  p(neg)={probs[0]:.4f}  p(pos)={probs[1]:.4f}")
