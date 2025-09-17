# sentiment_bilstm_and_han_imdb.py
# One-file IMDB sentiment classifier with:
#   1) Word-level BiLSTM + Additive Attention
#   2) Hierarchical Attention Network (HAN): word- and sentence-level attention
#
# Usage:
#   python sentiment_bilstm_and_han_imdb.py --model bilstm_attn
#   python sentiment_bilstm_and_han_imdb.py --model han
#
# Data:
#   IMDB Dataset.csv  (columns: review, sentiment)
#
# Notes on tensor shapes (common):
#   B = batch size
#   L = tokens per document (for word-level)
#   S = sentences per document (for HAN)
#   T = tokens per sentence (for HAN)
#   E = embedding dim
#   H = LSTM hidden size (per direction)
#   C = number of classes (2)
#
# Examples:
#   Word-level model forward:
#     x ............... [B, L]
#     embed(x) ........ [B, L, E]
#     BiLSTM .......... [B, L, 2H]
#     attention wts ... [B, L]
#     context vector .. [B, 2H]
#     logits .......... [B, C]
#
#   HAN forward:
#     x ............... [B, S, T]
#     embed(x) ........ [B, S, T, E]
#     word-BiLSTM ..... [B, S, T, 2H_w]
#     word-attn wts ... [B, S, T]
#     sentence vec .... [B, S, 2H_w]
#     sent-BiLSTM ..... [B, S, 2H_s]
#     sent-attn wts ... [B, S]
#     doc vec ......... [B, 2H_s]
#     logits .......... [B, C]

import os
import re
import math
import random
import argparse
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# ---------------------------
# Utils & Repro
# ---------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def normalize_sentiment(x) -> int:
    """
    Map incoming 'sentiment' to {0,1}. Accepts variations like strings or ints.
    """
    if pd.isna(x):
        return 0
    if isinstance(x, str):
        t = x.strip().lower()
        if t in {"positive", "pos", "good", "1", "true", "yes"}:
            return 1
        if t in {"negative", "neg", "bad", "0", "false", "no"}:
            return 0
        try:
            v = int(t)
            return 1 if v > 0 else 0
        except:
            return 0
    try:
        v = int(x)
        return 1 if v > 0 else 0
    except:
        return 0

# ---------------------------
# Tokenization (word + simple sentence split)
# ---------------------------
_WORD_RE = re.compile(r"[A-Za-z']+")
_SENT_SPLIT_RE = re.compile(r"([.!?])")

def simple_word_tokenize(text: str) -> List[str]:
    # lower-case and keep letters + apostrophes (basic)
    return [m.group(0).lower() for m in _WORD_RE.finditer(text)]

def simple_sentence_split(text: str) -> List[str]:
    """
    Splits by punctuation (., !, ?) while keeping things simple.
    """
    if not text:
        return [""]
    parts = _SENT_SPLIT_RE.split(text)
    sents = []
    buf = ""
    for p in parts:
        if _SENT_SPLIT_RE.fullmatch(p):
            # sentence end
            buf += p
            sents.append(buf.strip())
            buf = ""
        else:
            buf += p
    if buf.strip():
        sents.append(buf.strip())
    return sents if sents else [""]

# ---------------------------
# Vocabulary
# ---------------------------
PAD = "<PAD>"
UNK = "<UNK>"

class Vocab:
    def __init__(self, min_freq=2, max_size=60000):
        self.min_freq = min_freq
        self.max_size = max_size
        self.stoi: Dict[str, int] = {}
        self.itos: List[str] = []

    def build(self, texts: List[str]):
        freq: Dict[str, int] = {}
        for t in texts:
            for tok in simple_word_tokenize(t):
                freq[tok] = freq.get(tok, 0) + 1
        # specials first
        items = [(PAD, math.inf), (UNK, math.inf)]
        # sort by freq desc then lexicographically
        items += sorted([(w, c) for w, c in freq.items() if c >= self.min_freq],
                        key=lambda x: (-x[1], x[0]))
        if self.max_size:
            items = items[: self.max_size]
        self.itos = [w for w, _ in items]
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, tokens: List[str]) -> List[int]:
        s = self.stoi
        unk = s.get(UNK, 1)
        return [s.get(tok, unk) for tok in tokens]

    @property
    def pad_id(self) -> int:
        return self.stoi.get(PAD, 0)

    @property
    def unk_id(self) -> int:
        return self.stoi.get(UNK, 1)

# ---------------------------
# Datasets
# ---------------------------
class WordDataset(Dataset):
    """
    For word-level BiLSTM+Attention.
    Produces token IDs per document, to be padded/truncated to max_len in collate.
    """
    def __init__(self, texts: np.ndarray, labels: np.ndarray, vocab: Vocab):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int):
        text = str(self.texts[idx])
        toks = simple_word_tokenize(text)
        ids = self.vocab.encode(toks)  # variable length
        return ids, int(self.labels[idx])

def word_collate(batch, pad_id: int, max_len: int, device: str):
    """
    Pads/truncates to [B, L], returns (xb, yb)
    """
    seqs, labels = zip(*batch)
    L = max_len
    B = len(seqs)
    X = np.full((B, L), pad_id, dtype=np.int64)
    for i, seq in enumerate(seqs):
        seq = seq[:L]
        X[i, :len(seq)] = np.array(seq, dtype=np.int64)
    xb = torch.from_numpy(X).long().to(device)
    yb = torch.tensor(labels, dtype=torch.long).to(device)
    return xb, yb

class HANDataset(Dataset):
    """
    For HAN (hierarchical) — returns nested token IDs:
      sentences (S) x tokens (T)
    """
    def __init__(self, texts: np.ndarray, labels: np.ndarray, vocab: Vocab,
                 max_sents: int, max_words: int):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_sents = max_sents
        self.max_words = max_words

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int):
        text = str(self.texts[idx])
        sents = simple_sentence_split(text)
        sents = sents[: self.max_sents]

        encoded_sents = []
        for s in sents:
            toks = simple_word_tokenize(s)[: self.max_words]
            ids = self.vocab.encode(toks)
            encoded_sents.append(ids)

        # may be fewer than max_sents; pad in collate
        return encoded_sents, int(self.labels[idx])

def han_collate(batch, pad_id: int, max_sents: int, max_words: int, device: str):
    """
    Create [B, S, T] with padding.
    """
    nested, labels = zip(*batch)
    B = len(nested)
    S = max_sents
    T = max_words
    X = np.full((B, S, T), pad_id, dtype=np.int64)
    for i, doc in enumerate(nested):
        for s_idx, sent in enumerate(doc[:S]):
            sent = sent[:T]
            X[i, s_idx, :len(sent)] = np.array(sent, dtype=np.int64)
    xb = torch.from_numpy(X).long().to(device)
    yb = torch.tensor(labels, dtype=torch.long).to(device)
    return xb, yb

# ---------------------------
# Attention blocks
# ---------------------------
class AdditiveAttention(nn.Module):
    """
    Standard additive (Bahdanau-style) attention over a time dimension.
    Input H: [B, L, D]
    Output: context [B, D], weights [B, L]
    """
    def __init__(self, d_in: int, d_attn: int):
        super().__init__()
        self.W = nn.Linear(d_in, d_attn, bias=True)
        self.v = nn.Linear(d_attn, 1, bias=False)

    def forward(self, H: torch.Tensor):
        # H: [B, L, D]
        score = self.v(torch.tanh(self.W(H))).squeeze(-1)  # [B, L]
        weights = torch.softmax(score, dim=1)              # [B, L]
        ctx = torch.einsum("bl, bld -> bd", weights, H)    # [B, D]
        return ctx, weights

# ---------------------------
# Models
# ---------------------------
class BiLSTMAttn(nn.Module):
    """
    Word-level BiLSTM + Additive Attention -> logits
    """
    def __init__(self, vocab_size: int, emb_dim: int, hidden: int, num_classes: int, pad_id: int, dropout=0.2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(emb_dim, hidden, batch_first=True, bidirectional=True)
        self.attn = AdditiveAttention(d_in=2*hidden, d_attn=2*hidden)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2*hidden, num_classes)

    def forward(self, x: torch.Tensor):
        # x: [B, L]
        e = self.emb(x)                  # [B, L, E]
        H, _ = self.lstm(e)              # [B, L, 2H]
        ctx, w = self.attn(H)            # ctx: [B, 2H]
        z = self.dropout(ctx)
        return self.fc(z)                # [B, C]

class HAN(nn.Module):
    """
    Hierarchical Attention Network:
      Word encoder: Emb -> BiLSTM_w -> Attn_w  (per sentence)
      Sentence encoder: BiLSTM_s -> Attn_s     (across sentences)
    """
    def __init__(self,
                 vocab_size: int,
                 emb_dim: int,
                 hidden_word: int,
                 hidden_sent: int,
                 num_classes: int,
                 pad_id: int,
                 dropout=0.2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        # word-level
        self.lstm_w = nn.LSTM(emb_dim, hidden_word, batch_first=True, bidirectional=True)
        self.attn_w = AdditiveAttention(d_in=2*hidden_word, d_attn=2*hidden_word)
        # sentence-level
        self.lstm_s = nn.LSTM(2*hidden_word, hidden_sent, batch_first=True, bidirectional=True)
        self.attn_s = AdditiveAttention(d_in=2*hidden_sent, d_attn=2*hidden_sent)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2*hidden_sent, num_classes)

    def forward(self, x: torch.Tensor):
        """
        x: [B, S, T]
        """
        B, S, T = x.size()
        x = x.view(B*S, T)                     # [B*S, T]
        e = self.emb(x)                        # [B*S, T, E]
        Hw, _ = self.lstm_w(e)                 # [B*S, T, 2H_w]
        sw, _ = self.attn_w(Hw)                # [B*S, 2H_w]
        sent_vecs = sw.view(B, S, -1)          # [B, S, 2H_w]

        Hs, _ = self.lstm_s(sent_vecs)         # [B, S, 2H_s]
        ds, _ = self.attn_s(Hs)                # [B, 2H_s]
        z = self.dropout(ds)
        return self.fc(z)                      # [B, C]

# ---------------------------
# Trainer
# ---------------------------
class Trainer:
    def __init__(
        self,
        csv_path: str = "IMDB Dataset.csv",
        model_type: str = "bilstm_attn",     # or "han"
        max_len: int = 300,                  # for word-level model
        max_sents: int = 15,                 # for HAN
        max_words: int = 30,                 # for HAN
        emb_dim: int = 100,
        hidden_word: int = 64,
        hidden_sent: int = 64,
        num_epochs: int = 6,
        batch_size: int = 64,
        lr: float = 1e-3,
        min_freq: int = 2,
        seed: int = 42
    ):
        set_seed(seed)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.csv_path = csv_path
        self.model_type = model_type
        self.max_len = max_len
        self.max_sents = max_sents
        self.max_words = max_words
        self.emb_dim = emb_dim
        self.hidden_word = hidden_word
        self.hidden_sent = hidden_sent
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.min_freq = min_freq
        self.num_classes = 2

        self.vocab = Vocab(min_freq=min_freq, max_size=60000)

        self._prepare_data()
        self._build_model()

    def _prepare_data(self):
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")
        df = pd.read_csv(self.csv_path)
        if not {"review", "sentiment"}.issubset(df.columns):
            raise ValueError("CSV must contain columns: review, sentiment")

        df["review"] = df["review"].astype(str)
        df["label"] = df["sentiment"].apply(normalize_sentiment).astype(int)

        # Build vocab on train split text only (to avoid test leakage)
        X = df["review"].to_numpy()
        y = df["label"].to_numpy()

        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )

        self.train_texts, self.val_texts, self.test_texts = X_train, X_val, X_test
        self.train_labels, self.val_labels, self.test_labels = y_train, y_val, y_test

        # Build vocab from train texts only
        self.vocab.build(self.train_texts.tolist())
        print(f"[Vocab] size={len(self.vocab)}  min_freq={self.min_freq}")

        # Build datasets
        if self.model_type == "bilstm_attn":
            self.train_ds = WordDataset(self.train_texts, self.train_labels, self.vocab)
            self.val_ds   = WordDataset(self.val_texts,   self.val_labels,   self.vocab)
            self.test_ds  = WordDataset(self.test_texts,  self.test_labels,  self.vocab)
        elif self.model_type == "han":
            self.train_ds = HANDataset(self.train_texts, self.train_labels, self.vocab,
                                       max_sents=self.max_sents, max_words=self.max_words)
            self.val_ds   = HANDataset(self.val_texts,   self.val_labels,   self.vocab,
                                       max_sents=self.max_sents, max_words=self.max_words)
            self.test_ds  = HANDataset(self.test_texts,  self.test_labels,  self.vocab,
                                       max_sents=self.max_sents, max_words=self.max_words)
        else:
            raise ValueError("model_type must be 'bilstm_attn' or 'han'")

        # Dataloaders
        if self.model_type == "bilstm_attn":
            self.train_loader = DataLoader(
                self.train_ds, batch_size=self.batch_size, shuffle=True,
                collate_fn=lambda b: word_collate(b, self.vocab.pad_id, self.max_len, self.device)
            )
            self.val_loader = DataLoader(
                self.val_ds, batch_size=self.batch_size, shuffle=False,
                collate_fn=lambda b: word_collate(b, self.vocab.pad_id, self.max_len, self.device)
            )
            self.test_loader = DataLoader(
                self.test_ds, batch_size=self.batch_size, shuffle=False,
                collate_fn=lambda b: word_collate(b, self.vocab.pad_id, self.max_len, self.device)
            )
        else:
            self.train_loader = DataLoader(
                self.train_ds, batch_size=self.batch_size, shuffle=True,
                collate_fn=lambda b: han_collate(b, self.vocab.pad_id, self.max_sents, self.max_words, self.device)
            )
            self.val_loader = DataLoader(
                self.val_ds, batch_size=self.batch_size, shuffle=False,
                collate_fn=lambda b: han_collate(b, self.vocab.pad_id, self.max_sents, self.max_words, self.device)
            )
            self.test_loader = DataLoader(
                self.test_ds, batch_size=self.batch_size, shuffle=False,
                collate_fn=lambda b: han_collate(b, self.vocab.pad_id, self.max_sents, self.max_words, self.device)
            )

        xb, yb = next(iter(self.train_loader))
        print(f"[Sanity] sample batch: x={tuple(xb.shape)} y={tuple(yb.shape)}")

    def _build_model(self):
        if self.model_type == "bilstm_attn":
            self.model = BiLSTMAttn(
                vocab_size=len(self.vocab),
                emb_dim=self.emb_dim,
                hidden=self.hidden_word,
                num_classes=self.num_classes,
                pad_id=self.vocab.pad_id,
                dropout=0.2
            ).to(self.device)
        else:
            self.model = HAN(
                vocab_size=len(self.vocab),
                emb_dim=self.emb_dim,
                hidden_word=self.hidden_word,
                hidden_sent=self.hidden_sent,
                num_classes=self.num_classes,
                pad_id=self.vocab.pad_id,
                dropout=0.2
            ).to(self.device)

        self.opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()

    @torch.no_grad()
    def _evaluate(self, loader: DataLoader):
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

    def _train_epoch(self):
        self.model.train()
        total_loss, total_n = 0.0, 0
        y_true, y_pred = [], []
        for xb, yb in self.train_loader:
            logits = self.model(xb)
            loss = self.criterion(logits, yb)

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            total_loss += loss.item() * yb.size(0)
            total_n += yb.size(0)
            preds = logits.argmax(dim=-1)
            y_true.extend(yb.detach().cpu().tolist())
            y_pred.extend(preds.detach().cpu().tolist())
        train_loss = total_loss / max(1, total_n)
        train_acc = accuracy_score(y_true, y_pred)
        return train_loss, train_acc

    def fit(self):
        best_val, best_state = 0.0, None
        for epoch in range(1, self.num_epochs + 1):
            tr_loss, tr_acc = self._train_epoch()
            val_acc, val_f1 = self._evaluate(self.val_loader)
            test_acc, test_f1 = self._evaluate(self.test_loader)

            print(f"Epoch {epoch:02d} | "
                  f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
                  f"val_acc={val_acc:.4f} val_f1={val_f1:.4f} | "
                  f"test_acc={test_acc:.4f} test_f1={test_f1:.4f}")

            if val_acc > best_val:
                best_val = val_acc
                best_state = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}

        if best_state is not None:
            self.model.load_state_dict(best_state)
            print("[INFO] Restored best validation weights.")

        val_acc, val_f1 = self._evaluate(self.val_loader)
        test_acc, test_f1 = self._evaluate(self.test_loader)
        print(f"[FINAL] val_acc={val_acc:.4f} val_f1={val_f1:.4f} | "
              f"test_acc={test_acc:.4f} test_f1={test_f1:.4f}")

    @torch.no_grad()
    def predict(self, text: str):
        self.model.eval()
        if self.model_type == "bilstm_attn":
            toks = simple_word_tokenize(str(text))[: self.max_len]
            ids = self.vocab.encode(toks)
            X = np.full((1, self.max_len), self.vocab.pad_id, dtype=np.int64)
            X[0, :len(ids)] = np.array(ids, dtype=np.int64)
            xb = torch.from_numpy(X).long().to(self.device)
            logits = self.model(xb)
        else:
            sents = simple_sentence_split(str(text))[: self.max_sents]
            doc = []
            for s in sents:
                toks = simple_word_tokenize(s)[: self.max_words]
                doc.append(self.vocab.encode(toks))
            X = np.full((1, self.max_sents, self.max_words), self.vocab.pad_id, dtype=np.int64)
            for i, sent in enumerate(doc):
                X[0, i, :len(sent)] = np.array(sent, dtype=np.int64)
            xb = torch.from_numpy(X).long().to(self.device)
            logits = self.model(xb)

        probs = torch.softmax(logits, dim=-1).squeeze(0).detach().cpu().numpy()
        pred = int(np.argmax(probs))
        return pred, probs

# ---------------------------
# CLI
# ---------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="bilstm_attn", choices=["bilstm_attn", "han"])
    p.add_argument("--csv", type=str, default="IMDB Dataset.csv")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--emb", type=int, default=100)
    p.add_argument("--hidw", type=int, default=64, help="Hidden size for word-level LSTM")
    p.add_argument("--hids", type=int, default=64, help="Hidden size for sentence-level LSTM (HAN only)")
    p.add_argument("--max_len", type=int, default=300, help="Max tokens per doc (word-level model)")
    p.add_argument("--max_sents", type=int, default=15, help="Max sentences per doc (HAN)")
    p.add_argument("--max_words", type=int, default=30, help="Max words per sentence (HAN)")
    p.add_argument("--min_freq", type=int, default=2)
    args = p.parse_args()

    trainer = Trainer(
        csv_path=args.csv,
        model_type=args.model,
        max_len=args.max_len,
        max_sents=args.max_sents,
        max_words=args.max_words,
        emb_dim=args.emb,
        hidden_word=args.hidw,
        hidden_sent=args.hids,
        num_epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        min_freq=args.min_freq,
        seed=42
    )
    trainer.fit()

    print("\nType a review to predict. Press ENTER on empty line to exit.")
    while True:
        try:
            txt = input("Your review: ").strip()
        except EOFError:
            break
        if not txt:
            break
        label, probs = trainer.predict(txt)
        print(f"[{'positive' if label==1 else 'negative':8s}]  p(neg)={probs[0]:.4f}  p(pos)={probs[1]:.4f}")

if __name__ == "__main__":
    main()
