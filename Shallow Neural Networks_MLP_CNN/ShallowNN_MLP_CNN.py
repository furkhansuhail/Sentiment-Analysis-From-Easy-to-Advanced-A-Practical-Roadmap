# shallow_sentiment.py
# A basic Shallow NN (MLP or CNN) for IMDB-style sentiment analysis.
# - Reads a CSV with columns: review, sentiment
# - Cleans + tokenizes text, builds a small vocab, pads to fixed length
# - Trains either:
#       1) MLP: mean-pool(Embeddings) -> FC -> ReLU -> Dropout -> FC
#       2) CNN: Embeddings -> 1D conv (3/4/5) -> ReLU -> GlobalMaxPool -> FC
# - Prints accuracy and tests 10 example reviews
#
# NEW: Prompts user at runtime to choose model (mlp/cnn).
#      Also supports --model/-m CLI arg and MODEL_TYPE env var for automation.

import os
import re
import random
import argparse
import numpy as np
import pandas as pd
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tqdm.auto import tqdm

# -----------------------------
# Config
# -----------------------------
CSV_PATH = "IMDB Dataset.csv"     # <-- change if needed
DEFAULT_MODEL = "cnn"             # fallback if no input
random_seed = 42
max_len = 200
max_vocab_size = 30000
min_freq = 2
batch_size = 64
epochs = 3
lr = 1e-3
embed_dim = 128
dropout_p = 0.2
device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed(random_seed)

# -----------------------------
# CLI and interactive choice
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Shallow sentiment model (MLP/CNN)")
    p.add_argument("-m", "--model", choices=["mlp", "cnn"], help="Which model to use")
    p.add_argument("--csv", default=CSV_PATH, help="Path to CSV with 'review' and 'sentiment'")
    return p.parse_args()

def prompt_for_model(default=DEFAULT_MODEL):
    try:
        choice = input(f"Choose model [mlp/cnn] (default={default}): ").strip().lower()
        if not choice:
            return default
        if choice in {"mlp", "cnn"}:
            return choice
        print("Invalid choice; using default.")
        return default
    except (EOFError, KeyboardInterrupt):
        print("No input; using default.")
        return default

def choose_model_type(cli_choice, env_choice, default=DEFAULT_MODEL):
    # precedence: CLI > ENV > prompt (if interactive) > default
    if cli_choice in {"mlp", "cnn"}:
        return cli_choice
    if env_choice in {"mlp", "cnn"}:
        return env_choice
    # Try interactive prompt; if not possible, it will fall back to default
    return prompt_for_model(default)

# -----------------------------
# Basic text cleaning + tokenization
# -----------------------------
TOKEN_RE = re.compile(r"[A-Za-z']+")  # keep words/apostrophes

def clean_and_tokenize(text: str):
    text = str(text).lower()
    return TOKEN_RE.findall(text)

# -----------------------------
# Vocab building
# -----------------------------
PAD, UNK = "<PAD>", "<UNK>"

def build_vocab(texts, max_vocab=max_vocab_size, min_freq=min_freq):
    counter = Counter()
    for t in texts:
        counter.update(clean_and_tokenize(t))
    # keep tokens by frequency
    iterms = [(tok, freq) for tok, freq in counter.items() if freq >= min_freq]
    iterms.sort(key=lambda x: (-x[1], x[0]))
    iterms = iterms[: max_vocab - 2]  # reserve for PAD/UNK
    idx2tok = [PAD, UNK] + [tok for tok, _ in iterms]
    tok2idx = {tok: i for i, tok in enumerate(idx2tok)}
    return tok2idx, idx2tok

def encode(text, tok2idx, pad_id=0, unk_id=1, max_len=max_len):
    tokens = clean_and_tokenize(text)
    ids = [tok2idx.get(t, unk_id) for t in tokens]
    ids = ids[:max_len]
    # pad
    if len(ids) < max_len:
        ids = ids + [pad_id] * (max_len - len(ids))
    return ids

# -----------------------------
# Dataset
# -----------------------------
class TextClsDataset(Dataset):
    def __init__(self, texts, labels, tok2idx, max_len=200):
        self.texts = texts
        self.labels = labels
        self.tok2idx = tok2idx
        self.max_len = max_len
        self.pad_id = 0
        self.unk_id = 1

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        x_ids = encode(self.texts[i], self.tok2idx, pad_id=self.pad_id, unk_id=self.unk_id, max_len=self.max_len)
        y = int(self.labels[i])
        return torch.tensor(x_ids, dtype=torch.long), torch.tensor(y, dtype=torch.long)

# -----------------------------
# Models (Shallow)
# -----------------------------
class MLPClassifier(nn.Module):
    """
    Shallow MLP: mean-pool embeddings -> FC -> ReLU -> Dropout -> FC(2)
    """
    def __init__(self, vocab_size, embed_dim=128, dropout_p=0.2, pad_id=0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.fc1 = nn.Linear(embed_dim, 128)
        self.fc2 = nn.Linear(128, 2)
        self.dropout = nn.Dropout(dropout_p)
        self.pad_id = pad_id

    def forward(self, x):
        # x: (B, L)
        emb = self.embed(x)                 # (B, L, E)
        mask = (x != self.pad_id).unsqueeze(-1)  # (B, L, 1)
        emb = emb * mask
        # mean-pool (avoid div by zero)
        lengths = mask.sum(dim=1).clamp_min(1)   # (B, 1)
        pooled = emb.sum(dim=1) / lengths        # (B, E)
        h = self.dropout(F.relu(self.fc1(pooled)))
        logits = self.fc2(h)                     # (B, 2)
        return logits

class TextCNNClassifier(nn.Module):
    """
    Shallow Text-CNN: embeddings -> conv1d (3/4/5) -> ReLU -> global max pool -> concat -> FC(2)
    """
    def __init__(self, vocab_size, embed_dim=128, num_filters=100, filter_sizes=(3,4,5), dropout_p=0.2, pad_id=0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout_p)
        self.fc = nn.Linear(num_filters * len(filter_sizes), 2)

    def forward(self, x):
        # x: (B, L)
        emb = self.embed(x)           # (B, L, E)
        emb = emb.transpose(1, 2)     # (B, E, L) for Conv1d
        conv_outs = []
        for conv in self.convs:
            c = F.relu(conv(emb))         # (B, F, L')
            p = F.max_pool1d(c, kernel_size=c.shape[-1]).squeeze(-1)  # (B, F)
            conv_outs.append(p)
        h = torch.cat(conv_outs, dim=1)   # (B, F * len(fs))
        h = self.dropout(h)
        logits = self.fc(h)               # (B, 2)
        return logits

# -----------------------------
# Training / Evaluation
# -----------------------------
def run_epoch(model, loader, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, total = 0.0, 0
    all_preds, all_labels = [], []
    criterion = nn.CrossEntropyLoss()

    for xb, yb in tqdm(loader, disable=False, leave=False):
        xb = xb.to(device)
        yb = yb.to(device)

        logits = model(xb)
        loss = criterion(logits, yb)

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * yb.size(0)
        total += yb.size(0)
        preds = logits.argmax(dim=1).detach().cpu().numpy().tolist()
        all_preds.extend(preds)
        all_labels.extend(yb.detach().cpu().numpy().tolist())

    acc = accuracy_score(all_labels, all_preds)
    return total_loss / max(total, 1), acc

def train_model(model, train_loader, val_loader, epochs=3, lr=1e-3):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    best_val_acc = 0.0
    best_state = None

    for ep in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, optimizer=None)
        print(f"Epoch {ep:02d} | train_loss={train_loss:.4f} acc={train_acc:.4f} | val_loss={val_loss:.4f} acc={val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model

def evaluate(model, loader, label_names=("negative","positive")):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            preds = logits.argmax(dim=1).cpu().numpy().tolist()
            all_preds.extend(preds)
            all_labels.extend(yb.numpy().tolist())
    acc = accuracy_score(all_labels, all_preds)
    print("\nTest accuracy:", f"{acc:.4f}")
    print("\nClassification report:")
    print(classification_report(all_labels, all_preds, target_names=label_names, digits=4))
    return acc

def predict_texts(model, texts, tok2idx, id2label=("negative","positive")):
    model.eval()
    outs = []
    with torch.no_grad():
        for t in texts:
            ids = encode(t, tok2idx)
            x = torch.tensor([ids], dtype=torch.long, device=device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            pred = probs.argmax().item()
            outs.append((t, id2label[pred], float(probs[pred])))
    return outs

# -----------------------------
# Main
# -----------------------------
def main():
    args = parse_args()
    # Resolve model choice from CLI, ENV, or interactive prompt
    env_choice = (os.getenv("MODEL_TYPE") or "").strip().lower()
    model_type = choose_model_type(args.model, env_choice, default=DEFAULT_MODEL)
    csv_path = args.csv

    # 1) Load CSV
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found at {csv_path}. Please set --csv correctly.")
    df = pd.read_csv(csv_path)

    # 2) Normalize labels: accept strings ("positive"/"negative") or ints (1/0)
    if df["sentiment"].dtype == object:
        df["sentiment"] = df["sentiment"].str.strip().str.lower().map({"positive":1, "negative":0})
    df = df.dropna(subset=["review","sentiment"]).reset_index(drop=True)
    df["sentiment"] = df["sentiment"].astype(int)

    # (Optional) strip simple HTML artifacts in IMDB
    df["review"] = df["review"].astype(str).str.replace("<br />", " ", regex=False)

    # 3) Train/Val/Test split
    train_texts, temp_texts, train_y, temp_y = train_test_split(
        df["review"].tolist(), df["sentiment"].tolist(), test_size=0.2, random_state=random_seed, stratify=df["sentiment"]
    )
    val_texts, test_texts, val_y, test_y = train_test_split(
        temp_texts, temp_y, test_size=0.5, random_state=random_seed, stratify=temp_y
    )

    # 4) Build vocab on training only (prevents leakage)
    tok2idx, idx2tok = build_vocab(train_texts, max_vocab=max_vocab_size, min_freq=min_freq)
    vocab_size = len(idx2tok)
    print(f"Vocab size: {vocab_size}")

    # 5) Datasets/Loaders
    train_ds = TextClsDataset(train_texts, train_y, tok2idx, max_len=max_len)
    val_ds   = TextClsDataset(val_texts,   val_y, tok2idx, max_len=max_len)
    test_ds  = TextClsDataset(test_texts,  test_y, tok2idx, max_len=max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, drop_last=False)

    # 6) Pick model
    if model_type == "mlp":
        model = MLPClassifier(vocab_size=vocab_size, embed_dim=embed_dim, dropout_p=dropout_p, pad_id=0)
        print("Using model: Shallow MLP")
    elif model_type == "cnn":
        model = TextCNNClassifier(vocab_size=vocab_size, embed_dim=embed_dim, num_filters=100,
                                  filter_sizes=(3,4,5), dropout_p=dropout_p, pad_id=0)
        print("Using model: Shallow Text-CNN")
    else:
        raise ValueError("model_type must be 'mlp' or 'cnn'.")

    # 7) Train
    model = train_model(model, train_loader, val_loader, epochs=epochs, lr=lr)

    # 8) Evaluate
    _ = evaluate(model, test_loader, label_names=("negative","positive"))

    # 9) Try example reviews
    example_reviews = [
        "An absolute masterpiece! The performances were phenomenal and the story was gripping.",
        "Terrible. I was bored after the first 20 minutes and it never got better.",
        "Surprisingly fun and sweet—way better than I expected.",
        "A complete waste of time. Plot holes everywhere and wooden acting.",
        "It was okay, some parts dragged, but the ending was satisfying.",
        "The visuals were stunning and the soundtrack elevated every scene.",
        "I didn’t connect with any character, and the jokes fell flat.",
        "Heartwarming and clever! I smiled the entire time.",
        "Predictable from start to finish. Nothing new here.",
        "Dark, tense, and beautifully shot—I was on the edge of my seat."
    ]
    preds = predict_texts(model, example_reviews, tok2idx)
    print("\nExample predictions:")
    for text, label, prob in preds:
        # label is a str like "positive"/"negative"
        print(f"[{label.upper():8s}  p={prob:0.3f}] {text}")

if __name__ == "__main__":
    main()

