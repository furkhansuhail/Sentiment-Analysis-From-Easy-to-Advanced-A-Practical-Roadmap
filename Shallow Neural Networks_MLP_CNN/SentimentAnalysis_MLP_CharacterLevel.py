# movie_mlp_sentiment.py
import os
import json
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split


class MovieReviewsMLPSentimentModel:
    def __init__(self):
        # Paths
        base_directory = os.path.dirname(__file__)
        data_path = os.path.join(base_directory, "IMDB Dataset.csv")  # expects columns: review, sentiment
        self.ckpt_path = os.path.join(base_directory, "movie_char_mlp.pt")

        # Load
        self.data = pd.read_csv(data_path)

        # Device
        self.compute_device = "cuda" if torch.cuda.is_available() else "cpu"

        # Hyperparameters (you can tweak)
        self.embedding_dim = 64
        self.hidden1_size  = 128
        self.hidden2_size  = 64
        self.output_size   = 2
        self.batch_size    = 64
        self.seq_len       = 200          # max chars kept per review (for pooling cap)
        self.learning_rate = 1e-3
        self.num_epochs    = 8
        self.eval_interval = 500
        self.dropout_p     = 0.2
        self.pooling       = "mean"       # "mean" | "max" | "sos_eos_mean"

        # Vocab members
        self.character_vocabulary = None
        self.char_to_index = None
        self.index_to_char = None
        self.vocabulary_size = None
        self.pad_id = None
        self.sos_id = None
        self.eos_id = None

        # Splits & loaders
        self.train_features = None
        self.train_labels   = None
        self.val_features   = None
        self.val_labels     = None
        self.test_features  = None
        self.test_labels    = None

        self.train_loader = None
        self.val_loader   = None
        self.test_loader  = None

        # Model
        self.model = None

        # Build pipeline
        self.perform_exploratory_analysis()
        self.prepare_character_encoding(self.char_to_index, self.character_vocabulary)
        self._build_dataloaders()

    # ---------------------- Data prep ----------------------
    def _normalize_sentiment(self, s):
        """
        Map various sentiment encodings to {0,1}.
        Accepts 0/1, 1/2, or strings like 'negative'/'positive'.
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
                if v == 2: return 1
                if v >= 1: return 1 if v > 1 else 0
                return 0
            except:
                return 0
        else:
            try:
                v = int(s)
                if v == 2: return 1
                if v == 1: return 1
                if v == 0: return 0
                return 1 if v > 0 else 0
            except:
                return 0

    def perform_exploratory_analysis(self):
        print("*******************************************************")
        print(self.data.info())
        print("*******************************************************")
        print(f"Total rows: {len(self.data)}")
        print("*******************************************************")

        # Ensure columns exist
        if not {"review", "sentiment"}.issubset(set(self.data.columns)):
            raise ValueError("CSV must contain columns: 'review' and 'sentiment'.")

        # Cast to str
        self.data["review"] = self.data["review"].astype(str)

        # Normalize labels
        self.data["label"] = self.data["sentiment"].apply(self._normalize_sentiment).astype(int)

        # Build char vocab from all text
        all_reviews = self.data["review"].to_numpy()
        base_chars = sorted(list(set(''.join(all_reviews))))
        self.character_vocabulary = base_chars + ['<SOS>', '<EOS>', '<PAD>']

        # stoi/itos
        self.char_to_index = {ch: i for i, ch in enumerate(self.character_vocabulary)}
        self.index_to_char = {i: ch for i, ch in enumerate(self.character_vocabulary)}
        self.vocabulary_size = len(self.character_vocabulary)

        # Special IDs
        self.pad_id = self.char_to_index['<PAD>']
        self.sos_id = self.char_to_index['<SOS>']
        self.eos_id = self.char_to_index['<EOS>']

        print("Vocab size:", self.vocabulary_size)

    def prepare_character_encoding(self, _, __):
        X = self.data["review"].to_numpy()
        y = self.data["label"].to_numpy()

        # 80/10/10 split
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
        )

        self.train_features = X_train
        self.train_labels   = y_train
        self.val_features   = X_val
        self.val_labels     = y_val
        self.test_features  = X_test
        self.test_labels    = y_test

        # encoders
        self.encode_text = lambda s: [self.char_to_index.get(ch, self.pad_id) for ch in s]
        self.decode_ids  = lambda l: ''.join([self.index_to_char[i] for i in l])

        # Sanity mini-batch
        xb, yb = self.get_training_batch(32, self.train_features, self.train_labels, self.encode_text, self.char_to_index)
        print("Sample batch shapes:", xb.shape, yb.shape)

    def _encode_with_sos_eos_trim(self, text: str):
        ids = [self.sos_id] + self.encode_text(text) + [self.eos_id]
        if len(ids) > self.seq_len:
            ids = [self.sos_id] + ids[1:self.seq_len-1] + [self.eos_id]
        return ids

    def _collate(self, batch):
        texts, ys = zip(*batch)
        # For MLP, we still create a fixed-length sequence to pool over (cap at seq_len)
        encoded = [self._encode_with_sos_eos_trim(t) for t in texts]
        max_len = min(max(len(s) for s in encoded), self.seq_len)
        X = np.full((len(encoded), max_len), self.pad_id, dtype=np.int64)
        for i, ids in enumerate(encoded):
            ids = ids[:max_len]
            X[i, :len(ids)] = ids
        return torch.from_numpy(X).long().to(self.compute_device), torch.tensor(ys, dtype=torch.long).to(self.compute_device)

    class _CharDataset(Dataset):
        def __init__(self, texts, labels):
            self.texts = texts
            self.labels = labels
        def __len__(self):
            return len(self.texts)
        def __getitem__(self, idx):
            return self.texts[idx], int(self.labels[idx])

    def _build_dataloaders(self):
        train_ds = self._CharDataset(self.train_features, self.train_labels)
        val_ds   = self._CharDataset(self.val_features, self.val_labels)
        test_ds  = self._CharDataset(self.test_features, self.test_labels)

        self.train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True,  collate_fn=self._collate)
        self.val_loader   = DataLoader(val_ds,   batch_size=self.batch_size, shuffle=False, collate_fn=self._collate)
        self.test_loader  = DataLoader(test_ds,  batch_size=self.batch_size, shuffle=False, collate_fn=self._collate)

    def get_training_batch(self, batch_size, train_features, train_labels, encode_text, _char_to_index):
        idx = np.random.randint(0, len(train_features), (batch_size,))
        x_sample = [self._encode_with_sos_eos_trim(s) for s in train_features[idx]]
        # Pad/truncate to seq_len
        max_len = min(max(len(s) for s in x_sample), self.seq_len)
        X = np.full((batch_size, max_len), self.pad_id, dtype=np.int64)
        for i, ids in enumerate(x_sample):
            ids = ids[:max_len]
            X[i, :len(ids)] = ids
        xb = torch.from_numpy(X).long().to(self.compute_device)
        yb = torch.from_numpy(train_labels[idx].astype(np.int64)).long().to(self.compute_device)
        return xb, yb

    # ---------------------- Model / Save-Load / Predict ----------------------
    def _build_model(self):
        model = MLPTextClassifier(
            vocab_size=self.vocabulary_size,
            embedding_dim=self.embedding_dim,
            hidden1=self.hidden1_size,
            hidden2=self.hidden2_size,
            output_size=self.output_size,
            padding_idx=self.pad_id,
            pooling=self.pooling,
            dropout_p=self.dropout_p
        ).to(self.compute_device)
        return model

    def _save_checkpoint(self, model):
        payload = {
            "model_state_dict": model.state_dict(),
            "char_to_index": self.char_to_index,
            "index_to_char": self.index_to_char,
            "vocabulary_size": self.vocabulary_size,
            "pad_id": self.pad_id,
            "sos_id": self.sos_id,
            "eos_id": self.eos_id,
            "seq_len": self.seq_len,
            "embedding_dim": self.embedding_dim,
            "hidden1_size": self.hidden1_size,
            "hidden2_size": self.hidden2_size,
            "output_size": self.output_size,
            "pooling": self.pooling,
            "dropout_p": self.dropout_p,
        }
        torch.save(payload, self.ckpt_path)
        print(f"Saved checkpoint to: {self.ckpt_path}")

    def _load_checkpoint(self):
        print(f"Loading checkpoint from: {self.ckpt_path}")
        payload = torch.load(self.ckpt_path, map_location=self.compute_device)

        self.char_to_index = payload["char_to_index"]
        self.index_to_char = payload["index_to_char"]
        self.vocabulary_size = payload["vocabulary_size"]
        self.pad_id = payload["pad_id"]
        self.sos_id = payload["sos_id"]
        self.eos_id = payload["eos_id"]
        self.seq_len = payload.get("seq_len", self.seq_len)
        self.embedding_dim = payload.get("embedding_dim", self.embedding_dim)
        self.hidden1_size = payload.get("hidden1_size", self.hidden1_size)
        self.hidden2_size = payload.get("hidden2_size", self.hidden2_size)
        self.output_size = payload.get("output_size", self.output_size)
        self.pooling = payload.get("pooling", self.pooling)
        self.dropout_p = payload.get("dropout_p", self.dropout_p)

        model = self._build_model()
        model.load_state_dict(payload["model_state_dict"])
        model.eval()
        self.model = model
        print("Checkpoint loaded and model restored.")

    @torch.no_grad()
    def predict_text(self, text: str):
        if self.model is None:
            raise RuntimeError("Model is not initialized. Train or load a model first.")
        self.model.eval()
        ids = self._encode_with_sos_eos_trim(str(text))
        max_len = min(len(ids), self.seq_len)
        X = np.full((1, max_len), self.pad_id, dtype=np.int64)
        X[0, :max_len] = np.array(ids[:max_len], dtype=np.int64)
        xb = torch.from_numpy(X).long().to(self.compute_device)
        logits = self.model(xb)
        probs = torch.softmax(logits, dim=-1).squeeze(0).detach().cpu().numpy()
        pred_idx = int(np.argmax(probs))
        return pred_idx, probs

    # ---------------------- Train / Eval ----------------------
    def TrainingModule(self):
        model = self._build_model()
        self.model = model

        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()

        global_step = 0
        for epoch in range(1, self.num_epochs + 1):
            model.train()
            running_loss = 0.0
            for xb, yb in self.train_loader:
                logits = model(xb)
                loss = criterion(logits, yb)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if (global_step % self.eval_interval) == 0 and global_step > 0:
                    val_acc, val_f1 = self._evaluate(model, self.val_loader)
                    print(f"[step {global_step}] val_acc={val_acc:.4f}  val_f1={val_f1:.4f}")
                    model.train()
                global_step += 1

            epoch_loss = running_loss / max(1, len(self.train_loader))
            val_acc, val_f1 = self._evaluate(model, self.val_loader)
            print(f"Epoch {epoch}/{self.num_epochs} | loss={epoch_loss:.4f} | val_acc={val_acc:.4f} | val_f1={val_f1:.4f}")
            model.train()

        test_acc, test_f1 = self._evaluate(model, self.test_loader, verbose=True)
        print(f"TEST | acc={test_acc:.4f} | f1={test_f1:.4f}")

        self._save_checkpoint(model)
        return model

    @torch.no_grad()
    def _evaluate(self, model, loader, verbose=False):
        was_training = model.training
        model.eval()
        y_true, y_pred = [], []
        for xb, yb in loader:
            logits = model(xb)
            pred = logits.argmax(dim=-1)
            y_true.extend(yb.detach().cpu().tolist())
            y_pred.extend(pred.detach().cpu().tolist())
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro")
        if verbose:
            print(classification_report(y_true, y_pred, digits=4))
        if was_training:
            model.train()
        return acc, f1


class MLPTextClassifier(nn.Module):
    """
    Embedding -> pooling (over time) -> Linear+ReLU+Dropout -> Linear+ReLU+Dropout -> Linear
    Pooling options: mean, max, or mean over [SOS, EOS] only.
    """
    def __init__(self, vocab_size, embedding_dim, hidden1, hidden2, output_size, padding_idx, pooling="mean", dropout_p=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.pooling = pooling
        self.dropout_p = dropout_p

        self.fc1 = nn.Linear(embedding_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc_out = nn.Linear(hidden2, output_size)
        self.dropout = nn.Dropout(p=dropout_p)
        self.act = nn.ReLU()

    def _pool(self, emb, x):
        # emb: [B, T, E], x: [B, T] integer ids
        if self.pooling == "max":
            pooled, _ = emb.max(dim=1)              # [B, E]
            return pooled
        elif self.pooling == "sos_eos_mean":
            # average of the first and (last non-pad) token embeddings
            B, T, E = emb.shape
            # find last non-pad position per sequence
            mask = (x != 0)  # padding_idx may not be 0; fix it by using padding_idx from embedding
            pad_id = self.embedding.padding_idx
            mask = (x != pad_id)
            lengths = mask.long().sum(dim=1).clamp(min=1)  # [B]
            last_idx = (lengths - 1).unsqueeze(1).expand(B, E)  # [B, E]
            # gather first and last
            first = emb[:, 0, :]                              # [B, E]
            last = emb[torch.arange(B), lengths - 1, :]       # [B, E]
            return 0.5 * (first + last)
        else:
            # mean pool over non-pad tokens
            pad_id = self.embedding.padding_idx
            mask = (x != pad_id).unsqueeze(-1).float()        # [B, T, 1]
            masked = emb * mask                                # zero out pads
            lengths = mask.sum(dim=1).clamp(min=1.0)           # [B, 1]
            return masked.sum(dim=1) / lengths                 # [B, E]

    def forward(self, x):
        emb = self.embedding(x)           # [B, T, E]
        pooled = self._pool(emb, x)       # [B, E]
        h1 = self.act(self.fc1(pooled))
        h1 = self.dropout(h1)
        h2 = self.act(self.fc2(h1))
        h2 = self.dropout(h2)
        logits = self.fc_out(h2)
        return logits


# ---------------------- CLI entry ----------------------
if __name__ == "__main__":
    SentimentObject = MovieReviewsMLPSentimentModel()

    use_saved = False
    if os.path.exists(SentimentObject.ckpt_path):
        try:
            resp = input(
                f"Found saved model at '{SentimentObject.ckpt_path}'. "
                f"Use it instead of retraining? [y/N]: "
            ).strip().lower()
            use_saved = (resp == "y")
        except Exception:
            use_saved = False

    if use_saved:
        SentimentObject._load_checkpoint()
    else:
        SentimentObject.TrainingModule()

    print("\nEnter a review to predict sentiment. Press ENTER on an empty line to quit.")
    while True:
        try:
            user_text = input("\nYour review: ").strip()
        except EOFError:
            break
        if not user_text:
            break
        pred_idx, probs = SentimentObject.predict_text(user_text)
        print(f"Predicted class index: {pred_idx} (0=negative, 1=positive)")
        print(f"Probabilities: class0={probs[0]:.4f}, class1={probs[1]:.4f}")
