# sentiment_menu_models_load_or_retrain.py
# Keyboard-driven IMDB Sentiment with:
# - Model selector menu (BERT / RoBERTa / DistilBERT / XLNet / ELECTRA / ALBERT / Custom)
# - When output_dir exists: ask to Load, Resume, Retrain (overwrite), or Train to New Dir
# - Attention rollout explanations across backbones
# - Version-safe TrainingArguments

import os
import re
import json
import shutil
import random
import inspect
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils import shuffle
from packaging import version

import transformers as _tfv
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

# Quieter logs
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

MODEL_MENU = {
    "1": ("BERT",       "bert-base-uncased"),
    "2": ("RoBERTa",    "roberta-base"),
    "3": ("DistilBERT", "distilbert-base-uncased"),
    "4": ("XLNet",      "xlnet-base-cased"),
    "5": ("ELECTRA",    "google/electra-base-discriminator"),
    "6": ("ALBERT",     "albert-base-v2"),
    "7": ("Custom",     None),  # user provides HF id
}

def prompt_model_choice(default_key: str = "1") -> str:
    print("\nSelect a backbone model:")
    for k in ["1","2","3","4","5","6","7"]:
        name, hub = MODEL_MENU[k]
        hint = hub or "enter a Hugging Face model id"
        print(f"  {k}. {name:<10} ({hint})")
    choice = input(f"\nEnter 1-7 (default {default_key}): ").strip() or default_key
    while choice not in MODEL_MENU:
        choice = input("Invalid choice. Enter 1-7: ").strip()
    if choice == "7":
        custom = input("Type the full HF model id (e.g., bert-large-uncased): ").strip()
        while not custom:
            custom = input("Please enter a non-empty model id: ").strip()
        return custom
    return MODEL_MENU[choice][1]

def prompt_int(prompt: str, default: int) -> int:
    s = input(f"{prompt} (default {default}): ").strip()
    if s == "": return default
    try: return int(s)
    except ValueError:
        print("Not an integer; using default.")
        return default

def prompt_float(prompt: str, default: float) -> float:
    s = input(f"{prompt} (default {default}): ").strip()
    if s == "": return default
    try: return float(s)
    except ValueError:
        print("Not a number; using default.")
        return default

def has_model_files(path: str) -> bool:
    needed = ["config.json", "tokenizer_config.json"]
    return os.path.isdir(path) and all(os.path.exists(os.path.join(path, f)) for f in needed)

def find_latest_checkpoint(path: str) -> Optional[str]:
    # HF Trainer saves checkpoint-NNNNN dirs
    if not os.path.isdir(path): return None
    cps = []
    for name in os.listdir(path):
        full = os.path.join(path, name)
        if os.path.isdir(full) and re.match(r"^checkpoint-\d+$", name):
            try:
                step = int(name.split("-")[-1])
                cps.append((step, full))
            except: pass
    if not cps: return None
    cps.sort(key=lambda x: x[0])
    return cps[-1][1]

def prompt_existing_dir_action(output_dir: str) -> str:
    """
    Returns one of: 'load', 'resume', 'retrain', 'newdir'
    """
    print(f'\nOutput directory "{output_dir}" already exists.')
    latest_cp = find_latest_checkpoint(output_dir)
    has_cp = latest_cp is not None
    print("What would you like to do?")
    print("  1) Load existing model for inference only")
    print(f"  2) Resume training{' (from latest checkpoint)' if has_cp else ' (no checkpoints found; will continue from saved weights)'}")
    print("  3) Retrain from scratch (OVERWRITE this folder)")
    print("  4) Train to a NEW folder (keep the old one)")
    choice = input("Enter 1-4 (default 1): ").strip() or "1"
    while choice not in {"1","2","3","4"}:
        choice = input("Invalid choice. Enter 1-4: ").strip()
    return {"1":"load", "2":"resume", "3":"retrain", "4":"newdir"}[choice]

class SentimentAnalysis:
    """
    IMDB sentiment analysis with selectable Transformer backbones
    and attention-rollout explanations.
    """

    class _TextLabelDataset(Dataset):
        def __init__(self, encodings: Dict[str, Any], labels: List[int]):
            self.encodings = encodings
            self.labels = labels
        def __len__(self): return len(self.labels)
        def __getitem__(self, idx):
            item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
            return item

    _LABEL_MAP = {"negative": 0, "positive": 1, "neg": 0, "pos": 1}

    def __init__(
        self,
        csv_path: str = "IMDB Dataset.csv",
        model_name: str = "bert-base-uncased",
        output_dir: str = "./imdb_sentiment_model",
        max_length: int = 256,
        seed: int = 42,
    ):
        self.csv_path = csv_path
        self.model_name = model_name
        self.output_dir = output_dir
        self.max_length = max_length
        self.seed = seed

        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.df: Optional[pd.DataFrame] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForSequenceClassification] = None
        self.trainer: Optional[Trainer] = None

    # ---------- data ----------
    @classmethod
    def _normalize_label(cls, v):
        if isinstance(v, str):
            vs = v.strip().lower()
            if vs in cls._LABEL_MAP: return cls._LABEL_MAP[vs]
            if vs.isdigit(): return int(vs)
            return 1 if ("pos" in vs or vs in {"1","true","yes"}) else 0
        return int(v)

    def _load_csv(self) -> pd.DataFrame:
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV not found at {self.csv_path}")
        df = pd.read_csv(self.csv_path)
        if not {"review","sentiment"}.issubset(df.columns):
            raise ValueError("CSV must have columns: 'review' and 'sentiment'")
        df = df[["review","sentiment"]].dropna().copy()
        df["label"] = df["sentiment"].apply(self._normalize_label)
        df["text"]  = df["review"].astype(str)
        return shuffle(df, random_state=self.seed).reset_index(drop=True)

    # ---------- tokenizer & model ----------
    def _prepare_tokenizer_and_model(self, init_from: Optional[str] = None):
        """
        init_from:
          - None -> load base backbone (self.model_name)
          - path/hub id -> load from there (e.g., resume from saved dir)
        """
        load_from = init_from or self.model_name
        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(load_from, use_fast=True)
        if self.tokenizer.pad_token is None:
            fallback = self.tokenizer.eos_token or self.tokenizer.sep_token or self.tokenizer.unk_token
            if fallback is None:
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        # model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            load_from, num_labels=2
        )
        # resize if we added tokens
        if self.model.get_input_embeddings().num_embeddings < len(self.tokenizer):
            self.model.resize_token_embeddings(len(self.tokenizer))
        if getattr(self.model.config, "pad_token_id", None) is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

    def _tokenize(self, texts: List[str]) -> Dict[str, Any]:
        return self.tokenizer(
            texts, truncation=True, max_length=self.max_length,
            padding=False, return_attention_mask=True
        )

    # ---------- metrics ----------
    @staticmethod
    def _compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        return {"accuracy": accuracy_score(labels, preds),
                "f1_weighted": f1_score(labels, preds, average="weighted")}

    # ---------- TrainingArguments (version-safe) ----------
    def _build_training_args(self, kw: Dict[str, Any]) -> TrainingArguments:
        try:
            return TrainingArguments(**kw)
        except TypeError:
            for k in ["evaluation_strategy","save_strategy","load_best_model_at_end",
                      "metric_for_best_model","greater_is_better","warmup_ratio","report_to"]:
                kw.pop(k, None)
            kw.setdefault("save_steps", 500)
            kw.setdefault("logging_steps", 50)
            return TrainingArguments(**kw)

    def _build_trainer(self, train_ds, val_ds, args, collator):
        from transformers import Trainer
        trainer_kwargs = dict(
            model=self.model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=collator,
            compute_metrics=self._compute_metrics,
        )
        # HF → v4: `tokenizer` (deprecated); v5: `processing_class`
        if "processing_class" in inspect.signature(Trainer.__init__).parameters:
            trainer_kwargs["processing_class"] = self.tokenizer
        else:
            trainer_kwargs["tokenizer"] = self.tokenizer
        return Trainer(**trainer_kwargs)

    # ---------- train/load ----------
    def fit(
        self,
        mode: str = "train",          # 'train' | 'resume' | 'load'
        epochs: int = 3,
        batch_size: int = 16,
        lr: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        print_report: bool = True,
    ) -> Dict[str, Any]:
        """
        mode:
          - 'load'   : just load existing self.output_dir and return
          - 'resume' : continue training (init from output_dir; resume from checkpoint if available)
          - 'train'  : train from base backbone (self.model_name), even if output_dir exists (assumes caller handled it)
        """
        if mode == "load":
            self.load()
            return {"note": "loaded_existing_model"}

        # data
        self.df = self._load_csv()

        # choose init
        init_from = self.output_dir if (mode == "resume" and has_model_files(self.output_dir)) else None
        self._prepare_tokenizer_and_model(init_from=init_from)

        # split
        X_train, X_val, y_train, y_val = train_test_split(
            self.df["text"].tolist(),
            self.df["label"].tolist(),
            test_size=0.1,
            random_state=self.seed,
            stratify=self.df["label"].tolist(),
        )
        train_ds = self._TextLabelDataset(self._tokenize(X_train), y_train)
        val_ds   = self._TextLabelDataset(self._tokenize(X_val),   y_val)
        collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        # args
        kw = dict(
            output_dir=self.output_dir,
            learning_rate=lr,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=weight_decay,
            logging_steps=50,
            fp16=torch.cuda.is_available(),
            save_total_limit=2,
        )
        if version.parse(_tfv.__version__) >= version.parse("4.4.0"):
            kw.update(
                evaluation_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="f1_weighted",
                greater_is_better=True,
                warmup_ratio=warmup_ratio,
                report_to=[],  # disable wandb/mlflow unless wanted
            )
        args = self._build_training_args(kw)

        # self.trainer = Trainer(
        #     model=self.model,
        #     args=args,
        #     train_dataset=train_ds,
        #     eval_dataset=val_ds,
        #     tokenizer=self.tokenizer,  # <-- this causes the warning
        #     data_collator=collator,
        #     compute_metrics=self._compute_metrics,
        # )

        self.trainer = self._build_trainer(train_ds, val_ds, args, collator)

        # resume from latest checkpoint if in resume mode and checkpoints exist
        resume_ckpt = find_latest_checkpoint(self.output_dir) if mode == "resume" else None
        self.trainer.train(resume_from_checkpoint=resume_ckpt if resume_ckpt else None)
        metrics = self.trainer.evaluate()

        if print_report:
            preds = np.argmax(self.trainer.predict(val_ds).predictions, axis=1)
            print("\nValidation metrics:", json.dumps(metrics, indent=2))
            print("\nClassification report (validation):")
            print(classification_report(y_val, preds, digits=4))

        self.trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        return metrics

    def load(self):
        if not has_model_files(self.output_dir):
            raise FileNotFoundError(f'No saved model at "{self.output_dir}"')
        self.tokenizer = AutoTokenizer.from_pretrained(self.output_dir, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.output_dir)
        self.model.eval()

    @classmethod
    def from_saved(cls, output_dir: str):
        obj = cls(csv_path="", model_name=output_dir, output_dir=output_dir)
        obj.load()
        return obj

    # ---------- inference ----------
    def predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        if self.model is None or self.tokenizer is None:
            self.load()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)
        enc = self.tokenizer(
            texts, truncation=True, max_length=self.max_length,
            padding=True, return_tensors="pt", return_attention_mask=True
        ).to(device)
        with torch.no_grad():
            out = self.model(**enc)
            probs = torch.softmax(out.logits, dim=-1).cpu().numpy()
        return [{
            "text": t,
            "pred_label": int(np.argmax(p)),
            "prob_negative": float(p[0]),
            "prob_positive": float(p[1]),
        } for t, p in zip(texts, probs)]

    # ---------- attention rollout ----------
    @staticmethod
    def _attention_rollout(attentions: List[torch.Tensor], cls_index: int) -> torch.Tensor:
        with torch.no_grad():
            attn = [a.mean(dim=1) for a in attentions]  # avg heads -> [B,T,T]
            attn = [a + torch.eye(a.size(-1), device=a.device).unsqueeze(0) for a in attn]
            attn = [a / a.sum(dim=-1, keepdim=True) for a in attn]
            joint = attn[0]
            for i in range(1, len(attn)):
                joint = torch.bmm(joint, attn[i])
            return joint[:, cls_index]  # [B,T]

    def _find_cls_index(self, input_ids: torch.Tensor) -> int:
        cls_id = self.tokenizer.cls_token_id
        ids = input_ids.tolist()
        pos = [i for i, tid in enumerate(ids) if tid == cls_id]
        if not pos: return 0
        # XLNet heuristic: CLS at end
        return pos[-1] if "xlnet" in (self.model_name or "").lower() else pos[0]

    def explain(self, text: str, top_k: int = 10) -> Dict[str, float]:
        if self.model is None or self.tokenizer is None:
            self.load()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)
        self.model.eval()
        inputs = self.tokenizer(
            text, truncation=True, max_length=self.max_length,
            return_tensors="pt", return_attention_mask=True
        ).to(device)
        with torch.no_grad():
            out = self.model(**inputs, output_attentions=True, return_dict=True)
            cls_idx = self._find_cls_index(inputs["input_ids"][0])
            rollout = self._attention_rollout(out.attentions, cls_index=cls_idx)[0]
            scores = rollout.detach().cpu().numpy()
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        keep = [i for i in range(len(tokens)) if i != cls_idx]
        tokens = [tokens[i] for i in keep]
        scores = [scores[i] for i in keep]
        merged: Dict[str, float] = {}
        cur, cur_s = "", 0.0
        for tok, s in zip(tokens, scores):
            clean = tok.replace("Ġ", " ")
            if clean.startswith("##"):
                cur += clean[2:]; cur_s += float(s)
            elif clean.startswith("▁"):
                if cur: merged[cur] = merged.get(cur, 0.0) + cur_s
                cur, cur_s = clean[1:], float(s)
            else:
                if cur: merged[cur] = merged.get(cur, 0.0) + cur_s
                cur, cur_s = clean, float(s)
        if cur: merged[cur] = merged.get(cur, 0.0) + cur_s
        specials = {self.tokenizer.cls_token, self.tokenizer.sep_token,
                    self.tokenizer.pad_token, self.tokenizer.eos_token, self.tokenizer.bos_token}
        return dict(sorted({k:v for k,v in merged.items() if k and k not in specials}.items(),
                           key=lambda x: x[1], reverse=True)[:top_k])

# ---------------- Main ----------------
def main():
    print("\n=== IMDB Sentiment (Keyboard Model Selector + Load/Resume/Retrain) ===")
    csv_path = input('Path to CSV (default "IMDB Dataset.csv"): ').strip() or "IMDB Dataset.csv"
    model_name = prompt_model_choice(default_key="1")
    output_dir = input('Output dir (default "./imdb_sentiment_model"): ').strip() or "./imdb_sentiment_model"

    # If out dir exists, ask how to proceed
    action = "train"
    if os.path.isdir(output_dir):
        action = prompt_existing_dir_action(output_dir)

    # If "newdir", ask for a new folder and set action to "train"
    if action == "newdir":
        new_out = input("New output dir (e.g., ./imdb_sentiment_model_v2): ").strip()
        while not new_out:
            new_out = input("Please enter a non-empty directory: ").strip()
        output_dir = new_out
        action = "train"

    # If "retrain", wipe the folder first (dangerous; confirm)
    if action == "retrain":
        confirm = input(f'CONFIRM delete "{output_dir}"? This CANNOT be undone. [y/N]: ').strip().lower()
        if confirm in ("y", "yes"):
            shutil.rmtree(output_dir, ignore_errors=True)
        else:
            print("Cancelled. Exiting."); return

    # Common training/inference params
    max_len = prompt_int("Max sequence length", 256)
    if action != "load":
        epochs  = prompt_int("Epochs", 3)
        batch   = prompt_int("Batch size", 16)
        lr      = prompt_float("Learning rate", 2e-5)
        wd      = prompt_float("Weight decay", 0.01)
        warmup  = prompt_float("Warmup ratio", 0.1)

    print(f"\nConfig:\n  CSV: {csv_path}\n  Model: {model_name}\n  Output: {output_dir}\n  max_len={max_len}")
    if action != "load":
        print(f"  epochs={epochs} batch={batch} lr={lr} wd={wd} warmup={warmup}")

    sa = SentimentAnalysis(
        csv_path=csv_path,
        model_name=model_name,
        output_dir=output_dir,
        max_length=max_len,
    )

    if action == "load":
        sa.fit(mode="load")  # just loads
        print("Loaded model. You can run predictions and explanations below.")
    elif action == "resume":
        sa.fit(mode="resume", epochs=epochs, batch_size=batch, lr=lr, weight_decay=wd, warmup_ratio=warmup)
    else:  # 'train' or 'retrain'
        sa.fit(mode="train",  epochs=epochs, batch_size=batch, lr=lr, weight_decay=wd, warmup_ratio=warmup)

    # Quick interactive demo
    if input("\nRun a quick prediction + attention explain demo? [Y/n]: ").strip().lower() in ("", "y", "yes"):
        texts = [
            "Absolutely loved it. One of the best films this year!",
            "This was painfully boring and a total waste of time.",
            "It was not so very good; I wouldn't recommend it.",
        ]
        print("\n(Press Enter to use defaults, or type your own sentences separated by ' | ')")
        user_line = input("Custom texts (optional): ").strip()
        if user_line:
            texts = [t.strip() for t in user_line.split("|") if t.strip()]
        preds = sa.predict(texts)
        print("\nPredictions:")
        for r in preds:
            print(json.dumps(r, indent=2))
        print("\nTop tokens by attention (first sample):")
        for tok, score in sa.explain(texts[0], top_k=10).items():
            print(f"{tok}: {score:.4f}")

if __name__ == "__main__":
    main()


# # sentiment_menu_models.py
# # IMDB Sentiment with a keyboard menu for selecting:
# # 1) BERT  2) RoBERTa  3) DistilBERT  4) XLNet  5) ELECTRA  6) ALBERT  7) Custom HF id
# # - Version-safe TrainingArguments (old/new transformers)
# # - Attention-based explanation (rollout) across models
# # - Reads "IMDB Dataset.csv" with columns: review, sentiment
#
# import os
# import json
# import random
# from typing import List, Dict, Any, Optional
#
# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, f1_score, classification_report
# from sklearn.utils import shuffle
# from packaging import version
#
# import transformers as _tfv
# import torch
# from torch.utils.data import Dataset
# from transformers import (
#     AutoTokenizer,
#     AutoModelForSequenceClassification,
#     DataCollatorWithPadding,
#     Trainer,
#     TrainingArguments,
# )
#
# # Tame verbose logs
# os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
# os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
#
# # ---------------- Model Menu ----------------
# MODEL_ZOO = {
#     "1": ("BERT",      "bert-base-uncased"),
#     "2": ("RoBERTa",   "roberta-base"),
#     "3": ("DistilBERT","distilbert-base-uncased"),
#     "4": ("XLNet",     "xlnet-base-cased"),
#     "5": ("ELECTRA",   "google/electra-base-discriminator"),
#     "6": ("ALBERT",    "albert-base-v2"),
#     "7": ("Custom",    None),  # ask user to type a HF hub id
# }
#
# def prompt_model_choice(default_key: str = "1") -> str:
#     print("\nSelect a backbone model:")
#     for k in ["1","2","3","4","5","6","7"]:
#         name = MODEL_ZOO[k][0]
#         ex   = MODEL_ZOO[k][1] or "enter a Hugging Face model id"
#         print(f"  {k}. {name}  ({ex})")
#     choice = input(f"\nEnter 1-7 (default {default_key}): ").strip()
#     if choice == "":
#         choice = default_key
#     while choice not in MODEL_ZOO:
#         choice = input("Invalid choice. Enter 1-7: ").strip()
#     if choice == "7":
#         custom = input("Type the full Hugging Face model id (e.g., bert-large-uncased): ").strip()
#         while not custom:
#             custom = input("Please enter a non-empty model id: ").strip()
#         return custom
#     return MODEL_ZOO[choice][1]
#
# def prompt_int(prompt: str, default: int) -> int:
#     s = input(f"{prompt} (default {default}): ").strip()
#     if s == "":
#         return default
#     try:
#         return int(s)
#     except ValueError:
#         print("Not an integer; using default.")
#         return default
#
# def prompt_float(prompt: str, default: float) -> float:
#     s = input(f"{prompt} (default {default}): ").strip()
#     if s == "":
#         return default
#     try:
#         return float(s)
#     except ValueError:
#         print("Not a number; using default.")
#         return default
#
# # --------------- SentimentAnalysis ---------------
# class SentimentAnalysis:
#     """
#     IMDB sentiment analysis with selectable Transformers backbones.
#
#     API:
#       - fit(): train & save model
#       - load(): load saved model
#       - predict(texts): list of dicts with label & probs
#       - explain(text): attention-rollout top tokens
#     """
#
#     class _TextLabelDataset(Dataset):
#         def __init__(self, encodings: Dict[str, Any], labels: List[int]):
#             self.encodings = encodings
#             self.labels = labels
#         def __len__(self):
#             return len(self.labels)
#         def __getitem__(self, idx):
#             item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
#             item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
#             return item
#
#     _LABEL_MAP = {"negative": 0, "positive": 1, "neg": 0, "pos": 1}
#
#     def __init__(
#         self,
#         csv_path: str = "IMDB Dataset.csv",
#         model_name: str = "bert-base-uncased",
#         output_dir: str = "./imdb_sentiment_model",
#         max_length: int = 256,
#         seed: int = 42,
#     ):
#         self.csv_path = csv_path
#         self.model_name = model_name
#         self.output_dir = output_dir
#         self.max_length = max_length
#         self.seed = seed
#
#         random.seed(seed)
#         np.random.seed(seed)
#         torch.manual_seed(seed)
#         torch.backends.cudnn.deterministic = True
#         torch.backends.cudnn.benchmark = False
#
#         self.df: Optional[pd.DataFrame] = None
#         self.tokenizer: Optional[AutoTokenizer] = None
#         self.model: Optional[AutoModelForSequenceClassification] = None
#         self.trainer: Optional[Trainer] = None
#
#     # ------------- data & labels -------------
#     @classmethod
#     def _normalize_label(cls, v):
#         if isinstance(v, str):
#             vs = v.strip().lower()
#             if vs in cls._LABEL_MAP:
#                 return cls._LABEL_MAP[vs]
#             if vs.isdigit():
#                 return int(vs)
#             return 1 if ("pos" in vs or vs == "1" or vs == "true" or vs == "yes") else 0
#         return int(v)
#
#     def _load_csv(self) -> pd.DataFrame:
#         if not os.path.exists(self.csv_path):
#             raise FileNotFoundError(f"CSV not found at {self.csv_path}")
#         df = pd.read_csv(self.csv_path)
#         if not {"review", "sentiment"}.issubset(df.columns):
#             raise ValueError("CSV must have columns: 'review' and 'sentiment'")
#         df = df[["review", "sentiment"]].dropna().copy()
#         df["label"] = df["sentiment"].apply(self._normalize_label)
#         df["text"] = df["review"].astype(str)
#         return shuffle(df, random_state=self.seed).reset_index(drop=True)
#
#     # ------------- tokenizer & model -------------
#     def _prepare_tokenizer_and_model(self):
#         self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
#         # Ensure PAD token exists/known
#         if self.tokenizer.pad_token is None:
#             fallback = self.tokenizer.eos_token or self.tokenizer.sep_token or self.tokenizer.unk_token
#             if fallback is None:
#                 self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
#         self.model = AutoModelForSequenceClassification.from_pretrained(
#             self.model_name, num_labels=2
#         )
#         # If we added tokens, resize embeddings
#         if self.model.get_input_embeddings().num_embeddings < len(self.tokenizer):
#             self.model.resize_token_embeddings(len(self.tokenizer))
#         if getattr(self.model.config, "pad_token_id", None) is None:
#             self.model.config.pad_token_id = self.tokenizer.pad_token_id
#
#     def _tokenize(self, texts: List[str]) -> Dict[str, Any]:
#         return self.tokenizer(
#             texts, truncation=True, max_length=self.max_length,
#             padding=False, return_attention_mask=True,
#         )
#
#     # ------------- metrics -------------
#     @staticmethod
#     def _compute_metrics(eval_pred):
#         logits, labels = eval_pred
#         preds = logits.argmax(axis=-1)
#         return {
#             "accuracy": accuracy_score(labels, preds),
#             "f1_weighted": f1_score(labels, preds, average="weighted"),
#         }
#
#     # ------------- TrainingArguments (version-safe) -------------
#     def _build_training_args(self, kw: Dict[str, Any]) -> TrainingArguments:
#         try:
#             return TrainingArguments(**kw)
#         except TypeError:
#             for k in [
#                 "evaluation_strategy","save_strategy","load_best_model_at_end",
#                 "metric_for_best_model","greater_is_better","warmup_ratio","report_to",
#             ]:
#                 kw.pop(k, None)
#             kw.setdefault("save_steps", 500)
#             kw.setdefault("logging_steps", 50)
#             return TrainingArguments(**kw)
#
#     # ------------- train -------------
#     def fit(
#         self,
#         epochs: int = 3,
#         batch_size: int = 16,
#         lr: float = 2e-5,
#         weight_decay: float = 0.01,
#         warmup_ratio: float = 0.1,
#         retrain: bool = False,
#         print_report: bool = True,
#     ) -> Dict[str, float]:
#         if os.path.isdir(self.output_dir) and not retrain:
#             self.load()
#             return {"note": "loaded_existing_model"}
#
#         self.df = self._load_csv()
#         self._prepare_tokenizer_and_model()
#
#         X_train, X_val, y_train, y_val = train_test_split(
#             self.df["text"].tolist(),
#             self.df["label"].tolist(),
#             test_size=0.1,
#             random_state=self.seed,
#             stratify=self.df["label"].tolist(),
#         )
#
#         train_ds = self._TextLabelDataset(self._tokenize(X_train), y_train)
#         val_ds   = self._TextLabelDataset(self._tokenize(X_val),   y_val)
#         collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
#
#         kw = dict(
#             output_dir=self.output_dir,
#             learning_rate=lr,
#             per_device_train_batch_size=batch_size,
#             per_device_eval_batch_size=batch_size,
#             num_train_epochs=epochs,
#             weight_decay=weight_decay,
#             logging_steps=50,
#             fp16=torch.cuda.is_available(),
#             save_total_limit=2,
#         )
#         hf_ver = version.parse(_tfv.__version__)
#         if hf_ver >= version.parse("4.4.0"):
#             kw.update(
#                 evaluation_strategy="epoch",
#                 save_strategy="epoch",
#                 load_best_model_at_end=True,
#                 metric_for_best_model="f1_weighted",
#                 greater_is_better=True,
#                 warmup_ratio=warmup_ratio,
#                 report_to=[],  # no wandb/mlflow unless you add it
#             )
#         args = self._build_training_args(kw)
#
#         self.trainer = Trainer(
#             model=self.model,
#             args=args,
#             train_dataset= train_ds,
#             eval_dataset=  val_ds,
#             tokenizer=self.tokenizer,
#             data_collator=collator,
#             compute_metrics=self._compute_metrics,
#         )
#
#         self.trainer.train()
#         metrics = self.trainer.evaluate()
#
#         if print_report:
#             preds = np.argmax(self.trainer.predict(val_ds).predictions, axis=1)
#             print("\nValidation metrics:", json.dumps(metrics, indent=2))
#             print("\nClassification report (validation):")
#             print(classification_report(y_val, preds, digits=4))
#
#         self.trainer.save_model(self.output_dir)
#         self.tokenizer.save_pretrained(self.output_dir)
#         return metrics
#
#     # ------------- load/save -------------
#     def load(self):
#         if not os.path.isdir(self.output_dir):
#             raise FileNotFoundError(f"No saved model at {self.output_dir}")
#         self.tokenizer = AutoTokenizer.from_pretrained(self.output_dir, use_fast=True)
#         self.model = AutoModelForSequenceClassification.from_pretrained(self.output_dir)
#         self.model.eval()
#
#     @classmethod
#     def from_saved(cls, output_dir: str):
#         obj = cls(csv_path="", model_name="bert-base-uncased", output_dir=output_dir)
#         obj.load()
#         return obj
#
#     # ------------- inference -------------
#     def predict(self, texts: List[str]) -> List[Dict[str, Any]]:
#         if self.model is None or self.tokenizer is None:
#             self.load()
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.model.to(device)
#
#         enc = self.tokenizer(
#             texts, truncation=True, max_length=self.max_length,
#             padding=True, return_tensors="pt", return_attention_mask=True,
#         ).to(device)
#
#         with torch.no_grad():
#             out = self.model(**enc)
#             probs = torch.softmax(out.logits, dim=-1).cpu().numpy()
#
#         results = []
#         for t, p in zip(texts, probs):
#             results.append({
#                 "text": t,
#                 "pred_label": int(np.argmax(p)),
#                 "prob_negative": float(p[0]),
#                 "prob_positive": float(p[1]),
#             })
#         return results
#
#     # ------------- attention explanations -------------
#     @staticmethod
#     def _attention_rollout(attentions: List[torch.Tensor], cls_index: int) -> torch.Tensor:
#         with torch.no_grad():
#             attn = [a.mean(dim=1) for a in attentions]  # avg heads -> [B,T,T]
#             attn = [a + torch.eye(a.size(-1), device=a.device).unsqueeze(0) for a in attn]  # +I
#             attn = [a / a.sum(dim=-1, keepdim=True) for a in attn]  # row-norm
#             joint = attn[0]
#             for i in range(1, len(attn)):
#                 joint = torch.bmm(joint, attn[i])  # compose across layers
#             return joint[:, cls_index]  # [B,T]
#
#     def _find_cls_index(self, input_ids: torch.Tensor) -> int:
#         cls_id = self.tokenizer.cls_token_id
#         ids = input_ids.tolist()
#         positions = [i for i, tid in enumerate(ids) if tid == cls_id]
#         if not positions:
#             return 0
#         name = self.model_name.lower()
#         if "xlnet" in name:
#             return positions[-1]  # XLNet puts <cls> at end
#         return positions[0]
#
#     def explain(self, text: str, top_k: int = 10) -> Dict[str, float]:
#         if self.model is None or self.tokenizer is None:
#             self.load()
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.model.to(device)
#         self.model.eval()
#
#         inputs = self.tokenizer(
#             text, truncation=True, max_length=self.max_length,
#             return_tensors="pt", return_attention_mask=True,
#         ).to(device)
#
#         with torch.no_grad():
#             out = self.model(**inputs, output_attentions=True, return_dict=True)
#             cls_index = self._find_cls_index(inputs["input_ids"][0])
#             rollout = self._attention_rollout(out.attentions, cls_index=cls_index)[0]
#             scores = rollout.detach().cpu().numpy()
#
#         tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
#         keep = [i for i in range(len(tokens)) if i != cls_index]
#         tokens = [tokens[i] for i in keep]
#         scores = [scores[i] for i in keep]
#
#         merged: Dict[str, float] = {}
#         cur_word, cur_score = "", 0.0
#         for tok, s in zip(tokens, scores):
#             clean = tok.replace("Ġ", " ")  # RoBERTa space marker
#             if clean.startswith("##"):
#                 cur_word += clean[2:]
#                 cur_score += float(s)
#             elif clean.startswith("▁"):     # sentencepiece (ALBERT/XLNet)
#                 if cur_word:
#                     merged[cur_word] = merged.get(cur_word, 0.0) + cur_score
#                 cur_word, cur_score = clean[1:], float(s)
#             else:
#                 if cur_word:
#                     merged[cur_word] = merged.get(cur_word, 0.0) + cur_score
#                 cur_word, cur_score = clean, float(s)
#         if cur_word:
#             merged[cur_word] = merged.get(cur_word, 0.0) + cur_score
#
#         specials = {self.tokenizer.cls_token, self.tokenizer.sep_token,
#                     self.tokenizer.pad_token, self.tokenizer.eos_token, self.tokenizer.bos_token}
#         merged = {k: v for k, v in merged.items() if k and (k not in specials)}
#
#         return dict(sorted(merged.items(), key=lambda x: x[1], reverse=True)[:top_k])
#
# # ---------------- Main (interactive) ----------------
# def main():
#     print("\n=== IMDB Sentiment (Keyboard Model Selector) ===")
#     csv_path = input('Path to CSV (default "IMDB Dataset.csv"): ').strip() or "IMDB Dataset.csv"
#     model_name = prompt_model_choice(default_key="1")
#     output_dir = input('Output dir (default "./imdb_sentiment_model"): ').strip() or "./imdb_sentiment_model"
#     max_len = prompt_int("Max sequence length", 256)
#     epochs  = prompt_int("Epochs", 3)
#     batch   = prompt_int("Batch size", 16)
#     lr      = prompt_float("Learning rate", 2e-5)
#     wd      = prompt_float("Weight decay", 0.01)
#     warmup  = prompt_float("Warmup ratio", 0.1)
#     retrain = input("Retrain if folder exists? [y/N]: ").strip().lower().startswith("y")
#
#     print(f"\nConfig:\n  Model: {model_name}\n  CSV: {csv_path}\n  Out: {output_dir}\n"
#           f"  max_len={max_len}  epochs={epochs}  batch={batch}  lr={lr}  wd={wd}  warmup={warmup}\n")
#
#     sa = SentimentAnalysis(
#         csv_path=csv_path,
#         model_name=model_name,
#         output_dir=output_dir,
#         max_length=max_len,
#     )
#
#     metrics = sa.fit(
#         epochs=epochs,
#         batch_size=batch,
#         lr=lr,
#         weight_decay=wd,
#         warmup_ratio=warmup,
#         retrain=retrain,
#         print_report=True,
#     )
#     print("\nFinished training/eval. Metrics:\n", json.dumps(metrics, indent=2))
#
#     # Quick demo
#     if input("\nRun a quick prediction + attention explain demo? [Y/n]: ").strip().lower() in ("", "y", "yes"):
#         samples = [
#             "Absolutely loved it. One of the best films this year!",
#             "This was painfully boring and a total waste of time.",
#             "It was not so very good; I wouldn't recommend it.",
#         ]
#         preds = sa.predict(samples)
#         print("\nDemo predictions:")
#         for r in preds:
#             print(json.dumps(r, indent=2))
#         print("\nTop tokens by attention (sample 1):")
#         for tok, score in sa.explain(samples[0], top_k=10).items():
#             print(f"{tok}: {score:.4f}")
#
# if __name__ == "__main__":
#     main()
#
# # F:/Sentiment_Analysis_Master/MasterCode/Transformer Model/IMDB Dataset.csv