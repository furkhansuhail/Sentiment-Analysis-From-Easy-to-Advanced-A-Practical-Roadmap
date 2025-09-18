import os, re, json, glob, inspect, random
from typing import List, Dict, Any, Optional, Iterable, Tuple

import numpy as np
import pandas as pd
from packaging import version

import torch
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset

import transformers as _tfv
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForMaskedLM,
    DataCollatorWithPadding,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

# --- PEFT (LoRA) ---
try:
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
    _HAVE_PEFT = True
except Exception:
    _HAVE_PEFT = False

# -----------------------
# Utilities
# -----------------------
def seed_all(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def has_model_files(path: str) -> bool:
    needed = ["config.json", "tokenizer_config.json"]
    return os.path.isdir(path) and all(os.path.exists(os.path.join(path, f)) for f in needed)

def find_latest_checkpoint(path: str) -> Optional[str]:
    if not os.path.isdir(path): return None
    cps = []
    for name in os.listdir(path):
        full = os.path.join(path, name)
        if os.path.isdir(full) and re.match(r"^checkpoint-\d+$", name):
            try: cps.append((int(name.split("-")[-1]), full))
            except: pass
    if not cps: return None
    cps.sort(key=lambda x: x[0])
    return cps[-1][1]

# -----------------------
# Core Sentiment Class
# -----------------------
class SentimentPEFTDAPT:
    """
    Advanced Transformer Techniques:
      - LoRA/PEFT for parameter-efficient fine-tuning
      - Domain-Adaptive Pretraining (DAPT) via MLM on raw domain text
    """

    _LABEL_MAP = {"negative": 0, "positive": 1, "neg": 0, "pos": 1}

    class _TextLabelDataset(Dataset):
        def __init__(self, encodings: Dict[str, Any], labels: List[int]):
            self.encodings = encodings; self.labels = labels
        def __len__(self): return len(self.labels)
        def __getitem__(self, idx):
            item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
            return item

    def __init__(
        self,
        csv_path: str = "IMDB Dataset.csv",
        model_name: str = "bert-base-uncased",
        output_dir: str = "./imdb_advanced",
        max_length: int = 256,
        seed: int = 42,
        peft: Optional[str] = None,      # "lora" or None
        load_in_8bit: bool = False,      # requires bitsandbytes
        dapt_corpus: Optional[str] = None, # folder of .txt or a .txt/.csv(file has 'text' col)
        dapt_steps: int = 2000,
        dapt_lr: float = 5e-5,
        dapt_batch_size: int = 16,
    ):
        seed_all(seed)
        self.csv_path = csv_path
        self.model_name = model_name
        self.output_dir = output_dir
        self.max_length = max_length
        self.seed = seed
        self.peft = (peft or "").lower() or None
        self.load_in_8bit = load_in_8bit
        self.dapt_corpus = dapt_corpus
        self.dapt_steps = dapt_steps
        self.dapt_lr = dapt_lr
        self.dapt_batch_size = dapt_batch_size

        self.df: Optional[pd.DataFrame] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForSequenceClassification] = None
        self.trainer: Optional[Trainer] = None

    # ---------- IO ----------
    @classmethod
    def _normalize_label(cls, v):
        if isinstance(v, str):
            vs = v.strip().lower()
            if vs in cls._LABEL_MAP: return cls._LABEL_MAP[vs]
            if vs.isdigit(): return int(vs)
            return 1 if ("pos" in vs or vs in {"1", "true", "yes"}) else 0
        return int(v)

    def _load_csv(self) -> pd.DataFrame:
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV not found at {self.csv_path}")
        df = pd.read_csv(self.csv_path)
        need = {"review","sentiment"}
        if not need.issubset(df.columns):
            raise ValueError("CSV must have columns: 'review' and 'sentiment'")
        df = df[["review","sentiment"]].dropna().copy()
        df["label"] = df["sentiment"].apply(self._normalize_label)
        df["text"]  = df["review"].astype(str)
        return df.sample(frac=1.0, random_state=self.seed).reset_index(drop=True)

    # ---------- Tokenizer ----------
    def _prepare_tokenizer(self, from_path: Optional[str] = None):
        load_from = from_path or self.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(load_from, use_fast=True)
        if self.tokenizer.pad_token is None:
            fallback = self.tokenizer.eos_token or self.tokenizer.sep_token or self.tokenizer.unk_token
            if fallback is None:
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    def _tokenize(self, texts: List[str]) -> Dict[str, Any]:
        return self.tokenizer(
            texts, truncation=True, max_length=self.max_length,
            padding=False, return_attention_mask=True
        )

    # ---------- Version-safe Trainer ----------
    @staticmethod
    def _compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        from sklearn.metrics import accuracy_score, f1_score
        return {"accuracy": accuracy_score(labels, preds),
                "f1_weighted": f1_score(labels, preds, average="weighted")}

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

    def _build_trainer(self, model, train_ds, val_ds, args, collator):
        trainer_kwargs = dict(
            model=model, args=args,
            train_dataset=train_ds, eval_dataset=val_ds,
            data_collator=collator,
            compute_metrics=self._compute_metrics,
        )
        if "processing_class" in inspect.signature(Trainer.__init__).parameters:
            trainer_kwargs["processing_class"] = self.tokenizer
        else:
            trainer_kwargs["tokenizer"] = self.tokenizer
        return Trainer(**trainer_kwargs)

    # ---------- LoRA helpers ----------
    @staticmethod
    def _guess_lora_targets(model) -> List[str]:
        """
        Returns module name fragments commonly used across architectures.
        Covers BERT/RoBERTa/DistilBERT/ALBERT/XLNet/ELECTRA.
        """
        candidates = ["q_proj", "k_proj", "v_proj", "o_proj",  # LLaMA/OPT-style
                      "query", "key", "value", "dense"]        # BERT-style
        present = set()
        for n, _ in model.named_modules():
            # keep only the tail segment
            tail = n.split(".")[-1]
            for c in candidates:
                if tail == c or c in tail:
                    present.add(c)
        # Favor attention projections if available, else fall back to dense
        ordered = [x for x in ["q_proj","k_proj","v_proj","o_proj","query","key","value","dense"] if x in present]
        return ordered or ["dense"]

    def _wrap_with_lora(self, model):
        if not _HAVE_PEFT:
            raise RuntimeError("peft is not installed. Run: pip install peft")
        target_modules = self._guess_lora_targets(model)
        lcfg = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=16, lora_alpha=32, lora_dropout=0.05,
            bias="none",
            target_modules=target_modules
        )
        # Optional: prepare for k-bit
        if self.load_in_8bit:
            model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lcfg)
        model.print_trainable_parameters()
        return model

    # ---------- DAPT (Domain Adaptive Pretraining) ----------
    def _load_dapt_texts(self, path: str) -> List[str]:
        texts: List[str] = []
        if os.path.isdir(path):
            for p in glob.glob(os.path.join(path, "**/*.txt"), recursive=True):
                with open(p, "r", encoding="utf-8", errors="ignore") as f:
                    t = f.read().strip()
                if t: texts.append(t)
        else:
            if path.endswith(".csv"):
                df = pd.read_csv(path)
                col = "text" if "text" in df.columns else df.columns[0]
                texts = [str(x) for x in df[col].dropna().tolist()]
            else:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    texts = [line.strip() for line in f if line.strip()]
        return texts

    def run_dapt(self) -> Optional[str]:
        if not self.dapt_corpus:
            return None
        print(f"[DAPT] Loading domain corpus from: {self.dapt_corpus}")
        self._prepare_tokenizer()
        texts = self._load_dapt_texts(self.dapt_corpus)
        print(f"[DAPT] Loaded {len(texts)} texts")

        # Build MLM dataset
        def _tok(batch_texts: List[str]) -> Dict[str, Any]:
            return self.tokenizer(
                batch_texts, truncation=True, max_length=self.max_length,
                padding="max_length"
            )
        ds = HFDataset.from_dict({"text": texts}).map(
            lambda ex: _tok(ex["text"]), batched=True, remove_columns=["text"]
        )

        mlm_model = AutoModelForMaskedLM.from_pretrained(self.model_name)
        if mlm_model.get_input_embeddings().num_embeddings < len(self.tokenizer):
            mlm_model.resize_token_embeddings(len(self.tokenizer))

        collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=True, mlm_probability=0.15)

        args = self._build_training_args(dict(
            output_dir=os.path.join(self.output_dir, "dapt_mlm"),
            per_device_train_batch_size=self.dapt_batch_size,
            per_device_eval_batch_size=self.dapt_batch_size,
            learning_rate=self.dapt_lr,
            num_train_epochs=1,              # step-based, so set epochs=1
            logging_steps=50,
            save_total_limit=1,
            max_steps=self.dapt_steps,       # the main DAPT knob
            report_to=[],
            evaluation_strategy="no",
            save_strategy="steps",
            save_steps=500,
            fp16=torch.cuda.is_available(),
        ))

        trainer = self._build_trainer(mlm_model, ds, None, args, collator)
        trainer.train()
        dapt_path = os.path.join(self.output_dir, "dapt_adapted_backbone")
        trainer.save_model(dapt_path)
        self.tokenizer.save_pretrained(dapt_path)
        print(f"[DAPT] Saved domain-adapted backbone to: {dapt_path}")
        return dapt_path

    # ---------- Fit / Load / Predict / Explain ----------
    def fit(
        self,
        mode: str = "train",      # 'train' | 'resume' | 'load'
        epochs: int = 3,
        batch_size: int = 16,
        lr: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        print_report: bool = True,
    ):
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report

        if mode == "load":
            return self.load()

        # DAPT (optional) -> sets init path
        init_from = None
        if self.dapt_corpus:
            init_from = self.run_dapt()

        # Tokenizer
        self._prepare_tokenizer(from_path=init_from)

        # Classifier
        model_load_id = init_from or self.model_name
        cls_model = AutoModelForSequenceClassification.from_pretrained(model_load_id, num_labels=2)

        # LoRA (optional)
        if self.peft == "lora":
            cls_model = self._wrap_with_lora(cls_model)

        # If tokenizer added PAD token after load, resize embeddings
        if cls_model.get_input_embeddings().num_embeddings < len(self.tokenizer):
            cls_model.resize_token_embeddings(len(self.tokenizer))
        if getattr(cls_model.config, "pad_token_id", None) is None:
            cls_model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model = cls_model

        # Data
        self.df = self._load_csv()
        X_train, X_val, y_train, y_val = train_test_split(
            self.df["text"].tolist(),
            self.df["label"].tolist(),
            test_size=0.1, random_state=self.seed, stratify=self.df["label"].tolist()
        )
        train_ds = self._TextLabelDataset(self._tokenize(X_train), y_train)
        val_ds   = self._TextLabelDataset(self._tokenize(X_val),   y_val)
        collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        # Args
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
                report_to=[],
            )
        args = self._build_training_args(kw)

        # Resume support
        resume_ckpt = find_latest_checkpoint(self.output_dir) if mode == "resume" else None
        self.trainer = self._build_trainer(self.model, train_ds, val_ds, args, collator)
        self.trainer.train(resume_from_checkpoint=resume_ckpt if resume_ckpt else None)
        metrics = self.trainer.evaluate()

        if print_report:
            preds = np.argmax(self.trainer.predict(val_ds).predictions, axis=1)
            print("\nValidation metrics:", json.dumps(metrics, indent=2))
            print("\nClassification report (validation):")
            print(classification_report(y_val, preds, digits=4))

        # Save: if LoRA, this stores adapters; if full model, stores full weights
        self.trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        return metrics

    def load(self):
        if not has_model_files(self.output_dir):
            raise FileNotFoundError(f'No saved model at "{self.output_dir}"')
        self.tokenizer = AutoTokenizer.from_pretrained(self.output_dir, use_fast=True)
        # Try to load with PEFT if adapters exist
        try:
            if _HAVE_PEFT and os.path.exists(os.path.join(self.output_dir, "adapter_config.json")):
                base = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=2)
                from peft import PeftModel
                self.model = PeftModel.from_pretrained(base, self.output_dir)
            else:
                self.model = AutoModelForSequenceClassification.from_pretrained(self.output_dir)
        except Exception:
            self.model = AutoModelForSequenceClassification.from_pretrained(self.output_dir)
        self.model.eval()
        return {"note": "loaded_existing_model"}

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

    # — Attention rollout identical to your script —
    @staticmethod
    def _attention_rollout(attentions: List[torch.Tensor], cls_index: int) -> torch.Tensor:
        with torch.no_grad():
            attn = [a.mean(dim=1) for a in attentions]  # [B,H,T,T] -> [B,T,T]
            attn = [a + torch.eye(a.size(-1), device=a.device).unsqueeze(0) for a in attn]
            attn = [a / a.sum(dim=-1, keepdim=True) for a in attn]
            joint = attn[0]
            for i in range(1, len(attn)):
                joint = torch.bmm(joint, attn[i])
            return joint[:, cls_index]

    def _find_cls_index(self, input_ids: torch.Tensor) -> int:
        cls_id = self.tokenizer.cls_token_id
        ids = input_ids.tolist()
        pos = [i for i, tid in enumerate(ids) if tid == cls_id]
        if not pos: return 0
        # XLNet places CLS at end
        name = (self.model_name or "").lower()
        return pos[-1] if "xlnet" in name else pos[0]

    def explain(self, text: str, top_k: int = 10) -> Dict[str, float]:
        if self.model is None or self.tokenizer is None:
            self.load()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device); self.model.eval()
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
            if clean.startswith("##"): cur += clean[2:]; cur_s += float(s)
            elif clean.startswith("▁"):
                if cur: merged[cur] = merged.get(cur, 0.0) + cur_s
                cur, cur_s = clean[1:], float(s)
            else:
                if cur: merged[cur] = merged.get(cur, 0.0) + cur_s
                cur, cur_s = clean, float(s)
        if cur: merged[cur] = merged.get(cur, 0.0) + cur_s
        specials = {self.tokenizer.cls_token, self.tokenizer.sep_token,
                    self.tokenizer.pad_token, self.tokenizer.eos_token, self.tokenizer.bos_token}
        merged = {k:v for k,v in merged.items() if k and k not in specials}
        return dict(sorted(merged.items(), key=lambda x: x[1], reverse=True)[:top_k])


# ------------- CLI Demo -------------
if __name__ == "__main__":
    print("\n=== Advanced Transformers: LoRA PEFT + DAPT ===")
    csv = input('IMDB CSV path (default "IMDB Dataset.csv"): ').strip() or "IMDB Dataset.csv"
    model = input('Backbone (default "bert-base-uncased"): ').strip() or "bert-base-uncased"
    out   = input('Output dir (default "./imdb_advanced"): ').strip() or "./imdb_advanced"
    maxlen= input("Max length (default 256): ").strip() or "256"
    peft  = input('PEFT mode [none|lora] (default "lora"): ').strip().lower() or "lora"
    dcor  = input("DAPT corpus path (.txt or folder or .csv w/ 'text' col) [optional]: ").strip() or None

    runner = SentimentPEFTDAPT(
        csv_path=csv, model_name=model, output_dir=out,
        max_length=int(maxlen), peft=("lora" if peft!="none" else None),
        dapt_corpus=dcor
    )

    # train (with DAPT if provided)
    runner.fit(mode="train", epochs=3, batch_size=16, lr=2e-5, weight_decay=0.01)

    # quick check
    preds = runner.predict([
        "Absolutely loved it. One of the best films this year!",
        "This was painfully boring and a total waste of time.",
        "It was not so very good; I wouldn't recommend it."
    ])
    print("\nPredictions:")
    for r in preds:
        print(json.dumps(r, indent=2))

# F:/Sentiment_Analysis_Master/MasterCode/Advanced Transformer Techniques/IMDB Dataset.csv