#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Rule-based sentiment analysis over a dataset with a 'Review' column
using VADER, TextBlob, and AFINN — with tqdm progress bars.

- VADER: lexicon + heuristics (ALLCAPS, punctuation, emojis, degree words, negation)
- TextBlob: polarity/subjectivity from pattern-based rules
- AFINN: integer valence per token; here extended with simple negation + booster rules
- Explicit, documented threshold rules for labels (positive/neutral/negative)
- tqdm progress bars for row-wise processing
- Optional lexicon overrides for domain-specific tuning

Install (once):
    pip install pandas nltk textblob afinn tqdm
    python - <<'PY'
import nltk
nltk.download('vader_lexicon')
PY
    # (Optional) for some TextBlob extras:
    # python -m textblob.download_corpora
"""

import re
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List
from tqdm import tqdm
import pandas as pd

# Use notebook-friendly tqdm if you're in Jupyter; otherwise plain tqdm also works.
try:
    from tqdm.notebook import tqdm
except Exception:
    from tqdm import tqdm

# --- VADER (NLTK)
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# --- TextBlob (pattern-based)
from textblob import TextBlob

# --- AFINN (simple lexicon)
from afinn import Afinn


# ------------------------------------------------------------------------------
# Threshold configuration
# ------------------------------------------------------------------------------
@dataclass
class Thresholds:
    """
    Neutral 'gray zones' around 0 prevent overconfident labels on weak signals.
    Tune these based on validation data if you have labels.
    """
    # VADER 'compound' score is in [-1, 1]; authors suggest ±0.05 default.
    vader_neg_pos: Tuple[float, float] = (-0.05, 0.05)

    # TextBlob polarity is in [-1, 1]; slightly wider neutral band is common.
    textblob_neg_pos: Tuple[float, float] = (-0.10, 0.10)

    # AFINN raw sum is length-biased. We normalize by token count (see below)
    # and then use a small gray zone ±0.10.
    afinn_neg_pos: Tuple[float, float] = (-0.10, 0.10)


# ------------------------------------------------------------------------------
# Core analyzer (VADER, TextBlob, AFINN with explicit rules)
# ------------------------------------------------------------------------------
class RuleBasedSentiment:
    """
    Three rule-based methods:
      1) VADER   : Valence-aware dictionary + heuristics (caps/punct/emoji/modifiers/negation)
      2) TextBlob: Polarity/subjectivity from pattern-based lexicon/grammar
      3) AFINN   : Integer valence per token; we add negation/booster/overrides + normalization
    """

    def __init__(
        self,
        thresholds: Thresholds = Thresholds(),
        lexicon_overrides: Optional[Dict[str, int]] = None,
        language: str = "en",
        afinn_emoticons: bool = True,
        lowercase_for_rules: bool = True,
        negation_window: int = 3,
    ):
        """
        Parameters
        ----------
        thresholds : Thresholds
            Negative/neutral/positive cutoffs for each method.

        lexicon_overrides : dict[str, int], optional
            Domain lexicon tweaks (applied to VADER + AFINN):
            e.g., {'GOAT': 4, 'buggy': -3}.

        language : str
            AFINN language (default 'en').

        afinn_emoticons : bool
            Whether AFINN recognizes emoticons like ':-)'.

        lowercase_for_rules : bool
            Lowercase tokens for AFINN rules to normalize matching.

        negation_window : int
            For AFINN rule: how many tokens ahead a negation/booster can affect.
        """
        self.thresholds = thresholds
        self.lowercase_for_rules = lowercase_for_rules
        self.negation_window = max(1, int(negation_window))

        # --- Ensure the VADER lexicon is available
        try:
            nltk.data.find("sentiment/vader_lexicon.zip")
        except LookupError:
            nltk.download("vader_lexicon")

        # --- Initialize analyzers
        self.vader = SentimentIntensityAnalyzer()
        if lexicon_overrides:
            # VADER supports updates directly (keys should be lowercase)
            self.vader.lexicon.update({k.lower(): float(v) for k, v in lexicon_overrides.items()})

        # TextBlob is used via class reference
        self.textblob = TextBlob

        # AFINN and override store
        self.afinn = Afinn(language=language, emoticons=afinn_emoticons)
        self.afinn_overrides = {k.lower(): int(v) for k, v in (lexicon_overrides or {}).items()}

        # Simple token regex: words with letters, apostrophes, hyphens (e.g., it's, state-of-the-art)
        self._token_re = re.compile(r"[A-Za-z][A-Za-z'\-]+")

        # Negation cues (extend for your domain)
        self._negations = {
            "not", "no", "never", "n't", "cannot", "can't", "dont", "don't",
            "won't", "wont", "isn't", "aint", "ain't", "didn't", "didnt"
        }

        # Booster/dampener words (scale intensity of next sentiment token)
        self._boosters = {
            "very": 1.5, "really": 1.3, "extremely": 1.8, "super": 1.4, "so": 1.2,
            "slightly": 0.8, "somewhat": 0.9, "kinda": 0.9, "sorta": 0.9
        }

    # ---------------------------
    # VADER
    # ---------------------------
    def analyze_vader(self, text: str) -> Dict[str, float | str]:
        """
        VADER rules summarized:
        - Lexicon valence per token (handles emojis, slang like "meh", ":)", ":-(").
        - Punctuation emphasis: repeated '!' boosts intensity; contrastive conjunctions like "but" shift valence.
        - Capitalization: ALLCAPS boosts intensity.
        - Degree modifiers: 'very', 'extremely', 'slightly' alter intensity.
        - Negation: cues invert/attenuate subsequent valence.
        - 'compound' is normalized sum ∈ [-1, 1]; labeling uses configurable thresholds.
        """
        s = self.vader.polarity_scores(text)
        comp = s["compound"]
        neg_t, pos_t = self.thresholds.vader_neg_pos
        if comp >= pos_t:
            label = "positive"
        elif comp <= neg_t:
            label = "negative"
        else:
            label = "neutral"
        return {
            "vader_neg": s["neg"],
            "vader_neu": s["neu"],
            "vader_pos": s["pos"],
            "vader_compound": comp,
            "vader_label": label,
        }

    # ---------------------------
    # TextBlob
    # ---------------------------
    def analyze_textblob(self, text: str) -> Dict[str, float | str]:
        """
        TextBlob sentiment:
        - Polarity ∈ [-1, 1] (negative..positive)
        - Subjectivity ∈ [0, 1] (objective..subjective)
        Labeling rule (tunable neutral band):
            polarity >= +pos_t -> positive
            polarity <= -neg_t -> negative
            else -> neutral
        """
        b = self.textblob(text)
        pol = float(b.sentiment.polarity)
        sub = float(b.sentiment.subjectivity)
        neg_t, pos_t = self.thresholds.textblob_neg_pos
        if pol >= pos_t:
            label = "positive"
        elif pol <= neg_t:
            label = "negative"
        else:
            label = "neutral"
        return {
            "textblob_polarity": pol,
            "textblob_subjectivity": sub,
            "textblob_label": label,
        }

    # ---------------------------
    # AFINN (with explicit token rules)
    # ---------------------------
    def _tokenize(self, text: str) -> List[str]:
        toks = self._token_re.findall(text)
        return [t.lower() for t in toks] if self.lowercase_for_rules else toks

    def _lookup_afinn(self, token: str) -> int:
        # Domain overrides take precedence
        if token in self.afinn_overrides:
            return self.afinn_overrides[token]
        return self.afinn.score(token)  # 0 if unknown

    def _afinn_score_with_rules(self, text: str) -> tuple[float, int]:
        """
        AFINN baseline:
            sum integer valence per token (e.g., good=+3, terrible=-3)

        Our explicit rules:
          1) Negation flips sign of the *next* sentiment token within a small window
             e.g., "not good" -> +3 becomes -3
          2) Booster/dampener scales the *next* sentiment token
             e.g., "very good" -> +3 * 1.5 = +4.5
          3) Domain lexicon overrides (e.g., 'GOAT': +4) override AFINN values
          4) We return (#matched sentiment tokens) for diagnostics

        Returns:
            total_score: float
            matched_tokens: int
        """
        tokens = self._tokenize(text)
        total = 0.0
        matched = 0
        i = 0
        n = len(tokens)

        while i < n:
            tok = tokens[i]
            is_negation = tok in self._negations
            booster = self._boosters.get(tok, 1.0)

            if is_negation or booster != 1.0:
                # Look ahead up to negation_window tokens for the next sentiment-bearing word
                j = i + 1
                applied = False
                while j < n and j <= i + self.negation_window:
                    base = self._lookup_afinn(tokens[j])
                    if base != 0:
                        signed = base * (-1 if is_negation else 1) * booster
                        total += signed
                        matched += 1
                        applied = True
                        break
                    j += 1
                i += 1
                if applied:
                    # Skip the token we just handled
                    i = j + 1
                continue

            # Plain AFINN for this token
            val = self._lookup_afinn(tok)
            if val != 0:
                total += val
                matched += 1
            i += 1

        return total, matched

    def analyze_afinn(self, text: str) -> Dict[str, float | str]:
        """
        Final AFINN labeling:
          - Compute rule-augmented total
          - Normalize by token count to reduce length bias
          - Label using ±threshold
        """
        total, matched = self._afinn_score_with_rules(text)
        tokens = self._tokenize(text)

        # Normalization choice: divide by token count for interpretability
        norm = total / max(1, len(tokens))

        neg_t, pos_t = self.thresholds.afinn_neg_pos
        if norm >= pos_t:
            label = "positive"
        elif norm <= neg_t:
            label = "negative"
        else:
            label = "neutral"

        return {
            "afinn_total": float(total),
            "afinn_matched_tokens": int(matched),
            "afinn_norm_per_token": float(norm),
            "afinn_label": label,
        }

    # ---------------------------
    # Batch helpers with tqdm
    # ---------------------------
    def analyze_text(self, text: str) -> Dict[str, float | str]:
        """Run all three analyzers and merge outputs."""
        out = {}
        out.update(self.analyze_vader(text))
        out.update(self.analyze_textblob(text))
        out.update(self.analyze_afinn(text))
        return out

    def apply_to_dataframe(self, df: pd.DataFrame, text_col: str = "Review") -> pd.DataFrame:
        """
        Adds columns:
          VADER   -> vader_neg, vader_neu, vader_pos, vader_compound, vader_label
          TextBlob-> textblob_polarity, textblob_subjectivity, textblob_label
          AFINN   -> afinn_total, afinn_matched_tokens, afinn_norm_per_token, afinn_label
        Shows a tqdm progress bar while processing.
        """
        if text_col not in df.columns:
            raise KeyError(f"Column '{text_col}' not found in DataFrame.")

        result = df.copy()
        result[text_col] = result[text_col].astype(str)

        vader_cols = ["vader_neg", "vader_neu", "vader_pos", "vader_compound", "vader_label"]
        tb_cols    = ["textblob_polarity", "textblob_subjectivity", "textblob_label"]
        af_cols    = ["afinn_total", "afinn_matched_tokens", "afinn_norm_per_token", "afinn_label"]

        # Hook tqdm into pandas .apply (progress bar for each review)
        tqdm.pandas(desc="Analyzing Reviews")
        records = result[text_col].progress_apply(self.analyze_text)

        # Expand list of dicts into columns
        result[vader_cols + tb_cols + af_cols] = pd.DataFrame(list(records.values))

        return result

    @staticmethod
    def explain_rules() -> str:
        return (
            "VADER rules:\n"
            "- Sentiment lexicon with intensities; handles emojis, slang, ALLCAPS, repeated punctuation.\n"
            "- Degree modifiers (very/extremely/slightly) adjust intensity.\n"
            "- Negation cues (not, n't, never) invert/attenuate valence.\n"
            "- 'compound' ∈ [-1,1]; defaults: ≤-0.05 negative, ≥0.05 positive, else neutral.\n\n"
            "TextBlob rules:\n"
            "- Polarity ∈ [-1,1]; subjectivity ∈ [0,1].\n"
            "- Labels via thresholds (default ±0.10 neutral band).\n\n"
            "AFINN rules (in this implementation):\n"
            "- Base: sum integer valence per token (good=+3, terrible=-3, etc.).\n"
            "- Added: negation window flips sign of next sentiment token.\n"
            "- Added: booster words scale next sentiment token.\n"
            "- Added: domain-specific lexicon overrides.\n"
            "- Normalize by token count; label via ±0.10 neutral band (tunable).\n"
        )


# ------------------------------------------------------------------------------
# Simple, notebook-friendly runner (no CLI)
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Set your paths/column here
    input_path = "IMDB Dataset.csv"   # <- change to your CSV
    output_path = "reviews_with_sentiment.csv"
    text_col = "review"               # <- change to your text column name

    # Read CSV (you'll see the progress during analysis step below)
    df = pd.read_csv(input_path)

    th = Thresholds(
        vader_neg_pos=(-0.05, 0.05),
        textblob_neg_pos=(-0.10, 0.10),
        afinn_neg_pos=(-0.10, 0.10),
    )

    analyzer = RuleBasedSentiment(thresholds=th)

    print("\n=== RULE SUMMARY ===")
    print(analyzer.explain_rules())

    # Progress bar appears here while each review is analyzed
    out = analyzer.apply_to_dataframe(df, text_col=text_col)

    out.to_csv(output_path, index=False)
    print(f"\n[OK] Saved: {output_path}")
    print(out.head())



# import re
# import math
# import pandas as pd
# from dataclasses import dataclass
# from typing import Dict, Tuple, Optional, List
#
# # --- VADER (lexicon + heuristics for punctuation, caps, emojis, etc.)
# import nltk
# from nltk.sentiment import SentimentIntensityAnalyzer
#
# # --- TextBlob (pattern-based polarity/subjectivity)
# from textblob import TextBlob
#
# # --- AFINN (simple valence-word sum)
# from afinn import Afinn
#
# from tqdm import tqdm
# tqdm.pandas(desc="Analyzing Reviews")
#
#
# @dataclass
# class Thresholds:
#     """
#     Neutral 'gray zones' around 0 prevent overconfident labels for weak signals.
#     You can tighten/loosen these depending on your data.
#     """
#     # VADER uses 'compound' in [-1, 1]. Default (per paper/code): [-0.05, 0.05].
#     vader_neg_pos: Tuple[float, float] = (-0.05, 0.05)
#
#     # TextBlob polarity is also [-1, 1]. A slightly wider neutral band is common.
#     textblob_neg_pos: Tuple[float, float] = (-0.10, 0.10)
#
#     # AFINN raw sum grows with text length; we normalize by token count.
#     # After normalization to [-5, +5] typical scale, we use ±0.10 as neutral band.
#     afinn_neg_pos: Tuple[float, float] = (-0.10, 0.10)
#
#
# class RuleBasedSentiment:
#     """
#     Rule-based sentiment with three methods:
#     1) VADER  : Valence-aware dictionary with heuristics for ALLCAPS, punctuation, emojis, boosters ('very'), and negation.
#     2) TextBlob: Pattern-based polarity/subjectivity.
#     3) AFINN  : Sum of per-word valence; here we add explicit negation handling + score normalization.
#     """
#
#     def __init__(
#         self,
#         thresholds: Thresholds = Thresholds(),
#         lexicon_overrides: Optional[Dict[str, int]] = None,
#         language: str = "en",
#         afinn_emoticons: bool = True,
#         lowercase_for_rules: bool = True,
#     ):
#         """
#         Parameters
#         ----------
#         thresholds : Thresholds
#             Negative/neutral/positive cutoffs for each method.
#         lexicon_overrides : dict[str, int], optional
#             Custom AFINN/VADER overrides, e.g., {'fire': -3, 'awesome++': 5, 'GOAT': 4}.
#             VADER supports lexicon updates; AFINN we handle manually before scoring.
#         language : str
#             Language for AFINN (default 'en').
#         afinn_emoticons : bool
#             Whether AFINN should recognize emoticons like ':-)'.
#         lowercase_for_rules : bool
#             If True, tokenize and match AFINN on lowercased text (recommended).
#         """
#         self.thresholds = thresholds
#         self.lowercase_for_rules = lowercase_for_rules
#
#         # --- Initialize analyzers
#         self.vader = SentimentIntensityAnalyzer()
#         if lexicon_overrides:
#             # VADER allows lexicon updates directly (affects polarity_scores):
#             self.vader.lexicon.update(
#                 {k.lower(): float(v) for k, v in lexicon_overrides.items()}
#             )
#
#         self.textblob = TextBlob  # class reference for clarity
#
#         # For AFINN we instantiate and also keep an override dict for our rule handler
#         self.afinn = Afinn(language=language, emoticons=afinn_emoticons)
#         self.afinn_overrides = {k.lower(): int(v) for k, v in (lexicon_overrides or {}).items()}
#
#         # Simple regex tokenizer for AFINN rules
#         self._token_re = re.compile(r"[A-Za-z][A-Za-z'\-]+")
#
#         # Negation cue list (extend as needed)
#         self._negations = {
#             "not", "no", "never", "n't", "cannot", "can't", "dont", "don't",
#             "won't", "wont", "isn't", "aint", "ain't", "didn't", "didnt"
#         }
#
#         # Booster/dampener words for AFINN (optional, very light heuristic)
#         self._boosters = {
#             "very": 1.5, "really": 1.3, "extremely": 1.8, "super": 1.4, "so": 1.2,
#             "slightly": 0.8, "somewhat": 0.9, "kinda": 0.9, "sorta": 0.9
#         }
#
#     # ---------------------------
#     # VADER
#     # ---------------------------
#     def analyze_vader(self, text: str) -> Dict[str, float | str]:
#         """
#         Rules implemented by VADER (summarized):
#         - Lexicon valence per token (includes slang/emojis like :) :D :/ ).
#         - Punctuation emphasis: '!!!' increases intensity; 'but' shifts contrastive sentiment.
#         - Capitalization: ALLCAPS boosts intensity.
#         - Degree modifiers (booster/dampeners): 'very', 'extremely', 'slightly'.
#         - Negation handling: flips or lessens valence after cues ('not', "n't", etc.).
#         - 'Compound' = normalized sum in [-1, 1]; we label via thresholds.
#         """
#         scores = self.vader.polarity_scores(text)
#         comp = scores["compound"]
#         neg_t, pos_t = self.thresholds.vader_neg_pos
#         if comp >= pos_t:
#             label = "positive"
#         elif comp <= neg_t:
#             label = "negative"
#         else:
#             label = "neutral"
#         # Return full VADER breakdown plus label
#         return {
#             "vader_neg": scores["neg"],
#             "vader_neu": scores["neu"],
#             "vader_pos": scores["pos"],
#             "vader_compound": comp,
#             "vader_label": label,
#         }
#
#     # ---------------------------
#     # TextBlob
#     # ---------------------------
#     def analyze_textblob(self, text: str) -> Dict[str, float | str]:
#         """
#         Rules implemented by TextBlob sentiment:
#         - Polarity ∈ [-1, 1] (negative to positive).
#         - Subjectivity ∈ [0, 1] (objective to subjective).
#         - Based on pattern's lexicon & rules (adjectives/adv, degree words).
#         Labeling rule here:
#         - If polarity >= +pos_threshold -> positive
#         - If polarity <= -neg_threshold -> negative
#         - Else -> neutral
#         """
#         blob = self.textblob(text)
#         polarity = float(blob.sentiment.polarity)
#         subjectivity = float(blob.sentiment.subjectivity)
#         neg_t, pos_t = self.thresholds.textblob_neg_pos
#         if polarity >= pos_t:
#             label = "positive"
#         elif polarity <= neg_t:
#             label = "negative"
#         else:
#             label = "neutral"
#         return {
#             "textblob_polarity": polarity,
#             "textblob_subjectivity": subjectivity,
#             "textblob_label": label,
#         }
#
#     # ---------------------------
#     # AFINN (with explicit token rules)
#     # ---------------------------
#     def _tokenize(self, text: str) -> List[str]:
#         tokens = self._token_re.findall(text)
#         return [t.lower() for t in tokens] if self.lowercase_for_rules else tokens
#
#     def _afinn_score_with_rules(self, text: str) -> Tuple[float, int]:
#         """
#         AFINN baseline:
#           - Each token has an integer valence (e.g., 'good' = +3, 'terrible' = -3).
#           - Final score = sum of token valences.
#         We enhance with minimal rules:
#           1) Negation flips sign of the next sentiment-bearing token within a small window.
#           2) Booster words scale the next sentiment-bearing token (e.g., 'very good' -> +3 * 1.5).
#           3) Lexicon overrides (domain tuning) take precedence over AFINN built-ins.
#
#         Returns
#         -------
#         total_score : float (can be any real number)
#         matched : int (# of tokens that had non-zero valence before/after rules)
#         """
#         tokens = self._tokenize(text)
#         total = 0.0
#         matched = 0
#
#         i = 0
#         while i < len(tokens):
#             tok = tokens[i]
#
#             # Rule: Detect negation cue that affects the NEXT sentiment token
#             is_negation = tok in self._negations
#
#             # Rule: Booster/dampener to scale the NEXT sentiment token
#             booster = self._boosters.get(tok, 1.0)
#
#             if is_negation or booster != 1.0:
#                 # Look ahead 1–3 tokens for the next sentiment-bearing word
#                 lookahead = 3
#                 j = i + 1
#                 applied = False
#                 while j < len(tokens) and j <= i + lookahead:
#                     base = self._lookup_afinn(tokens[j])
#                     if base != 0:
#                         # Apply negation flip + booster scaling
#                         signed = base * (-1 if is_negation else 1) * booster
#                         total += signed
#                         matched += 1
#                         applied = True
#                         break
#                     j += 1
#                 # If nothing found, just continue
#                 i += 1
#                 if applied:
#                     # Skip to the word we just handled to avoid double-counting
#                     i = j + 1
#                 continue
#
#             # Regular AFINN scoring for this token
#             val = self._lookup_afinn(tok)
#             if val != 0:
#                 total += val
#                 matched += 1
#
#             i += 1
#
#         return total, matched
#
#     def _lookup_afinn(self, token: str) -> int:
#         # Overrides first (domain tuning)
#         if token in self.afinn_overrides:
#             return self.afinn_overrides[token]
#         # Fall back to Afinn internal dict
#         return self.afinn.score(token)
#
#     def analyze_afinn(self, text: str) -> Dict[str, float | str]:
#         """
#         Final labeling rule for AFINN:
#           1) Compute rule-augmented sum (negation/booster/overrides).
#           2) Normalize by token count (or sqrt length) to reduce length bias.
#           3) If normalized >= +pos_threshold -> positive
#              If normalized <= -neg_threshold -> negative
#              Else -> neutral
#         """
#         total, matched = self._afinn_score_with_rules(text)
#
#         tokens = self._tokenize(text)
#         # Normalization choice:
#         #   - Divide by max(1, #tokens) keeps it small for long texts.
#         #   - sqrt(len) is another option; we’ll use token count for interpretability.
#         norm = total / max(1, len(tokens))
#
#         neg_t, pos_t = self.thresholds.afinn_neg_pos
#         if norm >= pos_t:
#             label = "positive"
#         elif norm <= neg_t:
#             label = "negative"
#         else:
#             label = "neutral"
#
#         return {
#             "afinn_total": float(total),
#             "afinn_matched_tokens": int(matched),
#             "afinn_norm_per_token": float(norm),
#             "afinn_label": label,
#         }
#
#     # ---------------------------
#     # Batch helpers
#     # ---------------------------
#     def analyze_text(self, text: str) -> Dict[str, float | str]:
#         """
#         Run all three analyzers and merge into one dict.
#         """
#         out = {}
#         out.update(self.analyze_vader(text))
#         out.update(self.analyze_textblob(text))
#         out.update(self.analyze_afinn(text))
#         return out
#
#     def apply_to_dataframe(self, df: pd.DataFrame, text_col: str = "Review") -> pd.DataFrame:
#         """
#         Adds columns:
#           - VADER: vader_neg, vader_neu, vader_pos, vader_compound, vader_label
#           - TextBlob: textblob_polarity, textblob_subjectivity, textblob_label
#           - AFINN: afinn_total, afinn_matched_tokens, afinn_norm_per_token, afinn_label
#         """
#         if text_col not in df.columns:
#             raise KeyError(f"Column '{text_col}' not found in DataFrame.")
#
#         result = df.copy()
#         result[text_col] = result[text_col].astype(str)
#
#         vader_cols = ["vader_neg","vader_neu","vader_pos","vader_compound","vader_label"]
#         tb_cols    = ["textblob_polarity","textblob_subjectivity","textblob_label"]
#         af_cols    = ["afinn_total","afinn_matched_tokens","afinn_norm_per_token","afinn_label"]
#
#         # Apply once per row (vectorized apply)
#         records = result[text_col].apply(self.analyze_text)
#
#         # Expand dicts to columns
#         result[vader_cols + tb_cols + af_cols] = pd.DataFrame(list(records.values))
#
#         return result
#
#     @staticmethod
#     def explain_rules() -> str:
#         return (
#         "VADER rules:\n"
#         "- Uses a sentiment lexicon with intensities; handles emojis, slang, ALLCAPS, repeated punctuation.\n"
#         "- Degree modifiers increase/decrease intensity (e.g., 'very good' > 'good').\n"
#         "- Negation cues (not, n't, never) invert or attenuate subsequent valence.\n"
#         "- 'compound' is a normalized sum in [-1,1]; default labels: <=-0.05 negative, >=0.05 positive, else neutral.\n\n"
#         "TextBlob rules:\n"
#         "- Polarity in [-1,1] from pattern-based lexicon/grammar; subjectivity in [0,1].\n"
#         "- Labels via thresholds (default ±0.10) to keep weak signals neutral.\n\n"
#         "AFINN rules (this implementation):\n"
#         "- Base: sum integer valence per token (e.g., good=+3, terrible=-3).\n"
#         "- Added: simple negation window (flip next sentiment token within 3 words).\n"
#         "- Added: booster scaling for next sentiment token (very/really/extremely, etc.).\n"
#         "- Added: domain-specific lexicon overrides.\n"
#         "- Normalize by token count to reduce length bias; label via thresholds (default ±0.10).\n"
#         )
#
#
# # ---------------------------
# # Example usage
# # ---------------------------
# if __name__ == "__main__":
#     # 1) Load your CSV with a 'Review' column
#     #    Replace 'reviews.csv' with your path (no labels required for this step)
#     csv_path = "IMDB Dataset.csv"
#     df = pd.read_csv(csv_path)  # must contain a 'Review' column
#
#     # 2) Optional: domain-specific overrides (tune lexicon for your context)
#     #    Example: interpret 'GOAT' as strongly positive, 'buggy' as negative
#     overrides = {
#         "GOAT": 4,
#         "buggy": -3,
#         "laggy": -2,
#     }
#
#     model = RuleBasedSentiment(
#         thresholds=Thresholds(
#             vader_neg_pos=(-0.05, 0.05),
#             textblob_neg_pos=(-0.10, 0.10),
#             afinn_neg_pos=(-0.10, 0.10),
#         ),
#         lexicon_overrides=overrides,
#     )
#
#     print("\n=== RULE SUMMARY ===")
#     print(model.explain_rules())
#
#     # 3) Run analysis
#     out = model.apply_to_dataframe(df, text_col="review")
#
#     # 4) Save results
#     out.to_csv("reviews_with_rule_based_sentiment.csv", index=False)
#     print("\nSaved: reviews_with_rule_based_sentiment.csv")
#
#     # 5) Peek at first few rows
#     print(out.head(5))
