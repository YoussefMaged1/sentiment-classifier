"""
NLP Preprocessing Pipeline
Supports: Amazon Fine Food Reviews & Sentiment140
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Optional

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

logger = logging.getLogger(__name__)

@dataclass
class PreprocessorConfig:
    """
    Configuration for TextPreprocessor.
    Tweak these flags to adapt to Amazon reviews vs tweets.
    """

    # --- Cleaning ---
    remove_html: bool = True          # critical for Amazon (has <br/> tags)
    remove_urls: bool = True          # critical for Sentiment140 (lots of URLs)
    remove_mentions: bool = False     # True for tweets (@user)
    remove_hashtags: bool = False     # True for tweets (#topic) — keeps the word by default
    remove_numbers: bool = False      # keep numbers (e.g. "5 stars")
    remove_punctuation: bool = True
    lowercase: bool = True
    expand_contractions: bool = True  # don't -> do not, it's -> it is

    # --- Tokenization ---
    tokenizer: str = "nltk"           # "nltk" | "whitespace"

    # --- Normalization ---
    remove_stopwords: bool = True
    stemming: bool = False            # Porter stemmer
    lemmatization: bool = True        # WordNet lemmatizer (better quality than stemming)

    # --- Filtering ---
    min_token_length: int = 2
    max_token_length: int = 50

    # --- Domain hint (for logging/tracking only) ---
    domain: str = "generic"           # "amazon" | "twitter" | "generic"

    # Common English contractions
    contractions: dict = field(default_factory=lambda: {
        "won't": "will not", "can't": "cannot", "n't": " not",
        "'re": " are", "'s": " is", "'d": " would", "'ll": " will",
        "'ve": " have", "'m": " am", "it's": "it is", "i'm": "i am",
        "don't": "do not", "doesn't": "does not", "didn't": "did not",
        "isn't": "is not", "aren't": "are not", "wasn't": "was not",
        "weren't": "were not", "haven't": "have not", "hasn't": "has not",
        "hadn't": "had not", "wouldn't": "would not", "couldn't": "could not",
        "shouldn't": "should not", "mustn't": "must not",
    })


class TextPreprocessor:
    """
    Configurable NLP preprocessing pipeline.

    Works on both Amazon Fine Food Reviews and Sentiment140 tweets
    without rewriting code — just change the config.

    Example
    -------
    >>> # Amazon config (HTML removal, lemmatization, no mention removal)
    >>> amazon_cfg = PreprocessorConfig(domain="amazon", remove_html=True)
    >>> p = TextPreprocessor(amazon_cfg)
    >>> p.process("This product is <b>amazing</b>! Won't buy anything else.")
    'product amazing will buy anything else'

    >>> # Twitter config (mentions, hashtags, URLs)
    >>> twitter_cfg = PreprocessorConfig(
    ...     domain="twitter", remove_mentions=True,
    ...     remove_hashtags=False, remove_html=False
    ... )
    >>> p2 = TextPreprocessor(twitter_cfg)
    >>> p2.process("@user Check this out! https://t.co/abc #NLP is great")
    'check nlp great'
    """

    def __init__(self, config: Optional[PreprocessorConfig] = None):
        self.config = config or PreprocessorConfig()
        self._stemmer = PorterStemmer() if self.config.stemming else None
        self._lemmatizer = WordNetLemmatizer() if self.config.lemmatization else None
        self._stopwords = (
            set(stopwords.words("english")) if self.config.remove_stopwords else set()
        )
        logger.info(
            "TextPreprocessor initialized | domain=%s | stemming=%s | lemmatization=%s",
            self.config.domain,
            self.config.stemming,
            self.config.lemmatization,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, text: str) -> str:
        """Run full pipeline on a single string. Returns cleaned string."""
        if not isinstance(text, str) or not text.strip():
            return ""

        text = self._clean(text)
        tokens = self._tokenize(text)
        tokens = self._normalize(tokens)
        return " ".join(tokens)

    def process_batch(self, texts: list[str], show_progress: bool = False) -> list[str]:
        """Run pipeline on a list of texts."""
        if show_progress:
            try:
                from tqdm import tqdm
                return [self.process(t) for t in tqdm(texts, desc="Preprocessing")]
            except ImportError:
                pass
        return [self.process(t) for t in texts]

    # ------------------------------------------------------------------
    # Step 1 — Cleaning
    # ------------------------------------------------------------------

    def _clean(self, text: str) -> str:
        cfg = self.config

        if cfg.lowercase:
            text = text.lower()

        if cfg.expand_contractions:
            text = self._expand_contractions(text)

        if cfg.remove_html:
            text = re.sub(r"<[^>]+>", " ", text)          # strip HTML tags
            text = re.sub(r"&[a-z]+;", " ", text)          # strip HTML entities

        if cfg.remove_urls:
            text = re.sub(r"https?://\S+|www\.\S+", " ", text)

        if cfg.remove_mentions:
            text = re.sub(r"@\w+", " ", text)

        if cfg.remove_hashtags:
            text = re.sub(r"#(\w+)", " ", text)            # remove # but keep word
        else:
            text = re.sub(r"#", " ", text)                  # strip # symbol only

        if cfg.remove_numbers:
            text = re.sub(r"\b\d+\b", " ", text)

        if cfg.remove_punctuation:
            text = re.sub(r"[^\w\s]", " ", text)

        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _expand_contractions(self, text: str) -> str:
        for contraction, expansion in self.config.contractions.items():
            text = text.replace(contraction, expansion)
        return text

    # ------------------------------------------------------------------
    # Step 2 — Tokenization
    # ------------------------------------------------------------------

    def _tokenize(self, text: str) -> list[str]:
        if self.config.tokenizer == "nltk":
            return word_tokenize(text)
        return text.split()  # whitespace fallback

    # ------------------------------------------------------------------
    # Step 3 — Normalization
    # ------------------------------------------------------------------

    def _normalize(self, tokens: list[str]) -> list[str]:
        cfg = self.config
        result = []

        for token in tokens:
            # Length filter
            if not (cfg.min_token_length <= len(token) <= cfg.max_token_length):
                continue

            # Stopword removal
            if cfg.remove_stopwords and token in self._stopwords:
                continue

            # Stemming (mutually exclusive with lemmatization — lemma wins if both on)
            if cfg.lemmatization and self._lemmatizer:
                token = self._lemmatizer.lemmatize(token)
            elif cfg.stemming and self._stemmer:
                token = self._stemmer.stem(token)

            result.append(token)

        return result

    # ------------------------------------------------------------------
    # Helpers / Inspection
    # ------------------------------------------------------------------

    def explain(self, text: str) -> dict:
        """
        Return a step-by-step breakdown — useful for notebooks / debugging.

        >>> p.explain("This product is <b>great</b>! Won't buy anything else.")
        {
            'original': '...',
            'after_cleaning': '...',
            'tokens_raw': [...],
            'tokens_normalized': [...],
            'final': '...'
        }
        """
        original = text
        cleaned = self._clean(text)
        raw_tokens = self._tokenize(cleaned)
        normalized = self._normalize(raw_tokens)
        return {
            "original": original,
            "after_cleaning": cleaned,
            "tokens_raw": raw_tokens,
            "tokens_normalized": normalized,
            "final": " ".join(normalized),
        }

    @staticmethod
    def amazon_config() -> PreprocessorConfig:
        """Recommended config for Amazon Fine Food Reviews."""
        return PreprocessorConfig(
            domain="amazon",
            remove_html=True,
            remove_urls=True,
            remove_mentions=False,
            remove_hashtags=False,
            remove_numbers=False,
            expand_contractions=True,
            lemmatization=True,
            stemming=False,
        )

    @staticmethod
    def twitter_config() -> PreprocessorConfig:
        """Recommended config for Sentiment140 tweets."""
        return PreprocessorConfig(
            domain="twitter",
            remove_html=False,
            remove_urls=True,
            remove_mentions=True,
            remove_hashtags=False,   # keep hashtag words
            remove_numbers=True,
            expand_contractions=True,
            lemmatization=True,
            stemming=False,
        )
