"""
Text Vectorization Methods
Implements: BOW (Bag of Words), TF-IDF, BM25
"""

import numpy as np
import logging
from typing import List, Union, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod

import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


@dataclass
class VectorizerConfig:
    """Configuration for all vectorizers"""
    max_features: int = 1_000_000
    min_df: int = 2           # Ignore terms appearing in fewer than min_df documents
    max_df: float = 0.95      # Ignore terms appearing in more than max_df fraction of documents
    ngram_range: Tuple[int, int] = (1, 1)  # (1, 1) for unigrams, (1, 2) for unigrams + bigrams
    lowercase: bool = True
    strip_accents: str = "unicode"  # "ascii", "unicode", None


class BaseVectorizer(ABC):
    """Abstract base class for all vectorizers"""

    def __init__(self, config: VectorizerConfig = None):
        self.config = config or VectorizerConfig()
        self.vocabulary_ = None
        self.fitted = False

    @abstractmethod
    def fit(self, texts: List[str]) -> 'BaseVectorizer':
        """Fit the vectorizer on texts"""
        pass

    @abstractmethod
    def transform(self, texts: List[str]) -> Union[np.ndarray, sp.csr_matrix]:
        """Transform texts to vectors"""
        pass

    def fit_transform(self, texts: List[str]) -> Union[np.ndarray, sp.csr_matrix]:
        """Fit and transform in one step"""
        return self.fit(texts).transform(texts)

    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """Get the feature names (vocabulary)"""
        pass


class BagOfWordsVectorizer(BaseVectorizer):
    """
    Bag of Words Vectorizer
    Simple word frequency counts without any normalization or weighting.
    
    Example
    -------
    >>> bow = BagOfWordsVectorizer()
    >>> texts = ["hello world", "hello python world"]
    >>> bow.fit_transform(texts)
    """

    def __init__(self, config: VectorizerConfig = None):
        super().__init__(config)
        self.vectorizer = CountVectorizer(
            max_features=self.config.max_features,
            min_df=self.config.min_df,
            max_df=self.config.max_df,
            ngram_range=self.config.ngram_range,
            lowercase=self.config.lowercase,
            strip_accents=self.config.strip_accents,
        )

    def fit(self, texts: List[str]) -> 'BagOfWordsVectorizer':
        """Fit the BOW vectorizer"""
        self.vectorizer.fit(texts)
        self.vocabulary_ = self.vectorizer.get_feature_names_out()
        self.fitted = True
        logger.info(f"BagOfWordsVectorizer fitted | vocabulary size: {len(self.vocabulary_)}")
        return self

    def transform(self, texts: List[str]) -> sp.csr_matrix:
        """Transform texts to BOW sparse matrix"""
        if not self.fitted:
            raise ValueError("Vectorizer must be fitted before transform. Call fit() first.")
        return self.vectorizer.transform(texts)

    def get_feature_names(self) -> List[str]:
        """Get vocabulary"""
        if self.vocabulary_ is None:
            raise ValueError("Vectorizer not fitted yet")
        return list(self.vocabulary_)

    def get_dense_vectors(self, texts: List[str]) -> np.ndarray:
        """Transform to dense numpy array"""
        return self.transform(texts).toarray()


class TfidfVectorizer_(BaseVectorizer):
    """
    TF-IDF (Term Frequency - Inverse Document Frequency) Vectorizer
    
    TF-IDF = Term Frequency × Inverse Document Frequency
    - TF: How often a term appears in a document
    - IDF: How rare/unique a term is across all documents
    
    Example
    -------
    >>> tfidf = TfidfVectorizer_()
    >>> texts = ["hello world", "hello python world"]
    >>> vectors = tfidf.fit_transform(texts)
    """

    def __init__(self, config: VectorizerConfig = None):
        super().__init__(config)
        self.vectorizer = TfidfVectorizer(
            max_features=self.config.max_features,
            min_df=self.config.min_df,
            max_df=self.config.max_df,
            ngram_range=self.config.ngram_range,
            lowercase=self.config.lowercase,
            strip_accents=self.config.strip_accents,
            norm='l2',  # L2 normalization (default)
            sublinear_tf=False,  # Use sublinear_tf=True for large datasets
        )

    def fit(self, texts: List[str]) -> 'TfidfVectorizer_':
        """Fit the TF-IDF vectorizer"""
        self.vectorizer.fit(texts)
        self.vocabulary_ = self.vectorizer.get_feature_names_out()
        self.fitted = True
        logger.info(f"TfidfVectorizer fitted | vocabulary size: {len(self.vocabulary_)}")
        return self

    def transform(self, texts: List[str]) -> sp.csr_matrix:
        """Transform texts to TF-IDF sparse matrix"""
        if not self.fitted:
            raise ValueError("Vectorizer must be fitted before transform. Call fit() first.")
        return self.vectorizer.transform(texts)

    def get_feature_names(self) -> List[str]:
        """Get vocabulary"""
        if self.vocabulary_ is None:
            raise ValueError("Vectorizer not fitted yet")
        return list(self.vocabulary_)

    def get_dense_vectors(self, texts: List[str]) -> np.ndarray:
        """Transform to dense numpy array"""
        return self.transform(texts).toarray()

    def get_idf_scores(self) -> dict:
        """Get IDF (Inverse Document Frequency) scores for each term"""
        if not self.fitted:
            raise ValueError("Vectorizer not fitted yet")
        idf_dict = {}
        for idx, term in enumerate(self.vocabulary_):
            idf_dict[term] = self.vectorizer.idf_[idx]
        return idf_dict


class BM25Vectorizer(BaseVectorizer):
    """
    BM25 (Best Matching 25) Vectorizer
    A probabilistic information retrieval model widely used in search engines.
    
    BM25 = IDF × (f(qi, D) × (k1 + 1)) / (f(qi, D) + k1 × (1 - b + b × (|D| / avgdl)))
    
    Parameters:
    - k1: Controls non-linear term frequency saturation (default: 1.5)
    - b: Controls how much effect document length has on relevance (default: 0.75)
    
    Example
    -------
    >>> bm25 = BM25Vectorizer()
    >>> texts = ["hello world", "hello python world"]
    >>> bm25.fit(texts)
    >>> scores = bm25.rank("hello python")  # Rank documents by query
    """

    def __init__(self, config: VectorizerConfig = None, k1: float = 1.5, b: float = 0.75):
        super().__init__(config)
        self.k1 = k1
        self.b = b
        self.bm25 = None
        self.tokenized_corpus = None
        self.vocabulary_ = None

    def fit(self, texts: List[str]) -> 'BM25Vectorizer':
        """Fit the BM25 vectorizer"""
        # Tokenize all texts (assuming texts are already preprocessed)
        self.tokenized_corpus = [text.split() for text in texts]
        
        # Build vocabulary from corpus
        vocab_set = set()
        for tokens in self.tokenized_corpus:
            vocab_set.update(tokens)
        self.vocabulary_ = sorted(list(vocab_set))
        
        # Fit BM25 model
        self.bm25 = BM25Okapi(self.tokenized_corpus, k1=self.k1, b=self.b)
        self.fitted = True
        logger.info(f"BM25Vectorizer fitted | vocabulary size: {len(self.vocabulary_)}")
        return self

    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transform texts to BM25 scores.
        Returns a matrix where each row is BM25 scores against the corpus.
        """
        if not self.fitted:
            raise ValueError("Vectorizer must be fitted before transform. Call fit() first.")
        
        scores = []
        for text in texts:
            tokens = text.split()
            doc_scores = self.bm25.get_scores(tokens)
            scores.append(doc_scores)
        
        return np.array(scores)

    def rank(self, query: str, top_k: int = None) -> List[Tuple[int, float]]:
        """
        Rank documents by BM25 score for a given query.
        Returns list of (doc_index, score) sorted by score descending.
        """
        if not self.fitted:
            raise ValueError("Vectorizer must be fitted before ranking. Call fit() first.")
        
        tokens = query.split()
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        
        if top_k:
            ranked = ranked[:top_k]
        
        return ranked

    def get_feature_names(self) -> List[str]:
        """Get vocabulary"""
        if self.vocabulary_ is None:
            raise ValueError("Vectorizer not fitted yet")
        return self.vocabulary_

    def get_dense_vectors(self, texts: List[str]) -> np.ndarray:
        """Transform to dense numpy array"""
        return self.transform(texts)


class VectorizerComparison:
    """
    Compare different vectorization methods side by side.
    """

    def __init__(self, config: VectorizerConfig = None):
        self.config = config or VectorizerConfig()
        self.bow = BagOfWordsVectorizer(config)
        self.tfidf = TfidfVectorizer_(config)
        self.bm25 = BM25Vectorizer(config)
        self.texts = None

    def fit_all(self, texts: List[str]) -> 'VectorizerComparison':
        """Fit all three vectorizers"""
        self.texts = texts
        self.bow.fit(texts)
        self.tfidf.fit(texts)
        self.bm25.fit(texts)
        logger.info("All vectorizers fitted successfully")
        return self

    def get_vocab_sizes(self) -> dict:
        """Get vocabulary size for each method"""
        return {
            'BOW': len(self.bow.get_feature_names()),
            'TF-IDF': len(self.tfidf.get_feature_names()),
            'BM25': len(self.bm25.get_feature_names()),
        }

    def compare_first_document(self) -> dict:
        """Compare vectorization of the first document"""
        if not self.texts:
            raise ValueError("No texts fitted yet")
        
        first_text = [self.texts[0]]
        
        bow_vec = self.bow.get_dense_vectors(first_text)[0]
        tfidf_vec = self.tfidf.get_dense_vectors(first_text)[0]
        bm25_vec = self.bm25.transform(first_text)[0]
        
        # Get non-zero indices
        bow_nonzero = np.nonzero(bow_vec)[0]
        tfidf_nonzero = np.nonzero(tfidf_vec)[0]
        bm25_nonzero = np.nonzero(bm25_vec)[0]
        
        return {
            'BOW': {
                'non_zero_count': len(bow_nonzero),
                'top_words': [(self.bow.vocabulary_[i], bow_vec[i]) 
                             for i in np.argsort(bow_vec)[-5:][::-1]],
            },
            'TF-IDF': {
                'non_zero_count': len(tfidf_nonzero),
                'top_words': [(self.tfidf.vocabulary_[i], tfidf_vec[i]) 
                             for i in np.argsort(tfidf_vec)[-5:][::-1]],
            },
            'BM25': {
                'non_zero_count': len(bm25_nonzero),
                'top_words': [(self.bm25.vocabulary_[i], bm25_vec[i]) 
                             for i in np.argsort(bm25_vec)[-5:][::-1]],
            },
        }
