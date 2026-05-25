"""
Sentiment Classification Module
Implements multiple classifiers for sentiment analysis
"""

import numpy as np
import logging
from typing import Dict, Tuple, Any
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)

logger = logging.getLogger(__name__)


@dataclass
class ClassifierMetrics:
    """Container for classifier metrics"""
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc_score: float = None
    confusion_matrix: np.ndarray = None
    classification_report: str = None


class SentimentClassifier:
    """Base sentiment classifier wrapper"""
    
    def __init__(self, model_name: str, model, random_state: int = 42):
        """
        Initialize classifier
        
        Parameters:
        -----------
        model_name : str
            Name of the model (e.g., 'logistic_regression')
        model : sklearn model
            Fitted sklearn model
        random_state : int
            Random state for reproducibility
        """
        self.model_name = model_name
        self.model = model
        self.random_state = random_state
        self.is_fitted = False
        
    def fit(self, X, y):
        """Fit the classifier"""
        self.model.fit(X, y)
        self.is_fitted = True
        logger.info(f"{self.model_name} fitted successfully")
        return self
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted first")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted first")
        
        # LinearSVC doesn't have predict_proba, use decision_function
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        elif hasattr(self.model, 'decision_function'):
            # Convert decision function to probabilities using sigmoid
            decision = self.model.decision_function(X)
            proba = 1 / (1 + np.exp(-decision))
            return np.column_stack([1 - proba, proba])
        else:
            raise ValueError(f"{self.model_name} does not support probability predictions")
    
    def evaluate(self, X, y, y_pred=None) -> ClassifierMetrics:
        """
        Evaluate classifier on test set
        
        Parameters:
        -----------
        X : array-like
            Features
        y : array-like
            True labels
        y_pred : array-like, optional
            Pre-computed predictions
            
        Returns:
        --------
        ClassifierMetrics
            Evaluation metrics
        """
        if y_pred is None:
            y_pred = self.predict(X)
        
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, average='weighted')
        recall = recall_score(y, y_pred, average='weighted')
        f1 = f1_score(y, y_pred, average='weighted')
        
        # Try to get AUC score (binary classification)
        auc = None
        try:
            y_proba = self.predict_proba(X)
            auc = roc_auc_score(y, y_proba[:, 1])
        except Exception as e:
            logger.warning(f"Could not compute AUC: {e}")
        
        conf_matrix = confusion_matrix(y, y_pred)
        class_report = classification_report(y, y_pred)
        
        metrics = ClassifierMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            auc_score=auc,
            confusion_matrix=conf_matrix,
            classification_report=class_report
        )
        
        return metrics
    
    def get_feature_importance(self):
        """Get feature importance if available"""
        if hasattr(self.model, 'coef_'):
            return self.model.coef_[0] if len(self.model.coef_.shape) == 2 else self.model.coef_
        elif hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        else:
            return None


class LogisticRegressionClassifier(SentimentClassifier):
    """Logistic Regression classifier for sentiment analysis"""
    
    def __init__(self, C: float = 1.0, max_iter: int = 1000, random_state: int = 42):
        """
        Initialize Logistic Regression classifier
        
        Parameters:
        -----------
        C : float
            Inverse of regularization strength
        max_iter : int
            Maximum iterations
        random_state : int
            Random state for reproducibility
        """
        model = LogisticRegression(C=C, max_iter=max_iter, random_state=random_state)
        super().__init__('LogisticRegression', model, random_state)
        self.C = C


class SVMClassifier(SentimentClassifier):
    """Support Vector Machine classifier for sentiment analysis"""
    
    def __init__(self, C: float = 1.0, max_iter: int = 1000, random_state: int = 42):
        """
        Initialize SVM classifier
        
        Parameters:
        -----------
        C : float
            Regularization parameter
        max_iter : int
            Maximum iterations
        random_state : int
            Random state for reproducibility
        """
        model = LinearSVC(C=C, max_iter=max_iter, random_state=random_state, dual='auto')
        super().__init__('SVM', model, random_state)
        self.C = C


class RandomForestClassifier_(SentimentClassifier):
    """Random Forest classifier for sentiment analysis"""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = None, 
                 min_samples_split: int = 2, random_state: int = 42):
        """
        Initialize Random Forest classifier
        
        Parameters:
        -----------
        n_estimators : int
            Number of trees
        max_depth : int
            Maximum tree depth
        min_samples_split : int
            Minimum samples to split node
        random_state : int
            Random state for reproducibility
        """
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
            n_jobs=-1
        )
        super().__init__('RandomForest', model, random_state)
        self.n_estimators = n_estimators
        self.max_depth = max_depth


class ClassifierEnsemble:
    """Ensemble of sentiment classifiers"""
    
    def __init__(self):
        self.classifiers = {}
        self.metrics = {}
        
    def add_classifier(self, name: str, classifier: SentimentClassifier):
        """Add classifier to ensemble"""
        self.classifiers[name] = classifier
        logger.info(f"Added {name} to ensemble")
        return self
    
    def fit_all(self, X, y):
        """Fit all classifiers"""
        for name, clf in self.classifiers.items():
            logger.info(f"Fitting {name}...")
            clf.fit(X, y)
        return self
    
    def evaluate_all(self, X, y) -> Dict[str, ClassifierMetrics]:
        """Evaluate all classifiers"""
        self.metrics = {}
        for name, clf in self.classifiers.items():
            logger.info(f"Evaluating {name}...")
            self.metrics[name] = clf.evaluate(X, y)
        return self.metrics
    
    def predict_all(self, X) -> Dict[str, np.ndarray]:
        """Get predictions from all classifiers"""
        predictions = {}
        for name, clf in self.classifiers.items():
            predictions[name] = clf.predict(X)
        return predictions
    
    def predict_ensemble(self, X, method: str = 'majority_vote') -> np.ndarray:
        """
        Get ensemble prediction
        
        Parameters:
        -----------
        X : array-like
            Features
        method : str
            Voting method: 'majority_vote', 'average_proba'
            
        Returns:
        --------
        np.ndarray
            Ensemble predictions
        """
        if method == 'majority_vote':
            predictions = self.predict_all(X)
            pred_array = np.column_stack(list(predictions.values()))
            # Majority vote
            return np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=1, arr=pred_array)
        
        elif method == 'average_proba':
            proba_list = []
            for clf in self.classifiers.values():
                try:
                    proba_list.append(clf.predict_proba(X)[:, 1])
                except Exception as e:
                    logger.warning(f"Could not get probabilities: {e}")
            
            if not proba_list:
                raise ValueError("No probabilities available for averaging")
            
            avg_proba = np.mean(proba_list, axis=0)
            return (avg_proba > 0.5).astype(int)
        
        else:
            raise ValueError(f"Unknown voting method: {method}")
    
    def get_best_classifier(self, metric: str = 'f1') -> Tuple[str, SentimentClassifier]:
        """Get best performing classifier by metric"""
        if not self.metrics:
            raise ValueError("Must evaluate classifiers first")
        
        best_name = None
        best_score = -1
        
        for name, metrics in self.metrics.items():
            score = getattr(metrics, metric)
            if score > best_score:
                best_score = score
                best_name = name
        
        return best_name, self.classifiers[best_name]
    
    def get_metrics_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary of metrics for all classifiers"""
        summary = {}
        for name, metrics in self.metrics.items():
            summary[name] = {
                'accuracy': metrics.accuracy,
                'precision': metrics.precision,
                'recall': metrics.recall,
                'f1': metrics.f1,
                'auc_score': metrics.auc_score
            }
        return summary
