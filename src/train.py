"""
Training script for sentiment classification with MLflow tracking
Integrates with DagsHub for experiment tracking
"""

import os
import sys
import logging
import yaml
import json
from pathlib import Path
from typing import Dict, Any
import joblib

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

sys.path.append(str(Path(__file__).parent))

from preprocessor import TextPreprocessor, PreprocessorConfig
from vectorizer import BagOfWordsVectorizer, TfidfVectorizer_, VectorizerConfig
from classifier import (
    LogisticRegressionClassifier,
    SVMClassifier,
    RandomForestClassifier_,
    ClassifierEnsemble
)

from dotenv import load_dotenv
load_dotenv()
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DagsHubMLflowSetup:
    """Setup MLflow with DagsHub remote tracking"""
    
    @staticmethod
    def configure(repo_url: str, token: str = None):
        """
        Configure MLflow to use DagsHub as remote tracking server
        
        Parameters:
        -----------
        repo_url : str
            DagsHub repository URL (e.g., https://dagshub.com/user/repo)
        token : str
            DagsHub token (optional, can be set via env var)
        """
        if token is None:
            token = os.getenv('DAGSHUB_TOKEN')
        
        if not token:
            logger.warning("DAGSHUB_TOKEN not set. MLflow will track locally only.")
            return
        
        # Parse repo URL
        if 'dagshub.com' not in repo_url:
            raise ValueError("Invalid DagsHub URL format")
        
        # Extract username and repo name from URL
        parts = repo_url.rstrip('/').split('/')
        username = parts[-2]
        repo_name = parts[-1]
        
        # Set MLflow tracking URI to DagsHub
        dagshub_mlflow_uri = f"https://dagshub.com/{username}/{repo_name}.mlflow"
        
        mlflow.set_tracking_uri(dagshub_mlflow_uri)
        
        # Set credentials
        os.environ['MLFLOW_TRACKING_USERNAME'] = username
        os.environ['MLFLOW_TRACKING_PASSWORD'] = token
        
        logger.info(f"MLflow configured to track on DagsHub: {dagshub_mlflow_uri}")


class SentimentTrainer:
    """Main training orchestrator"""
    
    def __init__(self, config_path: str = None):
        """
        Initialize trainer
        
        Parameters:
        -----------
        config_path : str
            Path to training config file
        """
        self.config = self._load_config(config_path)
        self.data = {}
        self.vectorizers = {}
        self.classifiers = {}
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load config from YAML file"""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        
        # Default config
        return {
            'random_state': 42,
            'test_size': 0.2,
            'vectorizer': 'tfidf',
            'vectorizer_config': {
                'max_features': 5000,
                'min_df': 2,
                'max_df': 0.95,
                'ngram_range': (1, 1)
            },
            'classifiers': {
                'logistic_regression': {'C': 1.0, 'max_iter': 1000},
                'svm': {'C': 1.0, 'max_iter': 1000},
                'random_forest': {'n_estimators': 100, 'max_depth': 10}
            }
        }
    
    def load_data(self, amazon_path: str, twitter_path: str, sample_size: int = None):
        """Load and combine datasets"""
        logger.info("Loading datasets...")
        
        amazon_df = pd.read_csv(amazon_path)
        twitter_df = pd.read_csv(twitter_path)
        
        if sample_size:
            amazon_df = amazon_df.head(sample_size)
            twitter_df = twitter_df.head(sample_size)
        
        # Combine datasets
        combined_df = pd.concat([amazon_df, twitter_df], ignore_index=True)
        #combined_df = combined_df.sample(frac=1, random_state=self.config['random_state']).reset_index(drop=True)
        
        logger.info(f"Loaded {len(combined_df):,} documents (Amazon: {len(amazon_df):,}, Twitter: {len(twitter_df):,})")
        
        self.data['X'] = combined_df['processed'].values
        self.data['y'] = combined_df['label'].values
        self.data['full_df'] = combined_df
        
        return self
    
    def split_data(self):
        """Split data into train/test"""
        X_train, X_test, y_train, y_test = train_test_split(
            self.data['X'],
            self.data['y'],
            test_size=self.config['test_size'],
            random_state=self.config['random_state'],
            stratify=self.data['y']
        )
        
        self.data['X_train'] = X_train
        self.data['X_test'] = X_test
        self.data['y_train'] = y_train
        self.data['y_test'] = y_test
        
        logger.info(f"Split data: Train {len(X_train)}, Test {len(X_test)}")
        return self
    
    def vectorize_data(self):
        """Vectorize text data"""
        vectorizer_type = self.config['vectorizer'].lower()
        vec_config = VectorizerConfig(**self.config['vectorizer_config'])
        
        if vectorizer_type == 'bow':
            vectorizer = BagOfWordsVectorizer(vec_config)
        elif vectorizer_type == 'tfidf':
            vectorizer = TfidfVectorizer_(vec_config)
        else:
            raise ValueError(f"Unknown vectorizer: {vectorizer_type}")
        
        logger.info(f"Vectorizing with {vectorizer_type.upper()}...")

        self.data['X_train'] = [x if isinstance(x, str) else "" for x in self.data['X_train']]
        self.data['X_test'] = [x if isinstance(x, str) else "" for x in self.data['X_test']]

        vectorizer.fit(self.data['X_train'])
        
        X_train_vec = vectorizer.transform(self.data['X_train'])
        X_test_vec = vectorizer.transform(self.data['X_test'])
        
        self.data['X_train_vec'] = X_train_vec
        self.data['X_test_vec'] = X_test_vec
        self.vectorizers['active'] = vectorizer
        
        logger.info(f"Vectorized shape: {self.data['X_train_vec'].shape}")
        return self
    
    def train_classifiers(self, experiment_name: str = "sentiment-classification"):
        """Train all classifiers"""

        ensemble = ClassifierEnsemble()
        
        os.makedirs("models", exist_ok=True)

        for clf_name, clf_params in self.config['classifiers'].items():
            logger.info(f"Training {clf_name}...")
            
            with mlflow.start_run(run_name=clf_name, nested=True):
                # Create classifier
                if clf_name == 'logistic_regression':
                    classifier = LogisticRegressionClassifier(**clf_params)
                elif clf_name == 'svm':
                    classifier = SVMClassifier(**clf_params)
                elif clf_name == 'random_forest':
                    classifier = RandomForestClassifier_(**clf_params)
                else:
                    logger.warning(f"Unknown classifier: {clf_name}")
                    continue
                
                # Train
                classifier.fit(self.data['X_train_vec'], self.data['y_train'])
                
                # Evaluate
                metrics = classifier.evaluate(self.data['X_test_vec'], self.data['y_test'])

                model_path = os.path.join("models", f"{clf_name}.joblib")
                joblib.dump(classifier.model, model_path)
                logger.info(f"Saved {clf_name} model locally to {model_path}")
                
                # Log parameters
                mlflow.log_params(clf_params)
                
                # Log metrics
                mlflow.log_metrics({
                    'accuracy': metrics.accuracy,
                    'precision': metrics.precision,
                    'recall': metrics.recall,
                    'f1': metrics.f1,
                    'auc_score': metrics.auc_score if metrics.auc_score else 0
                })
                
                # Log model
                mlflow.sklearn.log_model(classifier.model, clf_name)
                
                # Log confusion matrix and classification report
                mlflow.log_text(metrics.classification_report, "classification_report.txt")
                
                logger.info(f"  Accuracy: {metrics.accuracy:.4f}, F1: {metrics.f1:.4f}")
                
                ensemble.add_classifier(clf_name, classifier)
        
        ensemble_path = os.path.join("models", "ensemble_classifier.joblib")
        joblib.dump(ensemble, ensemble_path)
        logger.info(f"Saved Ensemble model locally to {ensemble_path}")

        self.classifiers['ensemble'] = ensemble
        return self
    
    def evaluate_ensemble(self):
        """Evaluate ensemble predictions"""
        logger.info("Evaluating ensemble...")
        
        ensemble = self.classifiers['ensemble']
        ensemble.evaluate_all(self.data['X_test_vec'], self.data['y_test'])
        
        with mlflow.start_run(run_name="ensemble", nested=True):
            # Get summary
            summary = ensemble.get_metrics_summary()
            
         # Log individual classifier metrics
            for clf_name, metrics_dict in summary.items():
                for metric_name, value in metrics_dict.items():
                    if value is not None:
                        mlflow.log_metric(f"{clf_name}/{metric_name}", value)

            # Get ensemble predictions
            ensemble_pred_majority = ensemble.predict_ensemble(self.data['X_test_vec'], method='majority_vote')

            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            y_test = self.data['y_test']

            acc = accuracy_score(y_test, ensemble_pred_majority)
            prec = precision_score(y_test, ensemble_pred_majority, average='weighted')
            rec = recall_score(y_test, ensemble_pred_majority, average='weighted')
            f1 = f1_score(y_test, ensemble_pred_majority, average='weighted')

            # Log ensemble metrics
            mlflow.log_metrics({
                'ensemble_accuracy': acc,
                'ensemble_precision': prec,
                'ensemble_recall': rec,
                'ensemble_f1': f1,
            })
            
            logger.info(f"Ensemble F1: {f1:.4f}")
        
        return self
    
    def log_metadata(self):
        """Log training metadata"""
        metadata = {
            'dataset_size': len(self.data['X']),
            'train_size': len(self.data['X_train']),
            'test_size': len(self.data['X_test']),
            'vectorizer': self.config['vectorizer'],
            'vocab_size': len(self.vectorizers['active'].get_feature_names()),
            'num_classifiers': len(self.config['classifiers'])
        }

        mlflow.log_params(metadata)
        mlflow.log_dict(metadata, "training_metadata.json")
        
        logger.info("Metadata logged to MLflow")
    
    def run(self, amazon_path: str, twitter_path: str, sample_size: int = None):
        """Run full training pipeline"""
        try:
            self.load_data(amazon_path, twitter_path, sample_size)
            self.split_data()
            self.vectorize_data()

            mlflow.set_experiment("sentiment-classification")

            with mlflow.start_run(run_name="full_pipeline"):
                self.train_classifiers()
                self.evaluate_ensemble()
                self.log_metadata()
            
            logger.info("Training completed successfully!")
            return self
        
        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            raise


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train sentiment classifier with MLflow tracking')
    parser.add_argument('--amazon-path', default='data/processed/amazon_processed.csv',
                       help='Path to Amazon dataset')
    parser.add_argument('--twitter-path', default='data/processed/twitter_processed.csv',
                       help='Path to Twitter dataset')
    parser.add_argument('--config', help='Path to config YAML file')
    parser.add_argument('--sample-size', type=int, help='Sample size for each dataset')
    parser.add_argument('--dagshub-repo', help='DagsHub repository URL')
    parser.add_argument('--dagshub-token', help='DagsHub token (or set DAGSHUB_TOKEN env var)')
    
    args = parser.parse_args()
    
    DAGSHUB_REPO_URL = os.getenv('DAGSHUB_REPO_URL')
    DAGSHUB_TOKEN = os.getenv('DAGSHUB_TOKEN')

    
    parts = DAGSHUB_REPO_URL.rstrip('/').split('/')
    username = parts[-2]
    repo_name = parts[-1]
    
    # Configure MLflow
    mlflow.set_tracking_uri(f"https://dagshub.com/{username}/{repo_name}.mlflow")
    os.environ['MLFLOW_TRACKING_USERNAME'] = username
    os.environ['MLFLOW_TRACKING_PASSWORD'] = DAGSHUB_TOKEN
    
    print(f"✓ MLflow configured for DagsHub tracking: {DAGSHUB_REPO_URL}")
    
    # Run training
    trainer = SentimentTrainer(config_path=args.config)
    trainer.run(
        amazon_path=args.amazon_path,
        twitter_path=args.twitter_path,
        sample_size=args.sample_size
    )


if __name__ == '__main__':
    main()
