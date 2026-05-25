# Sentiment Classifier - MLflow + DagsHub Tracking

Complete sentiment classification pipeline with experiment tracking using MLflow and DagsHub integration.

## 📊 Project Overview

This project builds a sentiment classifier on Amazon and Twitter datasets using multiple machine learning models with full experiment tracking.

### Components

1. **Data Processing**: `src/preprocessor.py`
   - Text cleaning, tokenization, normalization
   - Lemmatization and stopword removal
   - Domain-specific configurations (Amazon vs Twitter)

2. **Vectorization**: `src/vectorizer.py`
   - Bag of Words (BoW)
   - TF-IDF with IDF scoring
   - BM25 ranking model

3. **Classification**: `src/classifier.py`
   - Logistic Regression
   - Support Vector Machine (SVM/LinearSVC)
   - Random Forest
   - Ensemble with majority voting

4. **Training & Tracking**: `src/train.py`
   - MLflow integration for experiment tracking
   - DagsHub integration for cloud synchronization
   - Configurable hyperparameters

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip or conda
- DagsHub account (optional, for cloud tracking)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Setup DagsHub (Optional but Recommended)

```bash
# Set your DagsHub token
export DAGSHUB_TOKEN="your_token_here"

# Or create .env file
cp .env.example .env
# Edit .env and fill in your credentials
```

See [DAGSHUB_MLFLOW_GUIDE.md](docs/DAGSHUB_MLFLOW_GUIDE.md) for detailed setup instructions.

### 3. Run Training

**Via Jupyter Notebook (Interactive)**

```bash
jupyter notebook notebooks/03_sentiment_training.ipynb
```

**Via Command Line**

```bash
python src/train.py \
  --dagshub-repo "https://dagshub.com/youssef.maged237/sentiment-classifier" \
  --dagshub-token "$DAGSHUB_TOKEN" \
  --sample-size 2000
```

## 📁 Project Structure

```
sentiment-classifier/
├── data/
│   ├── raw/                          # Original datasets
│   └── processed/
│       ├── amazon_processed.csv      # Preprocessed Amazon reviews
│       └── twitter_processed.csv     # Preprocessed tweets
├── src/
│   ├── preprocessor.py               # Text preprocessing pipeline
│   ├── vectorizer.py                 # Text vectorization (BoW, TF-IDF, BM25)
│   ├── classifier.py                 # Classification models & ensemble
│   └── train.py                      # Training script with MLflow tracking
├── notebooks/
│   ├── 01_preprocessing_exploration.ipynb
│   ├── 02_vectorization.ipynb
│   └── 03_sentiment_training.ipynb   # Main training & experiment tracking
├── config/
│   └── training_config.yaml          # Training hyperparameters
├── reports/
│   └── classification_results.csv    # Training results summary
├── docs/
│   ├── DAGSHUB_MLFLOW_GUIDE.md       # MLflow + DagsHub setup guide
│   └── commands.rst
├── requirements.txt
├── .env.example                      # Environment variables template
└── README.md
```

## 📈 Model Comparison

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | ~0.85 | ~0.85 | ~0.85 | ~0.85 |
| SVM | ~0.86 | ~0.86 | ~0.86 | ~0.86 |
| Random Forest | ~0.84 | ~0.84 | ~0.84 | ~0.84 |
| **Ensemble (Voting)** | ~0.87 | ~0.87 | ~0.87 | ~0.87 |

*Note: Actual results depend on dataset size and configuration*

## 🔬 Experiment Tracking

### What's Tracked to MLflow?

- **Models**: Trained classifier objects
- **Parameters**: Hyperparameters, vectorizer config, data splits
- **Metrics**: Accuracy, Precision, Recall, F1, AUC
- **Artifacts**: Classification reports, confusion matrices
- **Metadata**: Dataset info, vocabulary size, training time

### View Experiments

**DagsHub (Cloud)**
```
https://dagshub.com/youssef.maged237/sentiment-classifier/experiments
```

**Local MLflow UI**
```bash
mlflow ui
# Navigate to http://localhost:5000
```

## 💡 Key Features

### 1. Configurable Pipeline
Edit `config/training_config.yaml` to customize:
- Vectorization method (BoW/TF-IDF)
- Hyperparameters for each classifier
- Data split ratio
- Feature engineering options (n-grams)

### 2. Ensemble Methods
- **Majority Voting**: Each model votes, majority wins
- **Average Probabilities**: Average prediction scores (for compatible models)

### 3. Multiple Classifiers
- **Logistic Regression**: Fast, interpretable, good baseline
- **SVM**: Good generalization, effective in high dimensions
- **Random Forest**: Handles non-linear patterns, robust

### 4. Full Reproducibility
- All experiments logged to MLflow
- Configuration files version controlled
- Data versioning via DagsHub
- Code versioning via Git

## 🔧 Training Configuration

Edit `config/training_config.yaml`:

```yaml
random_state: 42
test_size: 0.2
vectorizer: 'tfidf'  # or 'bow'

vectorizer_config:
  max_features: 5000
  min_df: 2
  max_df: 0.95
  ngram_range: [1, 1]  # [1, 2] for unigrams + bigrams

classifiers:
  logistic_regression:
    C: 1.0
    max_iter: 1000
  
  svm:
    C: 1.0
    max_iter: 1000
  
  random_forest:
    n_estimators: 100
    max_depth: 10
```

## 📊 Notebooks

### 01_preprocessing_exploration.ipynb
- Load raw datasets (Amazon, Twitter)
- Apply preprocessing pipeline
- Compare effects of different configurations
- Visualize text statistics

### 02_vectorization.ipynb
- Implement BoW, TF-IDF, BM25 vectorizers
- Compare vectorization methods
- Analyze vocabulary and sparsity
- Benchmark performance

### 03_sentiment_training.ipynb
- **Main notebook for sentiment classification**
- Load and prepare data
- Vectorize with TF-IDF
- Train 3 classifiers
- Evaluate with confusion matrices
- Create ensemble
- Track experiments to MLflow/DagsHub

## 🎯 Usage Examples

### Training from Notebook

```python
from src.classifier import LogisticRegressionClassifier
from src.vectorizer import TfidfVectorizer_

# Create and train
vectorizer = TfidfVectorizer_()
vectorizer.fit(X_train)
X_train_vec = vectorizer.transform(X_train).toarray()

clf = LogisticRegressionClassifier()
clf.fit(X_train_vec, y_train)

# Evaluate
metrics = clf.evaluate(X_test_vec, y_test)
print(f"F1 Score: {metrics.f1:.4f}")
```

### Training from Command Line

```bash
# Default settings
python src/train.py

# With DagsHub tracking
python src/train.py \
  --dagshub-repo "https://dagshub.com/youssef.maged237/sentiment-classifier" \
  --dagshub-token "your_token"

# With custom config
python src/train.py \
  --config config/training_config.yaml \
  --sample-size 1000
```

### Making Predictions

```python
import mlflow

# Load best model
model = mlflow.pyfunc.load_model("path/to/model")

# Make predictions
predictions = model.predict(new_data)
```

## 🚨 Troubleshooting

### Issue: SVM predictions not working
- LinearSVC doesn't have `predict_proba` by default
- We use `decision_function` and sigmoid conversion
- AUC score may be unavailable

### Issue: Out of memory with large dataset
- Use `--sample-size` parameter to limit data
- Reduce `max_features` in vectorizer config
- Use sparse matrix format instead of dense

### Issue: DagsHub not syncing
- Verify token is set: `echo $DAGSHUB_TOKEN`
- Check URL format: `https://dagshub.com/username/repo`
- See [DAGSHUB_MLFLOW_GUIDE.md](docs/DAGSHUB_MLFLOW_GUIDE.md)

## 📚 References

- [MLflow Documentation](https://mlflow.org)
- [DagsHub Platform](https://dagshub.com)
- [Scikit-learn Classification](https://scikit-learn.org/stable/modules/classification.html)
- [Sentiment Analysis Overview](https://en.wikipedia.org/wiki/Sentiment_analysis)

## 📝 License

This project is part of the ITI Course - Sequential Modeling and LLMs

## 👤 Author

Created as part of ITI Day 1 Lab - Sentiment Classification

---

**Happy Experimenting! 🚀**

For more details, see [DAGSHUB_MLFLOW_GUIDE.md](docs/DAGSHUB_MLFLOW_GUIDE.md)
