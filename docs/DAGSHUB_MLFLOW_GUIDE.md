# MLflow + DagsHub Integration Guide

## Quick Setup for Sentiment Classifier Experiment Tracking

### 1. Get DagsHub Credentials

1. Go to [https://dagshub.com](https://dagshub.com) and sign up/log in
2. Navigate to your repository: `https://dagshub.com/youssef.maged237/sentiment-classifier`
3. Get your authentication token:
   - Click on your profile icon → Settings
   - Go to "Security" or "Tokens"
   - Create a new token (or copy existing one)
4. Copy the token

### 2. Configure Local Environment

**Option A: Using Environment Variable (Recommended)**

```bash
# On Linux/Mac
export DAGSHUB_TOKEN="your_token_here"

# On Windows (PowerShell)
$env:DAGSHUB_TOKEN="your_token_here"

# On Windows (CMD)
set DAGSHUB_TOKEN=your_token_here
```

**Option B: Using .env file**

1. Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

2. Edit `.env` and fill in:
```
DAGSHUB_REPO_URL=https://dagshub.com/youssef.maged237/sentiment-classifier
DAGSHUB_TOKEN=your_token_here
```

3. Load environment variables (in Python notebook):
```python
from dotenv import load_dotenv
load_dotenv()
```

### 3. Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt

# Or using uv (faster)
uv add mlflow pyyaml python-dotenv
```

### 4. Run Training with MLflow Tracking

**Option A: Notebook (Recommended for exploration)**

```bash
# Start Jupyter
jupyter notebook

# Open: notebooks/03_sentiment_training.ipynb
# Run all cells - experiments will be tracked automatically to DagsHub
```

**Option B: Command Line Script**

```bash
python src/train.py \
  --dagshub-repo "https://dagshub.com/youssef.maged237/sentiment-classifier" \
  --dagshub-token "$DAGSHUB_TOKEN"
```

**Option C: With Custom Config**

```bash
python src/train.py \
  --config config/training_config.yaml \
  --amazon-path data/processed/amazon_processed.csv \
  --twitter-path data/processed/twitter_processed.csv \
  --sample-size 2000
```

### 5. View Experiment Results

**On DagsHub (Cloud)**

1. Go to your repository: `https://dagshub.com/youssef.maged237/sentiment-classifier`
2. Click on "Experiments" tab
3. View all runs, metrics, and parameters tracked

**Locally (MLflow UI)**

```bash
# If not using DagsHub, view locally
mlflow ui

# Open browser to http://localhost:5000
```

## What Gets Tracked?

### Parameters Logged
- Vectorizer type and configuration (max_features, min_df, max_df)
- Model hyperparameters (C, max_iter, n_estimators, max_depth)
- Data split sizes (train/test)
- Vocabulary size

### Metrics Logged
- **Accuracy**: Correct predictions / total predictions
- **Precision**: True positives / (true positives + false positives)
- **Recall**: True positives / (true positives + false negatives)
- **F1 Score**: Harmonic mean of precision and recall
- **AUC Score**: Area under ROC curve (when available)

### Artifacts Logged
- Trained models (sklearn format)
- Classification reports (text)
- Confusion matrices (in metrics)
- Training configuration

### Run Structure

```
sentiment-classification (Experiment)
├── LogisticRegression (Run)
│   ├── Parameters: {C, max_iter, train_size, test_size, ...}
│   ├── Metrics: {accuracy, precision, recall, f1, auc_score}
│   ├── Models: logistic_regression/
│   └── Artifacts: LogisticRegression_report.txt
├── SVM (Run)
│   ├── Parameters: {...}
│   ├── Metrics: {...}
│   └── ...
├── RandomForest (Run)
│   └── ...
├── ensemble_majority_vote (Run)
│   ├── Parameters: {ensemble_method, num_models}
│   ├── Metrics: {ensemble_accuracy, ensemble_precision, ...}
│   └── ...
└── metadata (Run)
    └── training_metadata.json
```

## MLflow Model Registry (Advanced)

After running training, register the best model:

```python
import mlflow

# Register best model
model_uri = f"runs:/{best_run_id}/model_name"
mlflow.register_model(model_uri, "sentiment-classifier-prod")

# Load registered model
loaded_model = mlflow.pyfunc.load_model("models:/sentiment-classifier-prod/production")

# Make predictions
predictions = loaded_model.predict(new_data)
```

## Troubleshooting

### Error: "DAGSHUB_TOKEN not set"
- Make sure token is set as environment variable: `echo $DAGSHUB_TOKEN`
- Or set it before running: `export DAGSHUB_TOKEN=your_token && python script.py`

### Error: "Connection to DagsHub failed"
- Check token is valid
- Check URL is correct
- Try local MLflow first: `mlflow set-tracking-uri sqlite:///mlruns.db`

### Want to reset experiments?
- Delete local database: `rm -rf mlruns.db mlruns/`
- Clear DagsHub experiments via web UI
- Run training again fresh

## Tips & Best Practices

1. **Version Control**: Commit `config/training_config.yaml` to track configuration changes
2. **Tags**: Add custom tags to runs for easier filtering:
   ```python
   mlflow.set_tags({"dataset": "amazon+twitter", "version": "v1"})
   ```
3. **Notes**: Add notes to runs in DagsHub UI describing experiment changes
4. **Compare Runs**: Use DagsHub UI to compare metrics across multiple runs
5. **Model Promotion**: Use MLflow Model Registry to promote models to production

## References

- MLflow Documentation: https://mlflow.org/docs/
- DagsHub Platform: https://dagshub.com/docs/
- DagsHub MLflow Integration: https://dagshub.com/docs/integration_guide/mlflow/
- Scikit-learn Classification Metrics: https://scikit-learn.org/stable/modules/model_evaluation.html
