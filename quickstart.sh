#!/bin/bash
# Quick Start Script for Sentiment Classifier with MLflow + DagsHub

set -e  # Exit on error

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║     Sentiment Classifier - MLflow + DagsHub Quick Start        ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Check Python
echo "✓ Checking Python installation..."
python --version

# Install dependencies
echo ""
echo "📦 Installing dependencies..."
pip install -r requirements.txt -q

# Check DagsHub token
echo ""
if [ -z "$DAGSHUB_TOKEN" ]; then
    echo "⚠️  DAGSHUB_TOKEN not set. MLflow will track locally."
    echo ""
    echo "To enable DagsHub cloud tracking:"
    echo "  1. Get token: https://dagshub.com/user/settings/tokens"
    echo "  2. Set token: export DAGSHUB_TOKEN='your_token_here'"
    echo "  3. Re-run this script"
    echo ""
    TRACKING_MODE="LOCAL"
else
    echo "✓ DagsHub token found"
    TRACKING_MODE="DAGSHUB"
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "                    SETUP COMPLETE!"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Next steps:"
echo ""
echo "1️⃣  View Training Notebook (Interactive):"
echo "    jupyter notebook notebooks/03_sentiment_training.ipynb"
echo ""
echo "2️⃣  Or Run Training Script:"
echo "    python src/train.py"
echo ""
echo "3️⃣  View Experiments:"
if [ "$TRACKING_MODE" = "DAGSHUB" ]; then
    echo "    📊 DagsHub: https://dagshub.com/youssef.maged237/sentiment-classifier/experiments"
fi
echo "    📊 Local MLflow UI: mlflow ui"
echo ""
echo "4️⃣  Read Documentation:"
echo "    📖 MLflow Setup: docs/DAGSHUB_MLFLOW_GUIDE.md"
echo "    📖 Project Guide: docs/CLASSIFIER_README.md"
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "Tracking Mode: $TRACKING_MODE"
echo "════════════════════════════════════════════════════════════════"
