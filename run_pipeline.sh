#!/bin/bash

# Stop the pipeline if any command fails
set -e

# Create required output folders
mkdir -p data/processed
mkdir -p data/features
mkdir -p models
mkdir -p predictions

# 1. Preprocess training and test data
uv run python scripts/preprocess.py --input data/titanic/train.csv --output data/processed/train_processed.csv
uv run python scripts/preprocess.py --input data/titanic/test.csv --output data/processed/test_processed.csv

# 2. Create features
uv run python scripts/featurize.py --input data/processed/train_processed.csv --output data/features/train_features.csv
uv run python scripts/featurize.py --input data/processed/test_processed.csv --output data/features/test_features.csv

# 3. Train the model
uv run python scripts/train.py --input data/features/train_features.csv --output models/model.pkl

# 4. Make predictions
uv run python scripts/predict.py --input data/features/test_features.csv --model models/model.pkl --output predictions/predictions.csv

# 5. Evaluate the model
uv run python scripts/evaluate.py --input data/features/train_features.csv --model models/model.pkl

echo "Pipeline completed successfully!"