import argparse
import pickle
from pathlib import Path

import pandas as pd


def main():
    # Read command-line arguments
    parser = argparse.ArgumentParser(description="Make Titanic predictions")
    parser.add_argument("--input", required=True, help="Input feature CSV")
    parser.add_argument("--model", required=True, help="Trained model file")
    parser.add_argument("--output", required=True, help="Output predictions CSV")
    args = parser.parse_args()

    # Load test features
    df = pd.read_csv(args.input)

    # Remove target column if it exists
    X = df.drop(columns=["Survived"], errors="ignore")

    # Convert categorical columns into numeric dummy variables
    X = pd.get_dummies(X)

    # Load trained model
    with open(args.model, "rb") as f:
        model = pickle.load(f)

    # Match columns with the columns used during training
    if hasattr(model, "feature_names_in_"):
        X = X.reindex(columns=model.feature_names_in_, fill_value=0)

    # Make predictions
    predictions = model.predict(X)

    # Create output folder if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save predictions
    pd.DataFrame({"Survived": predictions}).to_csv(output_path, index=False)

    print(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    main()