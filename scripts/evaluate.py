
import argparse
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score


def main():
    # Read command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    # Load evaluation data
    df = pd.read_csv(args.input)

    # Separate target from features
    y = df["Survived"]
    X = df.drop(columns=["Survived"])

    # Convert categorical columns to numeric
    X = pd.get_dummies(X, drop_first=True)

    # Load trained model
    with open(args.model, "rb") as f:
        model = pickle.load(f)

    # Match columns with the columns used during training
    X = X.reindex(columns=model.model.feature_names_in_, fill_value=0)

    # Make predictions
    predictions = model.predict(X)

    # Calculate accuracy
    accuracy = accuracy_score(y, predictions)

    # Print evaluation metric
    print(f"Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
