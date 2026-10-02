import argparse
import pickle
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression


def build_parser():
    parser = argparse.ArgumentParser(description="Train Titanic survival model")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main():
    args = build_parser().parse_args()

    df = pd.read_csv(args.input)

    y = df["Survived"]
    X = df.drop(columns=["Survived"])

    # Convert categorical columns into numeric dummy variables
    X = pd.get_dummies(X, drop_first=True)

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump(model, f)

    print(f"Model saved to {output_path}")


if __name__ == "__main__":
    main()