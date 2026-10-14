
import argparse
import pickle
from pathlib import Path

import pandas as pd

from titanic_lab.models.logistic_regression import LogisticRegressionModel
from titanic_lab.models.random_forest import RandomForestModel
from titanic_lab.models.xgboost_model import XGBoostModel


def build_parser():
    parser = argparse.ArgumentParser(description="Train Titanic survival model")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--model",
        choices=["logistic", "random_forest", "xgboost"],
        default="logistic"
    )
    return parser


def main():
    args = build_parser().parse_args()

    df = pd.read_csv(args.input)

    y = df["Survived"]
    X = df.drop(columns=["Survived"])

    # Convert categorical columns into numeric dummy variables
    X = pd.get_dummies(X, drop_first=True)

    # Select the model
    if args.model == "logistic":
        model = LogisticRegressionModel()
    elif args.model == "random_forest":
        model = RandomForestModel()
    else:
        model = XGBoostModel()

    # Train the model
    model.train(X, y)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the trained model
    with open(output_path, "wb") as f:
        pickle.dump(model, f)

    print(f"Model: {args.model}")
    print(f"Model saved to {output_path}")


if __name__ == "__main__":
    main()
