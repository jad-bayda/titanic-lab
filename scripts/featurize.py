
import argparse
import pandas as pd
from titanic_lab.feature_engineering.feature_engineer import TitanicFeatureEngineer


# Command-line interface
def main():
    parser = argparse.ArgumentParser()

    # Input and output CSV paths
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    # Load preprocessed data
    df = pd.read_csv(args.input)

    # Apply feature engineering using the new class
    feature_engineer = TitanicFeatureEngineer()
    df = feature_engineer.transform(df)

    # Save the resulting dataset
    df.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
