import argparse
import pandas as pd


def preprocess_data(df):
    df = df.copy()

    if "Cabin" in df.columns:
        df.drop(columns=["Cabin"], inplace=True)

    if "Embarked" in df.columns:
        df["Embarked"] = df["Embarked"].fillna("S")

    if "Fare" in df.columns:
        df["Fare"] = df["Fare"].fillna(df["Fare"].mean())

    if "Age" in df.columns:
        df["Age"] = df.groupby(["Sex", "Pclass"])["Age"].transform(
            lambda x: x.fillna(x.median())
        )

    return df


def main():
    parser = argparse.ArgumentParser(description="Preprocess Titanic data")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df = preprocess_data(df)
    df.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()