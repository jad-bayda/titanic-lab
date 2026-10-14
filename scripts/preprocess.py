import argparse
import pandas as pd
from titanic_lab.preprocessing.preprocessor import TitanicPreprocessor


def main():
    parser = argparse.ArgumentParser(description="Preprocess Titanic data")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    preprocessor = TitanicPreprocessor()
    df = preprocessor.process(df)

    df.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()