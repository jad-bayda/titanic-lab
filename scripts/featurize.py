import argparse
import pandas as pd


# Convert numerical family size into categories
def family_size(number):
    if number == 1:
        return "Alone"
    elif number > 1 and number < 5:
        return "Small"
    else:
        return "Large"


# Feature engineering function
def featurize(df):

    # Extract passenger title from the Name column
    df["Title"] = df["Name"].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]

    # Group uncommon titles into the "Rare" category
    df["Title"] = df["Title"].replace(
        [
            "Lady",
            "the Countess",
            "Capt",
            "Col",
            "Don",
            "Dr",
            "Major",
            "Rev",
            "Sir",
            "Jonkheer",
            "Dona"
        ],
        "Rare"
    )

    # Standardize similar titles
    df["Title"] = df["Title"].replace("Mlle", "Miss")
    df["Title"] = df["Title"].replace("Ms", "Miss")
    df["Title"] = df["Title"].replace("Mme", "Mrs")

    # Create total family size
    # +1 represents the passenger themself
    df["Family_size"] = df["SibSp"] + df["Parch"] + 1

    # Convert family size into Alone, Small, or Large
    df["Family_size"] = df["Family_size"].apply(family_size)

    # Remove columns that are no longer needed
    df.drop(columns=["Name", "Parch", "SibSp", "Ticket"], inplace=True)

    return df


# Command-line interface
def main():
    parser = argparse.ArgumentParser()

    # Input and output CSV paths
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    # Load preprocessed data
    df = pd.read_csv(args.input)

    # Apply feature engineering
    df = featurize(df)

    # Save the resulting dataset
    df.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()