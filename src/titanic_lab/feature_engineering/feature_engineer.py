import pandas as pd
from .base_feature_engineer import BaseFeatureEngineer


def family_size(number):
    if number == 1:
        return "Alone"
    elif number > 1 and number < 5:
        return "Small"
    else:
        return "Large"


class TitanicFeatureEngineer(BaseFeatureEngineer):

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

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