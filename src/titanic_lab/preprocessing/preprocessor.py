import pandas as pd
from .base_preprocessor import BasePreprocessor


class TitanicPreprocessor(BasePreprocessor):

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
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