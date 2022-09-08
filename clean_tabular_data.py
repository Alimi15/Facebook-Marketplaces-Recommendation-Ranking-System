import pandas as pd
import numpy as np

def remove_missing_rows(df: pd.DataFrame):
    df.drop(df[df["location"].str.len() == 0].index, inplace=True)
    df.drop(df[df["location"] == np.nan].index, inplace=True)
    df.drop(df[df["price"] == np.nan].index, inplace=True)
    df.drop(df[df["product_description"] == np.nan].index, inplace=True)
    df.drop(df[df["product_description"].str.len() == 0].index, inplace=True)
    df.drop(df[df["category"] == np.nan].index, inplace=True)
    df.drop(df[df["category"].str.len() == 0].index, inplace=True)
    df.drop(df[df["product_name"] == np.nan].index, inplace=True)
    df.drop(df[df["product_name"].str.len() == 0].index, inplace=True)
    return df