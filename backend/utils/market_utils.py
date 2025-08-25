import pandas as pd

def extract_scalar(value):
    if isinstance(value, pd.Series):
        return value.iloc[0]
    return value