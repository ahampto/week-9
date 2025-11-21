import pandas as pd
import numpy as np



class GroupEstimate:
    def __init__(self, estimate):
        valid_estimates = {"mean", "median"}
        if estimate not in valid_estimates:
            raise ValueError(f"Invalid estimate: {estimate}. Must be 'mean' or 'median'.")
        self.estimate = estimate
        self.group_estimates_ = None 

    def fit(self, X, y):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of rows.")
        if pd.isnull(y).any():
            raise ValueError("y contains missing values.")

        # Combine X and y into a single DataFrame
        df = X.copy()
        df["__y__"] = y

        # Compute group-level estimates
        if self.estimate == "mean":
            grouped = df.groupby(list(X.columns))["__y__"].mean()
        else:  # "median"
            grouped = df.groupby(list(X.columns))["__y__"].median()

        self.group_estimates_ = grouped.reset_index()
        self.group_estimates_.rename(columns={"__y__": "estimate"}, inplace=True)

        return self

    def predict(self, X_):
        # Part 3: predict method
        if self.group_estimates_ is None:
            raise RuntimeError("The model must be fitted before prediction.")

        if not isinstance(X_, pd.DataFrame):
            X_ = pd.DataFrame(X_, columns=self.group_estimates_.columns[:-1])

        merged = X_.merge(self.group_estimates_, on=list(X_.columns), how="left")
        missing_count = merged["estimate"].isna().sum()

        if missing_count > 0:
            print(f"Warning: {missing_count} group(s) not found in training data. Returning NaN for these.")

        return merged["estimate"].values
