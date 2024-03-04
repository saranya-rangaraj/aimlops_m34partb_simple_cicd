import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# This mapper in used in iris model
class Mapper(BaseEstimator, TransformerMixin):
    """Categorical variable mapper."""
    def __init__(self, variables: str, mappings: dict):

        if not isinstance(variables, str):
            raise ValueError("variables should be a str")

        self.variables = variables
        self.mappings = mappings

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # pylint: disable=unused-argument
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[self.variables] = X[self.variables].map(self.mappings).astype(int)

        return X
