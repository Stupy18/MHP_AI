import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class TimeSlotAggregator(BaseEstimator, TransformerMixin):
    def __init__(self, time_columns_indices):
        # Expect indices, not names
        self.time_columns_indices = time_columns_indices

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # X is expected to be a NumPy array here
        # Compute 'booked_anytime' based on the specific column indices
        booked_anytime = (X[:, self.time_columns_indices].max(axis=1) > 0).astype(int)
        # Return result with 'booked_anytime' appended as a new column
        return np.hstack([X, booked_anytime.reshape(-1, 1)])
