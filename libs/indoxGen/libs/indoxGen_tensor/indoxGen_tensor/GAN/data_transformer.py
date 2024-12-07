import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd


class DataTransformer:
    """
    DataTransformer class for handling transformations and inverse transformations
    of categorical, mixed, and numerical columns for tabular GAN data preprocessing.

    Attributes:
    -----------
    categorical_columns : List[str]
        List of columns containing categorical data.
    mixed_columns : Dict[str, str]
        Dictionary mapping columns with mixed constraints, such as 'positive', 'negative', etc.
    integer_columns : List[str]
        List of columns containing integer data.
    scalers : Dict[str, StandardScaler]
        Dictionary containing fitted scalers for numerical columns.
    encoders : Dict[str, OneHotEncoder]
        Dictionary containing fitted one-hot encoders for categorical columns.
    column_order : List[str]
        List of columns in the original data order.
    column_stats : Dict[str, Dict[str, float]]
        Dictionary containing statistics like mean, min, and max for each column.
    real_data : pd.DataFrame
        Original data for metadata purposes.
    """

    def __init__(self, categorical_columns=None, mixed_columns=None, integer_columns=None):
        """
        Initializes the DataTransformer with optional lists for categorical, mixed, and integer columns.

        Parameters:
        -----------
        categorical_columns : List[str], optional
            List of categorical columns to one-hot encode.
        mixed_columns : Dict[str, str], optional
            Dictionary specifying constraints on mixed columns (e.g., 'positive', 'negative').
        integer_columns : List[str], optional
            List of integer columns for rounding during inverse transformation.
        """
        self.categorical_columns = categorical_columns or []
        self.mixed_columns = mixed_columns or {}
        self.integer_columns = integer_columns or []
        self.scalers = {}
        self.encoders = {}
        self.column_order = []
        self.column_stats = {}
        self.real_data = None

    def fit(self, data: pd.DataFrame):
        """
        Fits the transformers (scalers and encoders) to the provided data.

        Parameters:
        -----------
        data : pd.DataFrame
            The data to fit the transformers on.
        """
        self.column_order = data.columns.tolist()
        self.real_data = data
        for col in self.column_order:
            if col in self.categorical_columns:
                encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                self.encoders[col] = encoder.fit(data[[col]])
            elif col in self.mixed_columns or col in self.integer_columns or col not in self.categorical_columns:
                scaler = StandardScaler()
                self.scalers[col] = scaler.fit(data[[col]])
                self.column_stats[col] = {
                    'mean': data[col].mean(),
                    'min': data[col].min(),
                    'max': data[col].max()
                }

    def transform(self, data: pd.DataFrame) -> np.ndarray:
        """
        Transforms the provided data using the fitted transformers.

        Parameters:
        -----------
        data : pd.DataFrame
            The data to transform.

        Returns:
        --------
        np.ndarray:
            Transformed data in a NumPy array format.
        """
        transformed_data = []
        for col in self.column_order:
            if col in self.categorical_columns:
                transformed_data.append(self.encoders[col].transform(data[[col]]))
            elif col in self.mixed_columns or col in self.integer_columns or col not in self.categorical_columns:
                transformed_data.append(self.scalers[col].transform(data[[col]]))
        return np.hstack(transformed_data)

    def inverse_transform(self, data: np.ndarray) -> pd.DataFrame:
        inverted_data = {}
        start = 0
        for col in self.column_order:
            if col in self.categorical_columns:
                col_width = len(self.encoders[col].categories_[0])
                col_data = data[:, start:start + col_width]

                # For binary categorical columns (like 'income')
                if col_width == 2:
                    # Use a threshold of 0.5 to determine the class
                    inverted = (col_data[:, 1] > 0.5).astype(int)
                else:
                    # For non-binary categorical columns, use argmax as before
                    inverted = self.encoders[col].inverse_transform(col_data)

                inverted_data[col] = inverted.flatten()
                start += col_width
            elif col in self.mixed_columns or col in self.integer_columns or col not in self.categorical_columns:
                col_data = data[:, start:start + 1]
                inverted_data[col] = self.scalers[col].inverse_transform(col_data).flatten()

                # Apply constraints based on mixed_columns specification
                if col in self.mixed_columns:
                    constraint = self.mixed_columns[col]
                    if constraint == 'positive':
                        inverted_data[col] = np.clip(inverted_data[col], 0, None)
                    elif constraint == 'negative':
                        inverted_data[col] = np.clip(inverted_data[col], None, 0)
                    elif constraint == 'mean':
                        mean = self.column_stats[col]['mean']
                        inverted_data[col] = np.clip(inverted_data[col], mean, None)
                    elif constraint == 'zero_mean':
                        inverted_data[col] -= self.column_stats[col]['mean']
                    elif isinstance(constraint, tuple) and len(constraint) == 2:
                        min_val, max_val = constraint
                        inverted_data[col] = np.clip(inverted_data[col], min_val, max_val)

                # Handle integer columns
                if col in self.integer_columns:
                    inverted_data[col] = np.round(inverted_data[col]).astype(int)

                start += 1

        # Return a DataFrame with properly transformed data
        return pd.DataFrame(inverted_data)
