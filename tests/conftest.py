"""Pytest configuration and fixtures."""

import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'numeric1': [1, 2, 3, 4, 5],
        'numeric2': [10.0, 20.0, 30.0, 40.0, 50.0],
        'category1': ['A', 'B', 'A', 'C', 'B'],
        'category2': ['X', 'Y', 'X', 'Y', 'X'],
        'target': [0, 1, 0, 1, 0]
    })


@pytest.fixture
def sample_df_with_missing():
    """Create a sample DataFrame with missing values for testing."""
    return pd.DataFrame({
        'numeric1': [1, 2, np.nan, 4, 5],
        'numeric2': [10.0, np.nan, 30.0, 40.0, 50.0],
        'category1': ['A', 'B', None, 'C', 'B'],
        'target': [0, 1, 0, 1, 0]
    })


@pytest.fixture
def sample_df_with_outliers():
    """Create a sample DataFrame with outliers for testing."""
    return pd.DataFrame({
        'numeric1': [1, 2, 3, 4, 100],  # 100 is outlier
        'numeric2': [10.0, 20.0, 30.0, 40.0, 50.0],
        'category': ['A', 'B', 'A', 'C', 'B']
    })
