"""Tests for preprocessing module."""

import pandas as pd
import numpy as np
import pytest

from skyulf_mlflow_library.preprocessing import (
    SimpleImputer,
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    remove_duplicates,
    drop_missing_rows,
    drop_missing_columns,
)


def test_simple_imputer_mean():
    """Test SimpleImputer with mean strategy."""
    df = pd.DataFrame({
        "age": [25, 30, None, 40, 35],
        "salary": [50000, 60000, 55000, None, 70000],
    })
    
    imputer = SimpleImputer(strategy="mean")
    df_imputed = imputer.fit_transform(df)
    
    assert df_imputed["age"].isna().sum() == 0
    assert df_imputed["salary"].isna().sum() == 0
    assert df_imputed["age"].iloc[2] == pytest.approx(32.5, rel=1e-2)


def test_simple_imputer_median():
    """Test SimpleImputer with median strategy."""
    df = pd.DataFrame({
        "value": [10, 20, None, 40, 50],
    })
    
    imputer = SimpleImputer(strategy="median")
    df_imputed = imputer.fit_transform(df)
    
    assert df_imputed["value"].iloc[2] == 30.0


def test_simple_imputer_most_frequent():
    """Test SimpleImputer with most_frequent strategy."""
    df = pd.DataFrame({
        "city": ["NYC", "LA", np.nan, "NYC", "NYC"],
    })
    
    imputer = SimpleImputer(strategy="most_frequent")
    df_imputed = imputer.fit_transform(df)
    
    assert df_imputed["city"].iloc[2] == "NYC"


def test_standard_scaler():
    """Test StandardScaler."""
    df = pd.DataFrame({
        "age": [20, 30, 40, 50, 60],
        "salary": [30000, 40000, 50000, 60000, 70000],
    })
    
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    
    # After scaling, mean should be ~0 and std should be ~1 (using ddof=0 like sklearn)
    assert df_scaled["age"].mean() == pytest.approx(0, abs=1e-10)
    assert df_scaled["age"].std(ddof=0) == pytest.approx(1, abs=1e-10)


def test_standard_scaler_columns():
    """Test StandardScaler with specific columns."""
    df = pd.DataFrame({
        "age": [20, 30, 40],
        "salary": [30000, 40000, 50000],
        "city": ["A", "B", "C"],
    })
    
    scaler = StandardScaler(columns=["age", "salary"])
    df_scaled = scaler.fit_transform(df)
    
    assert "city" in df_scaled.columns
    assert df_scaled["city"].tolist() == ["A", "B", "C"]


def test_minmax_scaler():
    """Test MinMaxScaler."""
    df = pd.DataFrame({
        "value": [10, 20, 30, 40, 50],
    })
    
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df)
    
    assert df_scaled["value"].min() == 0.0
    assert df_scaled["value"].max() == 1.0


def test_minmax_scaler_custom_range():
    """Test MinMaxScaler with custom feature range."""
    df = pd.DataFrame({
        "value": [0, 50, 100],
    })
    
    scaler = MinMaxScaler(feature_range=(-1, 1))
    df_scaled = scaler.fit_transform(df)
    
    assert df_scaled["value"].min() == -1.0
    assert df_scaled["value"].max() == 1.0


def test_robust_scaler():
    """Test RobustScaler."""
    df = pd.DataFrame({
        "value": [1, 2, 3, 4, 5, 100],  # 100 is an outlier
    })
    
    scaler = RobustScaler()
    df_scaled = scaler.fit_transform(df)
    
    # RobustScaler should be less affected by the outlier
    assert df_scaled["value"].isna().sum() == 0


def test_remove_duplicates():
    """Test remove_duplicates function."""
    df = pd.DataFrame({
        "name": ["Alice", "Bob", "Alice", "Charlie"],
        "age": [25, 30, 25, 35],
    })
    
    df_clean = remove_duplicates(df)
    
    assert len(df_clean) == 3
    assert "Alice" in df_clean["name"].values


def test_drop_missing_rows():
    """Test drop_missing_rows function."""
    df = pd.DataFrame({
        "a": [1, 2, None, 4],
        "b": [5, None, 7, 8],
    })
    
    df_clean = drop_missing_rows(df)
    
    assert len(df_clean) == 2
    assert df_clean["a"].isna().sum() == 0


def test_drop_missing_columns():
    """Test drop_missing_columns function."""
    df = pd.DataFrame({
        "a": [1, 2, 3],
        "b": [None, None, None],
        "c": [4, 5, 6],
    })
    
    df_clean = drop_missing_columns(df, threshold=0.5)
    
    assert "b" not in df_clean.columns
    assert "a" in df_clean.columns
    assert "c" in df_clean.columns


def test_imputer_transform_only():
    """Test that transform works after fit."""
    df_train = pd.DataFrame({
        "age": [20, 30, None, 40],
    })
    df_test = pd.DataFrame({
        "age": [None, 35],
    })
    
    imputer = SimpleImputer(strategy="mean")
    imputer.fit(df_train)
    df_test_imputed = imputer.transform(df_test)
    
    # Should use mean from training data (30.0)
    assert df_test_imputed["age"].iloc[0] == 30.0


def test_scaler_transform_only():
    """Test that scaler transform works after fit."""
    df_train = pd.DataFrame({
        "value": [10, 20, 30],
    })
    df_test = pd.DataFrame({
        "value": [15, 25],
    })
    
    scaler = StandardScaler()
    scaler.fit(df_train)
    df_test_scaled = scaler.transform(df_test)
    
    assert df_test_scaled.shape == df_test.shape


def test_imputer_empty_dataframe():
    """Test imputer with empty DataFrame."""
    df = pd.DataFrame()
    
    imputer = SimpleImputer()
    
    # Should raise error for empty dataframe
    with pytest.raises(Exception):  # ImputationError
        df_result = imputer.fit_transform(df)


def test_scaler_empty_dataframe():
    """Test scaler with empty DataFrame."""
    df = pd.DataFrame()
    
    scaler = StandardScaler()
    
    # Should raise error for empty dataframe
    with pytest.raises(Exception):  # ScalingError
        df_scaled = scaler.fit_transform(df)
