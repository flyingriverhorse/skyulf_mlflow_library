"""Tests for feature encoding module."""

import pandas as pd
import numpy as np
import pytest

from skyulf_mlflow_library.features.encoding import (
    OneHotEncoder,
    LabelEncoder,
    OrdinalEncoder,
)


def test_onehot_encoder_basic():
    """Test OneHotEncoder basic functionality."""
    df = pd.DataFrame({
        "city": ["NYC", "LA", "SF", "NYC"],
        "value": [100, 200, 300, 400],
    })
    
    encoder = OneHotEncoder(columns=["city"])
    df_encoded = encoder.fit_transform(df)
    
    assert "city_LA" in df_encoded.columns
    assert "city_NYC" in df_encoded.columns
    assert "city_SF" in df_encoded.columns
    assert df_encoded.shape[1] > df.shape[1]


def test_onehot_encoder_multiple_columns():
    """Test OneHotEncoder with multiple columns."""
    df = pd.DataFrame({
        "city": ["NYC", "LA"],
        "category": ["A", "B"],
        "value": [100, 200],
    })
    
    encoder = OneHotEncoder(columns=["city", "category"])
    df_encoded = encoder.fit_transform(df)
    
    assert "city_NYC" in df_encoded.columns
    assert "category_A" in df_encoded.columns


def test_onehot_encoder_drop_original():
    """Test that original columns are dropped."""
    df = pd.DataFrame({
        "city": ["NYC", "LA", "SF"],
        "value": [100, 200, 300],
    })
    
    encoder = OneHotEncoder(columns=["city"])
    df_encoded = encoder.fit_transform(df)
    
    assert "city" not in df_encoded.columns
    assert "value" in df_encoded.columns


def test_label_encoder_basic():
    """Test LabelEncoder basic functionality."""
    df = pd.DataFrame({
        "category": ["low", "medium", "high", "low"],
        "value": [1, 2, 3, 4],
    })
    
    encoder = LabelEncoder(columns=["category"])
    df_encoded = encoder.fit_transform(df)
    
    # LabelEncoder creates new column with _encoded suffix
    assert "category_encoded" in df_encoded.columns
    assert df_encoded["category_encoded"].dtype in [np.int64, np.int32]
    assert len(df_encoded["category_encoded"].unique()) == 3


def test_label_encoder_transform():
    """Test LabelEncoder transform on new data."""
    df_train = pd.DataFrame({
        "category": ["A", "B", "C"],
    })
    df_test = pd.DataFrame({
        "category": ["B", "A", "C"],
    })
    
    encoder = LabelEncoder(columns=["category"])
    encoder.fit(df_train)
    df_test_encoded = encoder.transform(df_test)
    
    # New column created with _encoded suffix
    assert "category_encoded" in df_test_encoded.columns
    assert df_test_encoded["category_encoded"].dtype in [np.int64, np.int32]


def test_ordinal_encoder_basic():
    """Test OrdinalEncoder with explicit ordering."""
    df = pd.DataFrame({
        "size": ["small", "large", "medium", "small"],
    })
    
    encoder = OrdinalEncoder(
        columns=["size"],
        categories={"size": ["small", "medium", "large"]}
    )
    df_encoded = encoder.fit_transform(df)
    
    assert df_encoded["size"].dtype in [np.int64, np.int32, np.float64]
    # small=0, medium=1, large=2
    assert df_encoded["size"].iloc[0] == 0
    assert df_encoded["size"].iloc[1] == 2
    assert df_encoded["size"].iloc[2] == 1


def test_ordinal_encoder_multiple_columns():
    """Test OrdinalEncoder with multiple columns."""
    df = pd.DataFrame({
        "size": ["S", "M", "L"],
        "quality": ["low", "medium", "high"],
    })
    
    encoder = OrdinalEncoder(
        columns=["size", "quality"],
        categories={
            "size": ["S", "M", "L"],
            "quality": ["low", "medium", "high"],
        }
    )
    df_encoded = encoder.fit_transform(df)
    
    assert df_encoded.shape == df.shape
    assert df_encoded["size"].tolist() == [0, 1, 2]


def test_encoder_empty_dataframe():
    """Test encoder with empty DataFrame."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    
    # No categorical columns to encode
    encoder = OneHotEncoder(columns=None)
    # Should raise error when no columns to encode
    with pytest.raises(Exception):  # EncodingError
        df_encoded = encoder.fit_transform(df)


def test_onehot_encoder_sparse():
    """Test OneHotEncoder with sparse output."""
    df = pd.DataFrame({
        "category": ["A", "B", "C", "A"],
    })
    
    encoder = OneHotEncoder(columns=["category"], sparse=False)
    df_encoded = encoder.fit_transform(df)
    
    assert isinstance(df_encoded, pd.DataFrame)


def test_label_encoder_unseen_category():
    """Test LabelEncoder with unseen categories in transform."""
    df_train = pd.DataFrame({
        "category": ["A", "B", "C"],
    })
    df_test = pd.DataFrame({
        "category": ["A", "D"],  # D is unseen
    })
    
    encoder = LabelEncoder(columns=["category"], handle_unknown="use_encoded_value", unknown_value=-1)
    encoder.fit(df_train)
    
    # Should handle unseen categories gracefully
    df_test_encoded = encoder.transform(df_test)
    assert "category_encoded" in df_test_encoded.columns
    # Check that unseen category got the unknown value
    assert (df_test_encoded["category_encoded"] == -1).any()


def test_onehot_get_feature_names():
    """Test getting feature names from OneHotEncoder."""
    df = pd.DataFrame({
        "city": ["NYC", "LA", "SF"],
    })
    
    encoder = OneHotEncoder(columns=["city"])
    encoder.fit_transform(df)
    
    feature_names = encoder.get_feature_names_out()
    assert len(feature_names) >= 3
    assert any("NYC" in name for name in feature_names)


def test_encoder_mixed_types():
    """Test encoder with mixed data types."""
    df = pd.DataFrame({
        "category": ["A", "B", "C"],
        "numeric": [1, 2, 3],
        "text": ["x", "y", "z"],
    })
    
    encoder = OneHotEncoder(columns=["category"])
    df_encoded = encoder.fit_transform(df)
    
    assert "numeric" in df_encoded.columns
    assert "text" in df_encoded.columns
