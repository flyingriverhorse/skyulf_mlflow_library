"""Tests for feature transformation module."""

import pandas as pd
import numpy as np
import pytest

from skyulf_mlflow_library.features.transform import FeatureMath, PolynomialFeatures


def test_feature_math_basic():
    """Test basic FeatureMath operation."""
    df = pd.DataFrame({
        "a": [1, 2, 3],
        "b": [4, 5, 6],
    })
    
    operations = [
        {"type": "arithmetic", "method": "add", "columns": ["a", "b"], "output": "sum"},
    ]
    
    transformer = FeatureMath(operations=operations)
    df_transformed = transformer.fit_transform(df)
    
    assert "sum" in df_transformed.columns
    assert df_transformed["sum"].tolist() == [5, 7, 9]


def test_feature_math_multiple_operations():
    """Test FeatureMath with multiple operations."""
    df = pd.DataFrame({
        "x": [10, 20, 30],
        "y": [2, 4, 6],
    })
    
    operations = [
        {"type": "arithmetic", "method": "multiply", "columns": ["x", "y"], "output": "product"},
        {"type": "arithmetic", "method": "divide", "columns": ["x", "y"], "output": "ratio"},
    ]
    
    transformer = FeatureMath(operations=operations)
    df_transformed = transformer.fit_transform(df)
    
    assert "product" in df_transformed.columns
    assert "ratio" in df_transformed.columns
    assert df_transformed["product"].tolist() == [20, 80, 180]
    assert df_transformed["ratio"].tolist() == [5.0, 5.0, 5.0]


def test_feature_math_subtract():
    """Test FeatureMath subtraction."""
    df = pd.DataFrame({
        "a": [10, 20, 30],
        "b": [3, 5, 7],
    })
    
    operations = [
        {"type": "arithmetic", "method": "subtract", "columns": ["a", "b"], "output": "diff"},
    ]
    
    transformer = FeatureMath(operations=operations)
    df_transformed = transformer.fit_transform(df)
    
    assert df_transformed["diff"].tolist() == [7, 15, 23]


def test_polynomial_features_degree_2():
    """Test PolynomialFeatures with degree 2."""
    df = pd.DataFrame({
        "x": [1, 2, 3],
        "y": [4, 5, 6],
    })
    
    poly = PolynomialFeatures(degree=2, include_bias=False)
    df_poly = poly.fit_transform(df)
    
    # Should have x, y, x^2, xy, y^2
    assert df_poly.shape[1] >= 5


def test_polynomial_features_degree_3():
    """Test PolynomialFeatures with degree 3."""
    df = pd.DataFrame({
        "x": [1, 2],
    })
    
    poly = PolynomialFeatures(degree=3, include_bias=False)
    df_poly = poly.fit_transform(df)
    
    # Should have x, x^2, x^3
    assert df_poly.shape[1] >= 3


def test_polynomial_features_with_bias():
    """Test PolynomialFeatures with bias term."""
    df = pd.DataFrame({
        "x": [1, 2, 3],
    })
    
    poly = PolynomialFeatures(degree=2, include_bias=True)
    df_poly = poly.fit_transform(df)
    
    # Should have bias (1), x, x^2
    assert df_poly.shape[1] >= 3


def test_polynomial_features_interaction_only():
    """Test PolynomialFeatures with interaction_only."""
    df = pd.DataFrame({
        "a": [1, 2],
        "b": [3, 4],
    })
    
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    df_poly = poly.fit_transform(df)
    
    # Should have a, b, a*b (no a^2 or b^2)
    assert df_poly.shape[1] >= 3


def test_feature_math_empty_operations():
    """Test FeatureMath with empty operations list."""
    df = pd.DataFrame({
        "a": [1, 2, 3],
    })
    
    transformer = FeatureMath(operations=[])
    
    # Should raise error for empty operations
    with pytest.raises(Exception):  # FeatureEngineeringError
        df_transformed = transformer.fit_transform(df)


def test_feature_math_division_by_zero():
    """Test FeatureMath handles division by zero."""
    df = pd.DataFrame({
        "a": [10, 20, 30],
        "b": [2, 0, 5],
    })
    
    operations = [
        {"type": "arithmetic", "method": "divide", "columns": ["a", "b"], "output": "ratio"},
    ]
    
    transformer = FeatureMath(operations=operations)
    df_transformed = transformer.fit_transform(df)
    
    # Should handle zero division (inf or nan)
    assert "ratio" in df_transformed.columns


def test_polynomial_features_single_column():
    """Test PolynomialFeatures with single column."""
    df = pd.DataFrame({
        "x": [2, 3, 4],
    })
    
    poly = PolynomialFeatures(degree=2, include_bias=False)
    df_poly = poly.fit_transform(df)
    
    assert df_poly.shape[0] == 3
    assert df_poly.shape[1] >= 2  # x and x^2


def test_feature_math_transform_only():
    """Test FeatureMath transform on new data."""
    df_train = pd.DataFrame({
        "a": [1, 2, 3],
        "b": [4, 5, 6],
    })
    df_test = pd.DataFrame({
        "a": [7, 8],
        "b": [9, 10],
    })
    
    operations = [
        {"type": "arithmetic", "method": "add", "columns": ["a", "b"], "output": "sum"},
    ]
    
    transformer = FeatureMath(operations=operations)
    transformer.fit(df_train)
    df_test_transformed = transformer.transform(df_test)
    
    assert "sum" in df_test_transformed.columns
    assert df_test_transformed["sum"].tolist() == [16, 18]


def test_polynomial_features_get_feature_names():
    """Test getting feature names from PolynomialFeatures."""
    df = pd.DataFrame({
        "x": [1, 2],
        "y": [3, 4],
    })
    
    poly = PolynomialFeatures(degree=2, include_bias=False)
    poly.fit_transform(df)
    
    feature_names = poly.get_feature_names_out()
    assert len(feature_names) >= 5  # x, y, x^2, xy, y^2
