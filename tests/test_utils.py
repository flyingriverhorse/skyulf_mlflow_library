"""Tests for utils module."""

import pandas as pd
import numpy as np
import pytest

from skyulf_mlflow_library.utils import (
    train_test_split,
    train_val_test_split,
)


def test_train_test_split_basic():
    """Test basic train_test_split."""
    df = pd.DataFrame({
        "feature1": range(100),
        "feature2": range(100, 200),
        "target": range(200, 300),
    })
    
    X = df[["feature1", "feature2"]]
    y = df["target"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    assert len(X_train) == 80
    assert len(X_test) == 20
    assert len(y_train) == 80
    assert len(y_test) == 20


def test_train_test_split_custom_ratio():
    """Test train_test_split with custom ratio."""
    df = pd.DataFrame({
        "feature": range(100),
        "target": range(100),
    })
    
    X = df[["feature"]]
    y = df["target"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    assert len(X_train) == 70
    assert len(X_test) == 30


def test_train_test_split_stratify():
    """Test train_test_split with stratification."""
    df = pd.DataFrame({
        "feature": range(100),
        "target": [0] * 50 + [1] * 50,
    })
    
    X = df[["feature"]]
    y = df["target"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y
    )
    
    # Check class distribution is preserved
    train_ratio = sum(y_train) / len(y_train)
    test_ratio = sum(y_test) / len(y_test)
    
    assert abs(train_ratio - 0.5) < 0.1
    assert abs(test_ratio - 0.5) < 0.1


def test_train_test_split_random_state():
    """Test train_test_split reproducibility with random_state."""
    df = pd.DataFrame({
        "feature": range(50),
        "target": range(50),
    })
    
    X = df[["feature"]]
    y = df["target"]
    
    X_train1, X_test1, _, _ = train_test_split(X, y, random_state=42)
    X_train2, X_test2, _, _ = train_test_split(X, y, random_state=42)
    
    pd.testing.assert_frame_equal(X_train1, X_train2)
    pd.testing.assert_frame_equal(X_test1, X_test2)


def test_train_val_test_split_basic():
    """Test basic train_val_test_split."""
    df = pd.DataFrame({
        "feature1": range(100),
        "feature2": range(100, 200),
        "target": range(200, 300),
    })
    
    X = df[["feature1", "feature2"]]
    y = df["target"]
    
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
        X, y, val_size=0.2, test_size=0.2
    )
    
    assert len(X_train) == 60
    assert len(X_val) == 20
    assert len(X_test) == 20


def test_train_val_test_split_custom_ratios():
    """Test train_val_test_split with custom ratios."""
    df = pd.DataFrame({
        "feature": range(100),
        "target": range(100),
    })
    
    X = df[["feature"]]
    y = df["target"]
    
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
        X, y, val_size=0.15, test_size=0.15
    )
    
    # Due to rounding, sizes might vary by 1
    assert len(X_train) >= 69 and len(X_train) <= 70
    assert len(X_val) >= 15 and len(X_val) <= 16
    assert len(X_test) >= 14 and len(X_test) <= 15
    assert len(X_train) + len(X_val) + len(X_test) == 100


def test_train_val_test_split_stratify():
    """Test train_val_test_split with stratification."""
    df = pd.DataFrame({
        "feature": range(120),
        "target": [0] * 60 + [1] * 60,
    })
    
    X = df[["feature"]]
    y = df["target"]
    
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
        X, y, val_size=0.2, test_size=0.2, stratify=y
    )
    
    # Check class distribution is preserved
    train_ratio = sum(y_train) / len(y_train)
    val_ratio = sum(y_val) / len(y_val)
    test_ratio = sum(y_test) / len(y_test)
    
    assert abs(train_ratio - 0.5) < 0.1
    assert abs(val_ratio - 0.5) < 0.1
    assert abs(test_ratio - 0.5) < 0.1


def test_train_val_test_split_random_state():
    """Test train_val_test_split reproducibility."""
    df = pd.DataFrame({
        "feature": range(60),
        "target": range(60),
    })
    
    X = df[["feature"]]
    y = df["target"]
    
    result1 = train_val_test_split(X, y, random_state=42)
    result2 = train_val_test_split(X, y, random_state=42)
    
    pd.testing.assert_frame_equal(result1[0], result2[0])  # X_train
    pd.testing.assert_frame_equal(result1[1], result2[1])  # X_val
    pd.testing.assert_frame_equal(result1[2], result2[2])  # X_test


def test_train_test_split_no_shuffle():
    """Test train_test_split without shuffling."""
    df = pd.DataFrame({
        "feature": range(100),
        "target": range(100),
    })
    
    X = df[["feature"]]
    y = df["target"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    
    # First 80 samples should be in train, last 20 in test
    assert X_train.iloc[0]["feature"] == 0
    assert X_test.iloc[0]["feature"] == 80


def test_train_test_split_small_dataset():
    """Test train_test_split with small dataset."""
    df = pd.DataFrame({
        "feature": [1, 2, 3, 4, 5],
        "target": [1, 2, 3, 4, 5],
    })
    
    X = df[["feature"]]
    y = df["target"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    
    assert len(X_train) == 3
    assert len(X_test) == 2


def test_train_test_split_single_feature():
    """Test train_test_split with single feature."""
    X = pd.DataFrame({"feature": range(50)})
    y = pd.Series(range(50))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    assert len(X_train) == 40
    assert len(X_test) == 10


def test_train_val_test_split_equal_sizes():
    """Test train_val_test_split with equal validation and test sizes."""
    df = pd.DataFrame({
        "feature": range(90),
        "target": range(90),
    })
    
    X = df[["feature"]]
    y = df["target"]
    
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
        X, y, val_size=0.2, test_size=0.2
    )
    
    assert len(X_val) == len(X_test)
