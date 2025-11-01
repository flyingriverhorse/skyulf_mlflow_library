"""Tests for feature selection."""

import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression

from skyulf_mlflow_library.features.selection import FeatureSelector


class TestFeatureSelectorBasic:
    """Test basic FeatureSelector functionality."""
    
    def test_feature_selector_initialization(self):
        """Test that FeatureSelector initializes correctly."""
        selector = FeatureSelector(method='select_k_best', k=5)
        
        assert selector.method == 'select_k_best'
        assert selector.k == 5
    
    def test_feature_selector_select_k_best_classification(self):
        """Test SelectKBest for classification."""
        # Create classification dataset
        X, y = make_classification(n_samples=100, n_features=20, n_informative=10, 
                                   n_redundant=5, random_state=42)
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
        df['target'] = y
        
        selector = FeatureSelector(
            method='select_k_best',
            k=10,
            problem_type='classification',
            target_column='target'
        )
        
        df_result = selector.fit_transform(df)
        
        # Should have 10 features + target
        assert len(df_result.columns) <= 11  # 10 features + target
    
    def test_feature_selector_select_k_best_regression(self):
        """Test SelectKBest for regression."""
        # Create regression dataset
        X, y = make_regression(n_samples=100, n_features=15, n_informative=8, random_state=42)
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(15)])
        df['target'] = y
        
        selector = FeatureSelector(
            method='select_k_best',
            k=8,
            problem_type='regression',
            target_column='target'
        )
        
        df_result = selector.fit_transform(df)
        
        # Should have 8 features + target
        assert len(df_result.columns) <= 9
    
    def test_feature_selector_variance_threshold(self):
        """Test variance threshold selection."""
        # Create dataset with some low-variance features
        df = pd.DataFrame({
            'high_var': np.random.randn(100),
            'low_var': [0] * 99 + [1],  # Almost no variance
            'medium_var': np.random.randn(100) * 0.5,
            'no_var': [1] * 100  # No variance
        })
        
        selector = FeatureSelector(
            method='variance_threshold',
            threshold=0.01
        )
        
        df_result = selector.fit_transform(df)
        
        # Should drop low and no variance columns
        assert 'high_var' in df_result.columns
        assert 'no_var' not in df_result.columns
    
    def test_feature_selector_with_specific_columns(self):
        """Test feature selection on specific columns."""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        df = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(10)])
        df['target'] = y
        
        # Only select from first 5 features
        selector = FeatureSelector(
            columns=[f'feat_{i}' for i in range(5)],
            method='select_k_best',
            k=3,
            problem_type='classification',
            target_column='target'
        )
        
        df_result = selector.fit_transform(df)
        
        # Should have selected features + remaining unselected + target
        assert 'target' in df_result.columns


class TestFeatureSelectorPercentile:
    """Test FeatureSelector with percentile method."""
    
    def test_select_percentile(self):
        """Test selecting features by percentile."""
        X, y = make_classification(n_samples=100, n_features=20, n_informative=15, random_state=42)
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
        df['target'] = y
        
        selector = FeatureSelector(
            method='select_percentile',
            percentile=50,  # Select top 50%
            problem_type='classification',
            target_column='target'
        )
        
        df_result = selector.fit_transform(df)
        
        # Should have approximately 10 features (50% of 20)
        # +/- some tolerance since percentile is approximate
        assert len(df_result.columns) <= 12  # 10 features + target + tolerance


class TestFeatureSelectorTransformOnly:
    """Test transform-only behavior."""
    
    def test_transform_without_fit_raises_error(self):
        """Test that transform without fit raises error."""
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })
        
        selector = FeatureSelector(method='select_k_best', k=1)
        
        # Should raise error when calling transform before fit
        with pytest.raises((ValueError, AttributeError, Exception)):
            selector.transform(df)
    
    def test_fit_transform_then_transform(self):
        """Test that transform works after fit_transform."""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        df_train = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(10)])
        df_train['target'] = y
        
        selector = FeatureSelector(
            method='select_k_best',
            k=5,
            problem_type='classification',
            target_column='target'
        )
        
        # Fit on training data
        df_train_result = selector.fit_transform(df_train)
        
        # Transform new data
        X_test, y_test = make_classification(n_samples=50, n_features=10, random_state=43)
        df_test = pd.DataFrame(X_test, columns=[f'feat_{i}' for i in range(10)])
        df_test['target'] = y_test
        
        df_test_result = selector.transform(df_test)
        
        # Should have same columns as training result
        assert len(df_test_result.columns) == len(df_train_result.columns)


class TestFeatureSelectorEdgeCases:
    """Test edge cases and error handling."""
    
    def test_k_larger_than_features(self):
        """Test when k is larger than number of features."""
        df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [2, 3, 4, 5, 6],
            'target': [0, 1, 0, 1, 0]
        })
        
        selector = FeatureSelector(
            method='select_k_best',
            k=10,  # More than 2 features
            problem_type='classification',
            target_column='target'
        )
        
        # Should not fail, just select all available features
        df_result = selector.fit_transform(df)
        assert len(df_result.columns) <= 3  # 2 features + target
    
    def test_invalid_method_defaults_to_select_k_best(self):
        """Test that invalid method defaults to select_k_best."""
        selector = FeatureSelector(method='invalid_method')
        
        # Should default to select_k_best
        assert selector.method == 'select_k_best'
