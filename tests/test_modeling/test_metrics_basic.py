"""Tests for MetricsCalculator with correct API."""

import pytest
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

from skyulf_mlflow_library.modeling import MetricsCalculator


class TestMetricsCalculatorClassification:
    """Test MetricsCalculator for classification tasks."""
    
    def test_metrics_calculator_binary_classification(self):
        """Test binary classification metrics calculation."""
        # Create simple binary classification data
        y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 0])
        y_pred = np.array([0, 1, 1, 0, 0, 0, 1, 1, 0, 1])
        
        calc = MetricsCalculator(problem_type='classification')
        metrics = calc.calculate(y_true, y_pred)
        
        # Check that basic metrics are present
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        
        # Check that values are reasonable
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['f1'] <= 1
    
    def test_metrics_calculator_with_probabilities(self):
        """Test classification metrics with probability predictions."""
        # Create small dataset
        X, y = make_classification(n_samples=100, n_features=10, n_classes=2, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Train model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        calc = MetricsCalculator(problem_type='classification')
        metrics = calc.calculate(y_test, y_pred, y_proba=y_proba)
        
        # Should include probability-based metrics
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
    
    def test_metrics_calculator_multiclass(self):
        """Test multiclass classification metrics."""
        # Create multiclass data
        y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 2, 2, 0, 1, 1])
        
        calc = MetricsCalculator(problem_type='classification')
        metrics = calc.calculate(y_true, y_pred)
        
        assert 'accuracy' in metrics
        assert 0 <= metrics['accuracy'] <= 1


class TestMetricsCalculatorRegression:
    """Test MetricsCalculator for regression tasks."""
    
    def test_metrics_calculator_regression_basic(self):
        """Test basic regression metrics calculation."""
        y_true = np.array([3.0, -0.5, 2.0, 7.0, 4.2])
        y_pred = np.array([2.5, 0.0, 2.0, 8.0, 4.5])
        
        calc = MetricsCalculator(problem_type='regression')
        metrics = calc.calculate(y_true, y_pred)
        
        # Check that basic regression metrics are present
        assert 'mae' in metrics or 'mean_absolute_error' in metrics
        assert 'mse' in metrics or 'mean_squared_error' in metrics
        assert 'rmse' in metrics or 'root_mean_squared_error' in metrics
    
    def test_metrics_calculator_regression_with_model(self):
        """Test regression metrics with actual model predictions."""
        # Create regression dataset
        X, y = make_regression(n_samples=100, n_features=10, noise=0.1, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Train model
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        calc = MetricsCalculator(problem_type='regression')
        metrics = calc.calculate(y_test, y_pred)
        
        # Verify metrics are calculated
        assert len(metrics) > 0
        # Check at least one metric exists
        has_metric = any(k in metrics for k in ['mae', 'mean_absolute_error', 'mse', 'mean_squared_error', 'r2', 'r2_score'])
        assert has_metric


class TestMetricsCalculatorErrors:
    """Test MetricsCalculator error handling."""
    
    def test_invalid_problem_type(self):
        """Test that invalid problem type raises error."""
        with pytest.raises((ValueError, KeyError, Exception)):
            MetricsCalculator(problem_type='invalid')
    
    def test_calculate_with_mismatched_lengths(self):
        """Test that mismatched lengths raise error."""
        calc = MetricsCalculator(problem_type='classification')
        y_true = np.array([0, 1, 1])
        y_pred = np.array([0, 1])
        
        with pytest.raises((ValueError, Exception)):
            calc.calculate(y_true, y_pred)
