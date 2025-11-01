"""Tests for EDA domain analyzer module."""

import pandas as pd
import pytest

from skyulf_mlflow_library.eda import DomainAnalyzer, DomainInferenceResult, infer_domain


def test_domain_analyzer_initialization():
    """Test DomainAnalyzer initialization."""
    analyzer = DomainAnalyzer(enable_ml_classifier=False)
    assert analyzer is not None
    assert analyzer.enable_ml_classifier is False


def test_domain_analyzer_retail_dataset():
    """Test domain detection for retail data."""
    df = pd.DataFrame({
        "order_id": ["A-100", "A-101", "A-102"],
        "customer_id": ["C001", "C002", "C003"],
        "price": [99.99, 149.99, 79.99],
        "quantity": [1, 2, 1],
        "category": ["Electronics", "Clothing", "Books"],
    })
    
    analyzer = DomainAnalyzer(enable_ml_classifier=False)
    result = analyzer.analyze_dataset_domain(df.columns, df)
    
    assert isinstance(result, DomainInferenceResult)
    assert result.primary_domain in ["retail", "finance", "general"]
    assert 0 <= result.primary_confidence <= 1
    assert isinstance(result.recommendations, list)
    assert len(result.recommendations) > 0


def test_domain_analyzer_marketing_dataset():
    """Test domain detection for marketing data."""
    df = pd.DataFrame({
        "campaign_id": ["C1", "C2", "C3"],
        "impressions": [10000, 15000, 12000],
        "clicks": [500, 750, 600],
        "conversions": [50, 75, 60],
        "spend": [1000.0, 1500.0, 1200.0],
        "channel": ["Email", "Social", "Search"],
    })
    
    analyzer = DomainAnalyzer(enable_ml_classifier=False)
    result = analyzer.analyze_dataset_domain(df.columns, df)
    
    assert isinstance(result, DomainInferenceResult)
    assert result.primary_domain == "marketing"
    assert result.primary_confidence > 0
    assert any("campaign" in rec.lower() or "marketing" in rec.lower() 
               for rec in result.recommendations)


def test_domain_analyzer_time_series_dataset():
    """Test domain detection for time series data."""
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=10, freq="D"),
        "value": [100, 105, 103, 108, 110, 115, 112, 118, 120, 125],
        "metric": ["sales"] * 10,
    })
    
    analyzer = DomainAnalyzer(enable_ml_classifier=False)
    result = analyzer.analyze_dataset_domain(df.columns, df)
    
    assert isinstance(result, DomainInferenceResult)
    assert "has_time_series" in str(result.patterns)


def test_domain_analyzer_healthcare_dataset():
    """Test domain detection for healthcare data."""
    df = pd.DataFrame({
        "patient_id": ["P001", "P002", "P003"],
        "diagnosis": ["Flu", "Cold", "Allergy"],
        "treatment": ["Medication A", "Rest", "Medication B"],
        "age": [45, 32, 28],
    })
    
    analyzer = DomainAnalyzer(enable_ml_classifier=False)
    result = analyzer.analyze_dataset_domain(df.columns, df)
    
    assert isinstance(result, DomainInferenceResult)
    assert result.primary_domain == "healthcare"


def test_domain_analyzer_fraud_detection():
    """Test domain detection for fraud data."""
    df = pd.DataFrame({
        "transaction_id": ["T001", "T002", "T003", "T004"],
        "amount": [100.0, 5000.0, 50.0, 10000.0],
        "fraud": [0, 1, 0, 1],
        "risk_score": [0.1, 0.9, 0.2, 0.95],
        "flag": ["normal", "suspicious", "normal", "alert"],
    })
    
    analyzer = DomainAnalyzer(enable_ml_classifier=False)
    result = analyzer.analyze_dataset_domain(df.columns, df)
    
    assert isinstance(result, DomainInferenceResult)
    assert result.primary_domain == "fraud"


def test_domain_analyzer_geospatial_dataset():
    """Test domain detection for geospatial data."""
    df = pd.DataFrame({
        "location_id": ["L1", "L2", "L3"],
        "latitude": [40.7128, 34.0522, 41.8781],
        "longitude": [-74.0060, -118.2437, -87.6298],
        "city": ["New York", "Los Angeles", "Chicago"],
        "country": ["USA", "USA", "USA"],
    })
    
    analyzer = DomainAnalyzer(enable_ml_classifier=False)
    result = analyzer.analyze_dataset_domain(df.columns, df)
    
    assert isinstance(result, DomainInferenceResult)
    assert result.primary_domain == "geospatial"
    assert result.patterns["data_patterns"]["has_geospatial_coordinates"]


def test_domain_analyzer_computer_vision():
    """Test domain detection for computer vision data."""
    df = pd.DataFrame({
        "image_id": ["IMG001", "IMG002", "IMG003"],
        "image_path": ["/path/img1.jpg", "/path/img2.png", "/path/img3.jpg"],
        "bbox_x_min": [10, 20, 15],
        "bbox_y_min": [10, 25, 20],
        "bbox_x_max": [100, 120, 110],
        "bbox_y_max": [100, 125, 115],
        "label": ["cat", "dog", "bird"],
    })
    
    analyzer = DomainAnalyzer(enable_ml_classifier=False)
    result = analyzer.analyze_dataset_domain(df.columns, df)
    
    assert isinstance(result, DomainInferenceResult)
    assert result.primary_domain == "computer_vision"
    assert result.patterns["data_patterns"]["has_object_detection_annotations"]


def test_domain_analyzer_empty_dataframe():
    """Test domain analyzer with empty DataFrame."""
    df = pd.DataFrame()
    
    analyzer = DomainAnalyzer(enable_ml_classifier=False)
    result = analyzer.analyze_dataset_domain([], df)
    
    assert isinstance(result, DomainInferenceResult)
    assert result.primary_domain == "general"


def test_domain_analyzer_to_dict():
    """Test DomainInferenceResult serialization."""
    df = pd.DataFrame({
        "price": [10.0, 20.0, 30.0],
        "category": ["A", "B", "C"],
    })
    
    analyzer = DomainAnalyzer(enable_ml_classifier=False)
    result = analyzer.analyze_dataset_domain(df.columns, df)
    
    result_dict = result.to_dict()
    
    assert isinstance(result_dict, dict)
    assert "primary_domain" in result_dict
    assert "primary_confidence" in result_dict
    assert "domain_scores" in result_dict
    assert "recommendations" in result_dict


def test_infer_domain_helper():
    """Test infer_domain convenience function."""
    df = pd.DataFrame({
        "campaign": ["C1", "C2", "C3"],
        "clicks": [100, 200, 150],
        "conversions": [10, 20, 15],
    })
    
    result = infer_domain(df)
    
    assert isinstance(result, DomainInferenceResult)
    assert result.primary_domain is not None
    assert result.primary_confidence >= 0


def test_domain_analyzer_secondary_domains():
    """Test that secondary domains are detected."""
    df = pd.DataFrame({
        "customer_id": ["C1", "C2", "C3"],
        "price": [99.99, 149.99, 79.99],
        "campaign": ["Summer", "Winter", "Fall"],
        "impressions": [1000, 2000, 1500],
    })
    
    analyzer = DomainAnalyzer(enable_ml_classifier=False)
    result = analyzer.analyze_dataset_domain(df.columns, df)
    
    assert isinstance(result.secondary_domains, list)
    # May or may not have secondary domains depending on scoring


def test_domain_analyzer_nlp_dataset():
    """Test domain detection for NLP data."""
    df = pd.DataFrame({
        "text": [
            "This is a long piece of text that represents a document",
            "Another document with different content and structure",
            "Third document in our corpus with various words",
        ],
        "label": ["positive", "negative", "neutral"],
    })
    
    analyzer = DomainAnalyzer(enable_ml_classifier=False)
    result = analyzer.analyze_dataset_domain(df.columns, df)
    
    assert isinstance(result, DomainInferenceResult)
    assert result.patterns["data_patterns"]["has_text_columns"]


def test_domain_analyzer_environmental_dataset():
    """Test domain detection for environmental data."""
    df = pd.DataFrame({
        "sensor_id": ["S1", "S2", "S3"],
        "temperature": [22.5, 23.1, 21.8],
        "humidity": [65.0, 68.5, 62.3],
        "air_quality": [45, 52, 48],
        "timestamp": pd.date_range("2024-01-01", periods=3, freq="H"),
    })
    
    analyzer = DomainAnalyzer(enable_ml_classifier=False)
    result = analyzer.analyze_dataset_domain(df.columns, df)
    
    assert isinstance(result, DomainInferenceResult)
    assert result.primary_domain == "environmental"


def test_domain_analyzer_recommendations_length():
    """Test that recommendations are capped appropriately."""
    df = pd.DataFrame({
        "col1": [1, 2, 3],
        "col2": [4, 5, 6],
    })
    
    analyzer = DomainAnalyzer(enable_ml_classifier=False)
    result = analyzer.analyze_dataset_domain(df.columns, df)
    
    # Recommendations should be capped at 6
    assert len(result.recommendations) <= 6
