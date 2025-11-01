"""Tests for EDA text insights module."""

import pandas as pd
import pytest

from skyulf_mlflow_library.eda import get_text_insights


def test_get_text_insights_basic():
    """Test basic text insights extraction."""
    df = pd.DataFrame({
        "review": [
            "Great product, highly recommend",
            "Not satisfied with quality",
            "Amazing service and fast delivery",
            "Could be better",
            "Excellent value for money",
        ]
    })
    
    result = get_text_insights(df["review"], "review")
    
    assert "avg_text_length" in result
    assert result["avg_text_length"] > 0
    assert "text_category" in result
    assert "eda_recommendations" in result


def test_get_text_insights_mixed_content():
    """Test text insights with mixed content types."""
    df = pd.DataFrame({
        "feedback": [
            "Short",
            "This is a much longer feedback comment with more details",
            "Medium length text here",
            "Another short one",
            "Final comment with reasonable length",
        ]
    })
    
    result = get_text_insights(df["feedback"], "feedback")
    
    assert "min_text_length" in result
    assert result["min_text_length"] > 0
    assert "max_text_length" in result
    assert result["max_text_length"] > result["min_text_length"]
    assert result["min_text_length"] == 5  # "Short"
    assert result["max_text_length"] == 56  # longest string


def test_get_text_insights_empty_dataframe():
    """Test behavior with empty DataFrame."""
    df = pd.DataFrame({"text": []})
    
    result = get_text_insights(df["text"], "text")
    
    assert "avg_text_length" in result
    assert "text_category" in result


def test_get_text_insights_numeric_column():
    """Test text insights on numeric column converted to string."""
    df = pd.DataFrame({
        "numbers": [1, 2, 3, 4, 5],
    })
    
    result = get_text_insights(df["numbers"].astype(str), "numbers")
    
    assert "avg_text_length" in result
    assert "text_category" in result
    assert result["avg_text_length"] == 1.0  # Single digit numbers


def test_get_text_insights_special_characters():
    """Test text with special characters and punctuation."""
    df = pd.DataFrame({
        "text": [
            "Hello! How are you?",
            "Email: test@example.com",
            "Price: $99.99 (discount applied)",
            "Check #123 - approved",
            "Rating: ★★★★☆",
        ]
    })
    
    result = get_text_insights(df["text"], "text")
    
    assert "contains_punctuation_pct" in result
    assert result["contains_punctuation_pct"] > 0
    assert "text_category" in result
    assert "patterns" in result


def test_get_text_insights_duplicates():
    """Test text insights with duplicate values."""
    df = pd.DataFrame({
        "status": [
            "Completed",
            "Pending",
            "Completed",
            "Completed",
            "Failed",
            "Pending",
        ]
    })
    
    result = get_text_insights(df["status"], "status")
    
    assert "avg_text_length" in result
    assert "text_category" in result
    # text_category might be "categorical" due to duplicates
    assert result["text_category"] in ["categorical", "semi_structured", "identifier"]


def test_get_text_insights_null_handling():
    """Test text insights with null values."""
    df = pd.DataFrame({
        "notes": [
            "First note",
            None,
            "Third note",
            None,
            "Fifth note",
        ]
    })
    
    result = get_text_insights(df["notes"], "notes")
    
    # Should handle nulls gracefully  
    assert "avg_text_length" in result
    assert "text_category" in result
    # Function should compute stats on non-null values
    assert result["avg_text_length"] > 0


def test_get_text_insights_multicolumn():
    """Test text insights can be called multiple times for different columns."""
    df = pd.DataFrame({
        "title": ["Title A", "Title B", "Title C"],
        "description": [
            "Long description here",
            "Another description",
            "Third description",
        ],
        "category": ["Cat1", "Cat2", "Cat1"],
    })
    
    result1 = get_text_insights(df["title"], "title")
    result2 = get_text_insights(df["description"], "description")
    result3 = get_text_insights(df["category"], "category")
    
    assert "avg_text_length" in result1
    assert "avg_text_length" in result2
    assert "avg_text_length" in result3
    assert result1["avg_text_length"] == 7  # "Title A", "Title B", "Title C"
    assert result2["avg_text_length"] > result1["avg_text_length"]  # Descriptions are longer
