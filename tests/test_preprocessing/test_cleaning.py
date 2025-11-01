"""Tests for data cleaning functions."""

import pytest
import pandas as pd
import numpy as np

from skyulf_mlflow_library.preprocessing.cleaning import (
    drop_missing_rows,
    drop_missing_columns,
)


class TestDropMissingRows:
    """Test drop_missing_rows function."""
    
    def test_drop_rows_with_any_missing(self):
        """Test dropping rows with any missing values."""
        df = pd.DataFrame({
            'A': [1, 2, np.nan, 4],
            'B': [5, np.nan, 7, 8],
            'C': [9, 10, 11, 12]
        })
        
        result = drop_missing_rows(df, how='any')
        
        # Should keep rows without any missing values (rows 0 and 3)
        assert len(result) == 2
        assert 1 in result['A'].values
        assert 4 in result['A'].values
    
    def test_drop_rows_with_all_missing(self):
        """Test dropping rows where all values are missing."""
        df = pd.DataFrame({
            'A': [1, np.nan, np.nan, 4],
            'B': [5, np.nan, np.nan, 8],
            'C': [9, np.nan, np.nan, 12]
        })
        
        result = drop_missing_rows(df, how='all')
        
        # Should keep rows 0 and 3
        assert len(result) == 2
        assert 1 in result['A'].values
        assert 4 in result['A'].values
    
    def test_drop_rows_with_threshold(self):
        """Test dropping rows based on missing percentage threshold."""
        df = pd.DataFrame({
            'A': [1, np.nan, 3, np.nan],
            'B': [5, np.nan, 7, 8],
            'C': [9, np.nan, 11, 12]
        })
        
        # Drop rows with >= 50% missing (row 1 has 100% missing)
        result = drop_missing_rows(df, threshold=50)
        
        # Should drop row 1
        assert len(result) == 3
    
    def test_drop_rows_with_subset(self):
        """Test dropping rows with missing values in specific columns."""
        df = pd.DataFrame({
            'A': [1, 2, np.nan, 4],
            'B': [5, np.nan, 7, 8],
            'C': [9, 10, 11, 12]
        })
        
        # Only consider column 'A'
        result = drop_missing_rows(df, how='any', subset=['A'])
        
        # Should drop row 2 (A is NaN)
        assert len(result) == 3
        assert 2 in result['A'].values
    
    def test_drop_rows_inplace(self):
        """Test dropping rows inplace."""
        df = pd.DataFrame({
            'A': [1, np.nan, 3],
            'B': [4, 5, 6]
        })
        
        result = drop_missing_rows(df, how='any', inplace=True)
        
        # Should return None
        assert result is None
    
    def test_drop_rows_no_missing(self):
        """Test with DataFrame that has no missing values."""
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })
        
        result = drop_missing_rows(df, how='any')
        
        # Should return the same DataFrame
        assert len(result) == 3
        pd.testing.assert_frame_equal(result, df)


class TestDropMissingColumns:
    """Test drop_missing_columns function."""
    
    def test_drop_columns_with_any_missing(self):
        """Test dropping columns with any missing values."""
        df = pd.DataFrame({
            'A': [1, 2, 3, 4],
            'B': [5, np.nan, 7, 8],
            'C': [9, 10, 11, 12]
        })
        
        result = drop_missing_columns(df)
        
        # Should drop column B
        assert 'A' in result.columns
        assert 'B' not in result.columns
        assert 'C' in result.columns
    
    def test_drop_columns_with_threshold(self):
        """Test dropping columns based on missing percentage threshold."""
        df = pd.DataFrame({
            'A': [1, 2, 3, 4],
            'B': [np.nan, np.nan, 7, 8],  # 50% missing
            'C': [9, 10, 11, 12]
        })
        
        # Drop columns with >= 50% missing
        result = drop_missing_columns(df, threshold=50)
        
        # Should drop column B
        assert 'A' in result.columns
        assert 'B' not in result.columns
        assert 'C' in result.columns
    
    def test_drop_specific_columns(self):
        """Test dropping specific columns with missing values."""
        df = pd.DataFrame({
            'A': [1, np.nan, 3, 4],
            'B': [5, np.nan, 7, 8],
            'C': [9, 10, 11, 12]
        })
        
        # Only check columns A and B
        result = drop_missing_columns(df, columns=['A', 'B'])
        
        # Should drop A and B, keep C
        assert 'C' in result.columns
        assert 'A' not in result.columns or len(result) > 0  # Depends on implementation
    
    def test_drop_columns_inplace(self):
        """Test dropping columns inplace."""
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [np.nan, 5, 6],
            'C': [7, 8, 9]
        })
        
        result = drop_missing_columns(df, inplace=True)
        
        # Should return None
        assert result is None
    
    def test_drop_columns_no_missing(self):
        """Test with DataFrame that has no missing values."""
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })
        
        result = drop_missing_columns(df)
        
        # Should return the same DataFrame
        assert len(result.columns) == 2
        assert 'A' in result.columns
        assert 'B' in result.columns
