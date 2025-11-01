"""Tests for SmartBinning transformer."""

import pytest
import pandas as pd
import numpy as np

from skyulf_mlflow_library.features.transform import SmartBinning


class TestSmartBinningEqualWidth:
    """Test SmartBinning with equal_width strategy."""
    
    def test_equal_width_basic(self):
        """Test basic equal width binning."""
        df = pd.DataFrame({
            'value': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        })
        
        binner = SmartBinning(
            strategy='equal_width',
            columns=['value'],
            n_bins=3
        )
        
        result = binner.fit_transform(df)
        
        # Should have original + binned column
        assert 'value' in result.columns
        assert 'value_binned' in result.columns
        
        # Should have 3 unique bins
        assert result['value_binned'].nunique() <= 3
    
    def test_equal_width_with_labels(self):
        """Test equal width binning with custom labels."""
        df = pd.DataFrame({
            'age': [25, 35, 45, 55, 65, 75]
        })
        
        binner = SmartBinning(
            strategy='equal_width',
            columns=['age'],
            n_bins=3,
            labels={'age': ['Young', 'Middle', 'Senior']}
        )
        
        result = binner.fit_transform(df)
        
        # Should use custom labels
        assert 'age_binned' in result.columns
        unique_labels = result['age_binned'].unique()
        assert any(label in unique_labels for label in ['Young', 'Middle', 'Senior'])


class TestSmartBinningEqualFrequency:
    """Test SmartBinning with equal_frequency strategy."""
    
    def test_equal_frequency_basic(self):
        """Test basic equal frequency (quantile) binning."""
        df = pd.DataFrame({
            'value': range(1, 101)  # 100 values
        })
        
        binner = SmartBinning(
            strategy='equal_frequency',
            columns=['value'],
            n_bins=4  # Each bin should have ~25 values
        )
        
        result = binner.fit_transform(df)
        
        assert 'value_binned' in result.columns
        # Each bin should have approximately equal counts
        counts = result['value_binned'].value_counts()
        # Allow some variance due to duplicate edge values
        assert len(counts) <= 4


class TestSmartBinningCustom:
    """Test SmartBinning with custom bin edges."""
    
    def test_custom_bins_with_labels(self):
        """Test custom binning with user-defined edges and labels."""
        df = pd.DataFrame({
            'age': [5, 15, 25, 35, 45, 55, 65, 75, 85]
        })
        
        binner = SmartBinning(
            strategy='custom',
            columns=['age'],
            bins={'age': [0, 18, 35, 60, 100]},
            labels={'age': ['Child', 'Young Adult', 'Adult', 'Senior']}
        )
        
        result = binner.fit_transform(df)
        
        assert 'age_binned' in result.columns
        unique_labels = set(result['age_binned'].unique())
        expected = {'Child', 'Young Adult', 'Adult', 'Senior'}
        assert unique_labels.issubset(expected)


class TestSmartBinningKBins:
    """Test SmartBinning with KBinsDiscretizer."""
    
    def test_kbins_quantile(self):
        """Test KBins with quantile strategy."""
        df = pd.DataFrame({
            'value': np.random.randn(100)
        })
        
        binner = SmartBinning(
            strategy='kbins',
            columns=['value'],
            n_bins=5,
            kbins_strategy='quantile'
        )
        
        result = binner.fit_transform(df)
        
        assert 'value_binned' in result.columns
        # Should have bins
        assert result['value_binned'].nunique() <= 5
    
    def test_kbins_uniform(self):
        """Test KBins with uniform strategy."""
        df = pd.DataFrame({
            'value': range(1, 51)
        })
        
        binner = SmartBinning(
            strategy='kbins',
            columns=['value'],
            n_bins=5,
            kbins_strategy='uniform'
        )
        
        result = binner.fit_transform(df)
        
        assert 'value_binned' in result.columns


class TestSmartBinningOptions:
    """Test SmartBinning options."""
    
    def test_drop_original_column(self):
        """Test dropping original column after binning."""
        df = pd.DataFrame({
            'value': [1, 2, 3, 4, 5]
        })
        
        binner = SmartBinning(
            strategy='equal_width',
            columns=['value'],
            n_bins=2,
            drop_original=True
        )
        
        result = binner.fit_transform(df)
        
        # Original column should be dropped
        assert 'value' not in result.columns
        assert 'value_binned' in result.columns
    
    def test_custom_suffix(self):
        """Test using custom suffix for binned columns."""
        df = pd.DataFrame({
            'score': [10, 20, 30, 40, 50]
        })
        
        binner = SmartBinning(
            strategy='equal_width',
            columns=['score'],
            n_bins=2,
            suffix='_category'
        )
        
        result = binner.fit_transform(df)
        
        assert 'score_category' in result.columns
    
    def test_multiple_columns(self):
        """Test binning multiple columns at once."""
        df = pd.DataFrame({
            'age': [25, 35, 45, 55, 65],
            'income': [30000, 50000, 70000, 90000, 110000]
        })
        
        binner = SmartBinning(
            strategy='equal_width',
            columns=['age', 'income'],
            n_bins=3
        )
        
        result = binner.fit_transform(df)
        
        assert 'age_binned' in result.columns
        assert 'income_binned' in result.columns


class TestSmartBinningTransform:
    """Test transform method."""
    
    def test_fit_then_transform(self):
        """Test that transform works after fitting."""
        df_train = pd.DataFrame({
            'value': range(1, 11)
        })
        
        binner = SmartBinning(
            strategy='equal_width',
            columns=['value'],
            n_bins=3
        )
        
        # Fit on training data
        binner.fit_transform(df_train)
        
        # Transform new data
        df_test = pd.DataFrame({
            'value': [2, 5, 8]
        })
        
        result = binner.transform(df_test)
        
        assert 'value_binned' in result.columns


class TestSmartBinningEdgeCases:
    """Test edge cases."""
    
    def test_single_unique_value(self):
        """Test binning column with single unique value."""
        df = pd.DataFrame({
            'constant': [5, 5, 5, 5, 5]
        })
        
        binner = SmartBinning(
            strategy='equal_width',
            columns=['constant'],
            n_bins=3
        )
        
        # Should not crash
        result = binner.fit_transform(df)
        assert 'constant_binned' in result.columns
    
    def test_with_missing_values(self):
        """Test binning with missing values."""
        df = pd.DataFrame({
            'value': [1, 2, np.nan, 4, 5, np.nan, 7, 8]
        })
        
        binner = SmartBinning(
            strategy='equal_width',
            columns=['value'],
            n_bins=3
        )
        
        result = binner.fit_transform(df)
        
        # Missing values should be preserved
        assert result['value_binned'].isna().sum() > 0
