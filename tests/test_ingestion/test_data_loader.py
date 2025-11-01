"""Tests for data ingestion module."""

import pandas as pd
import pytest
from pathlib import Path
import tempfile
import os

from skyulf_mlflow_library.data_ingestion import DataLoader, save_data, load_data


def test_data_loader_csv(tmp_path):
    """Test loading CSV files."""
    # Create test CSV
    csv_file = tmp_path / "test.csv"
    df_original = pd.DataFrame({
        "name": ["Alice", "Bob", "Charlie"],
        "age": [25, 30, 35],
        "city": ["NYC", "LA", "SF"],
    })
    df_original.to_csv(csv_file, index=False)
    
    # Load using DataLoader
    loader = DataLoader(csv_file)
    df_loaded = loader.load()
    
    assert df_loaded.shape == df_original.shape
    assert list(df_loaded.columns) == list(df_original.columns)
    assert df_loaded["age"].tolist() == [25, 30, 35]


def test_data_loader_csv_with_params(tmp_path):
    """Test loading CSV with custom parameters."""
    csv_file = tmp_path / "test.csv"
    df_original = pd.DataFrame({
        "name": ["Alice", "Bob"],
        "value": [100, 200],
    })
    df_original.to_csv(csv_file, index=False, sep=";")
    
    loader = DataLoader(csv_file, sep=";")
    df_loaded = loader.load()
    
    assert df_loaded.shape == (2, 2)


def test_save_and_load_data(tmp_path):
    """Test save_data and load_data convenience functions."""
    csv_file = tmp_path / "output.csv"
    df_original = pd.DataFrame({
        "x": [1, 2, 3],
        "y": [4, 5, 6],
    })
    
    # Save
    save_data(df_original, csv_file)
    
    # Load
    df_loaded = load_data(csv_file)
    
    assert df_loaded.shape == df_original.shape
    pd.testing.assert_frame_equal(df_loaded, df_original)


@pytest.mark.skip(reason="openpyxl not installed")
def test_data_loader_excel(tmp_path):
    """Test loading Excel files."""
    excel_file = tmp_path / "test.xlsx"
    df_original = pd.DataFrame({
        "product": ["A", "B", "C"],
        "price": [10.0, 20.0, 30.0],
    })
    df_original.to_excel(excel_file, index=False)
    
    loader = DataLoader(excel_file)
    df_loaded = loader.load()
    
    assert df_loaded.shape == df_original.shape
    assert "product" in df_loaded.columns


def test_data_loader_json(tmp_path):
    """Test loading JSON files."""
    json_file = tmp_path / "test.json"
    df_original = pd.DataFrame({
        "id": [1, 2, 3],
        "name": ["A", "B", "C"],
    })
    df_original.to_json(json_file, orient="records")
    
    loader = DataLoader(json_file)
    df_loaded = loader.load()
    
    assert df_loaded.shape == df_original.shape


def test_data_loader_auto_detection(tmp_path):
    """Test automatic file type detection."""
    csv_file = tmp_path / "data.csv"
    df = pd.DataFrame({"col": [1, 2, 3]})
    df.to_csv(csv_file, index=False)
    
    # Don't specify file_type, should auto-detect from extension
    loader = DataLoader(csv_file)
    df_loaded = loader.load()
    
    assert df_loaded.shape == (3, 1)


def test_data_loader_nonexistent_file():
    """Test error handling for nonexistent files."""
    with pytest.raises(FileNotFoundError):
        loader = DataLoader("nonexistent_file.csv")
        loader.load()


def test_save_data_create_directory(tmp_path):
    """Test that save_data creates directories if needed."""
    nested_path = tmp_path / "subdir1" / "subdir2" / "data.csv"
    df = pd.DataFrame({"a": [1, 2]})
    
    save_data(df, nested_path)
    
    assert nested_path.exists()
    df_loaded = load_data(nested_path)
    assert df_loaded.shape == (2, 1)


def test_data_loader_empty_csv(tmp_path):
    """Test loading empty CSV causes error."""
    csv_file = tmp_path / "empty.csv"
    pd.DataFrame().to_csv(csv_file, index=False)
    
    loader = DataLoader(csv_file)
    
    # pandas raises EmptyDataError for completely empty CSV
    with pytest.raises(Exception):  # EmptyDataError
        df = loader.load()


def test_save_data_parquet(tmp_path):
    """Test saving and loading Parquet files."""
    parquet_file = tmp_path / "data.parquet"
    df_original = pd.DataFrame({
        "num": [1, 2, 3, 4, 5],
        "text": ["a", "b", "c", "d", "e"],
    })
    
    save_data(df_original, parquet_file)
    df_loaded = load_data(parquet_file)
    
    pd.testing.assert_frame_equal(df_loaded, df_original)
