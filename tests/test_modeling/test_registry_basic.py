"""Tests for ModelRegistry with correct API."""

import pytest
import tempfile
import shutil
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression

from skyulf_mlflow_library.modeling import ModelRegistry


class TestModelRegistryBasic:
    """Test basic ModelRegistry functionality."""
    
    @pytest.fixture
    def temp_registry_path(self):
        """Create a temporary registry directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_registry_initialization(self, temp_registry_path):
        """Test that registry initializes correctly."""
        registry = ModelRegistry(registry_path=temp_registry_path)
        
        # Check that registry path exists
        assert Path(temp_registry_path).exists()
        
        # Check that models directory exists
        models_path = Path(temp_registry_path) / 'models'
        assert models_path.exists()
        
        # Check that database exists
        db_path = Path(temp_registry_path) / 'registry.db'
        assert db_path.exists()
    
    def test_save_and_load_model(self, temp_registry_path):
        """Test saving and loading a model."""
        registry = ModelRegistry(registry_path=temp_registry_path)
        
        # Create and train a simple model
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        X = np.random.rand(50, 5)
        y = np.random.randint(0, 2, 50)
        model.fit(X, y)
        
        # Save model
        model_id = registry.save_model(
            model=model,
            name='test_classifier',
            problem_type='classification',
            metrics={'accuracy': 0.85}
        )
        
        # Load model
        loaded_model = registry.load_model('test_classifier')
        
        # Verify model works
        assert loaded_model is not None
        predictions = loaded_model.predict(X[:5])
        assert len(predictions) == 5
    
    def test_save_model_with_version(self, temp_registry_path):
        """Test saving multiple versions of the same model."""
        registry = ModelRegistry(registry_path=temp_registry_path)
        
        # Save first version
        model1 = LogisticRegression(random_state=42)
        X = np.random.rand(50, 5)
        y = np.random.randint(0, 2, 50)
        model1.fit(X, y)
        
        id1 = registry.save_model(
            model=model1,
            name='versioned_model',
            problem_type='classification'
        )
        
        # Save second version
        model2 = LogisticRegression(random_state=43)
        model2.fit(X, y)
        
        id2 = registry.save_model(
            model=model2,
            name='versioned_model',
            problem_type='classification'
        )
        
        # IDs should be different
        assert id1 != id2
    
    def test_list_models(self, temp_registry_path):
        """Test listing registered models."""
        registry = ModelRegistry(registry_path=temp_registry_path)
        
        # Initially should be empty
        models = registry.list_models()
        assert len(models) == 0
        
        # Add a model
        model = RandomForestRegressor(n_estimators=5, random_state=42)
        X = np.random.rand(50, 5)
        y = np.random.rand(50)
        model.fit(X, y)
        
        registry.save_model(
            model=model,
            name='test_regressor',
            problem_type='regression'
        )
        
        # Should now have 1 model
        models = registry.list_models()
        assert len(models) >= 1
    
    def test_save_model_with_tags(self, temp_registry_path):
        """Test saving model with tags."""
        registry = ModelRegistry(registry_path=temp_registry_path)
        
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        X = np.random.rand(50, 5)
        y = np.random.randint(0, 2, 50)
        model.fit(X, y)
        
        model_id = registry.save_model(
            model=model,
            name='tagged_model',
            problem_type='classification',
            tags=['production', 'v1']
        )
        
        assert model_id is not None
    
    def test_load_nonexistent_model(self, temp_registry_path):
        """Test loading a model that doesn't exist."""
        registry = ModelRegistry(registry_path=temp_registry_path)
        
        with pytest.raises((ValueError, KeyError, FileNotFoundError, Exception)):
            registry.load_model('nonexistent_model')


class TestModelRegistryAdvanced:
    """Test advanced ModelRegistry features."""
    
    @pytest.fixture
    def temp_registry_path(self):
        """Create a temporary registry directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_registry_with_description(self, temp_registry_path):
        """Test saving model with description."""
        registry = ModelRegistry(registry_path=temp_registry_path)
        
        model = LogisticRegression(random_state=42)
        X = np.random.rand(50, 5)
        y = np.random.randint(0, 2, 50)
        model.fit(X, y)
        
        model_id = registry.save_model(
            model=model,
            name='described_model',
            problem_type='classification',
            description='A test model with description'
        )
        
        assert model_id is not None
