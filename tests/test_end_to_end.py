"""
End-to-end tests covering complete ML workflows.

These tests ensure that the entire pipeline from data loading through
model training, evaluation, and registry works correctly with proper
train/test splitting to prevent data leakage.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path

from skyulf_mlflow_library.preprocessing import SimpleImputer, StandardScaler, MinMaxScaler
from skyulf_mlflow_library.features.encoding import OneHotEncoder, LabelEncoder
from skyulf_mlflow_library.features import SmartBinning, FeatureMath
from skyulf_mlflow_library.features.selection import FeatureSelector
from skyulf_mlflow_library.modeling import RandomForestClassifier, RandomForestRegressor, MetricsCalculator
from skyulf_mlflow_library.modeling.registry import ModelRegistry


class TestEndToEndClassification:
    """Test complete classification workflow with proper train/test split."""
    
    @pytest.fixture
    def sample_classification_data(self):
        """Create sample classification dataset."""
        np.random.seed(42)
        n_samples = 200
        
        df = pd.DataFrame({
            'age': np.random.randint(18, 70, n_samples),
            'salary': np.random.randint(30000, 150000, n_samples),
            'experience': np.random.randint(0, 40, n_samples),
            'city': np.random.choice(['NYC', 'LA', 'Chicago', 'Boston'], n_samples),
            'education': np.random.choice(['BS', 'MS', 'PhD'], n_samples),
            'target': np.random.choice([0, 1], n_samples)
        })
        
        # Add missing values
        missing_idx = np.random.choice(df.index, size=20, replace=False)
        df.loc[missing_idx[:10], 'age'] = np.nan
        df.loc[missing_idx[10:], 'salary'] = np.nan
        
        return df
    
    def test_complete_pipeline_with_split(self, sample_classification_data):
        """Test complete pipeline: split → impute → encode → scale → train → evaluate."""
        df = sample_classification_data
        
        # Step 1: Split data FIRST
        train_df = df.sample(frac=0.8, random_state=42)
        test_df = df.drop(train_df.index)
        
        assert train_df.shape[0] == 160
        assert test_df.shape[0] == 40
        
        # Step 2: Imputation (fit on train)
        imputer = SimpleImputer(columns=['age', 'salary'], strategy='mean')
        train_imputed = imputer.fit_transform(train_df.copy())
        test_imputed = imputer.transform(test_df.copy())
        
        assert train_imputed['age'].isna().sum() == 0
        assert test_imputed['salary'].isna().sum() == 0
        
        # Step 3: Encoding (fit on train)
        encoder = OneHotEncoder(columns=['city', 'education'])
        train_encoded = encoder.fit_transform(train_imputed)
        test_encoded = encoder.transform(test_imputed)
        
        # Check that both have same columns
        assert set(train_encoded.columns) == set(test_encoded.columns)
        
        # Step 4: Scaling (fit on train)
        scaler = StandardScaler(columns=['age', 'salary', 'experience'])
        train_scaled = scaler.fit_transform(train_encoded)
        test_scaled = scaler.transform(test_encoded)
        
        # Step 5: Prepare features and target
        feature_cols = [col for col in train_scaled.columns if col != 'target']
        X_train = train_scaled[feature_cols]
        y_train = train_scaled['target']
        X_test = test_scaled[feature_cols]
        y_test = test_scaled['target']
        
        # Step 6: Train model
        model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        
        # Step 7: Evaluate
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        metrics_calc = MetricsCalculator(problem_type='classification')
        metrics = metrics_calc.calculate(y_test, y_pred, y_pred_proba)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 0 <= metrics['accuracy'] <= 1
    
    def test_feature_engineering_pipeline(self, sample_classification_data):
        """Test feature engineering with SmartBinning and FeatureMath."""
        df = sample_classification_data
        
        # Split first
        train_df = df.sample(frac=0.8, random_state=42)
        test_df = df.drop(train_df.index)
        
        # Impute missing values first
        imputer = SimpleImputer(columns=['age', 'salary'], strategy='mean')
        train_df = imputer.fit_transform(train_df.copy())
        test_df = imputer.transform(test_df.copy())
        
        # Feature 1: Smart Binning (fit on train)
        binning = SmartBinning(
            columns=['age', 'experience'],
            n_bins=4,
            strategy='equal_frequency',
            labels=['Q1', 'Q2', 'Q3', 'Q4']
        )
        train_binned = binning.fit_transform(train_df.copy())
        test_binned = binning.transform(test_df.copy())
        
        assert 'age_binned' in train_binned.columns
        assert 'age_binned' in test_binned.columns
        
        # Feature 2: FeatureMath with proper operations configuration
        operations = [
            {
                'type': 'arithmetic',
                'method': 'add',
                'columns': ['salary', 'experience'],
                'output': 'total_comp'
            },
            {
                'type': 'arithmetic',
                'method': 'multiply',
                'columns': ['salary', 'experience'],
                'output': 'salary_exp'
            }
        ]
        feature_math = FeatureMath(operations=operations)
        train_with_math = feature_math.fit_transform(train_binned.copy())
        test_with_math = feature_math.transform(test_binned.copy())
        
        assert 'total_comp' in train_with_math.columns
        assert 'salary_exp' in train_with_math.columns
        assert 'total_comp' in test_with_math.columns
        assert 'salary_exp' in test_with_math.columns
    
    def test_feature_selection_workflow(self, sample_classification_data):
        """Test feature selection with proper train/test handling."""
        df = sample_classification_data
        
        # Split and impute
        train_df = df.sample(frac=0.8, random_state=42)
        test_df = df.drop(train_df.index)
        
        imputer = SimpleImputer(columns=['age', 'salary'], strategy='mean')
        train_df = imputer.fit_transform(train_df.copy())
        test_df = imputer.transform(test_df.copy())
        
        # Encode categoricals
        encoder = OneHotEncoder(columns=['city', 'education'])
        train_encoded = encoder.fit_transform(train_df)
        test_encoded = encoder.transform(test_df)
        
        # Feature selection (fit on train)
        selector = FeatureSelector(
            method='select_k_best',
            k=5,
            score_func='f_classif'
        )
        
        train_selected = selector.fit_transform(
            train_encoded.drop(columns=['target']),
            train_encoded['target']
        )
        test_selected = selector.transform(test_encoded.drop(columns=['target']))
        
        # Should have 5 features selected
        assert train_selected.shape[1] == 5
        assert test_selected.shape[1] == 5
        assert set(train_selected.columns) == set(test_selected.columns)
        
        selected_cols = selector.get_selected_columns()
        assert len(selected_cols) == 5


class TestEndToEndRegression:
    """Test complete regression workflow."""
    
    @pytest.fixture
    def sample_regression_data(self):
        """Create sample regression dataset."""
        np.random.seed(42)
        n_samples = 200
        
        df = pd.DataFrame({
            'feature_1': np.random.randn(n_samples),
            'feature_2': np.random.randn(n_samples),
            'feature_3': np.random.randn(n_samples),
            'feature_4': np.random.randn(n_samples),
            'category': np.random.choice(['A', 'B', 'C'], n_samples),
        })
        
        # Create target with some relationship
        df['target'] = (
            2 * df['feature_1'] + 
            1.5 * df['feature_2'] - 
            0.5 * df['feature_3'] + 
            np.random.randn(n_samples) * 0.5
        )
        
        return df
    
    def test_regression_pipeline(self, sample_regression_data):
        """Test regression workflow with proper splitting."""
        df = sample_regression_data
        
        # Split
        train_df = df.sample(frac=0.8, random_state=42)
        test_df = df.drop(train_df.index)
        
        # Encode category
        encoder = OneHotEncoder(columns=['category'])
        train_encoded = encoder.fit_transform(train_df.copy())
        test_encoded = encoder.transform(test_df.copy())
        
        # Scale features
        feature_cols = ['feature_1', 'feature_2', 'feature_3', 'feature_4']
        scaler = StandardScaler(columns=feature_cols)
        train_scaled = scaler.fit_transform(train_encoded)
        test_scaled = scaler.transform(test_encoded)
        
        # Prepare data
        X_train = train_scaled.drop(columns=['target'])
        y_train = train_scaled['target']
        X_test = test_scaled.drop(columns=['target'])
        y_test = test_scaled['target']
        
        # Train
        model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Evaluate
        metrics_calc = MetricsCalculator(problem_type='regression')
        metrics = metrics_calc.calculate(y_test, y_pred)
        
        assert 'mse' in metrics
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'r2' in metrics
        assert metrics['r2'] <= 1.0


class TestModelRegistry:
    """Test model registry with trained models."""
    
    def test_save_and_load_model(self, tmp_path):
        """Test saving and loading model with metadata."""
        # Create simple dataset
        np.random.seed(42)
        X = pd.DataFrame({
            'feature_1': np.random.randn(100),
            'feature_2': np.random.randn(100),
        })
        y = np.random.choice([0, 1], 100)
        
        # Train model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        # Save to registry (use registry_path parameter)
        registry = ModelRegistry(registry_path=str(tmp_path / 'models'))
        
        # Save model with proper API
        model_id = registry.save_model(
            model=model,
            name='test_classifier',
            problem_type='classification',
            metrics={'accuracy': 0.85, 'f1': 0.80},
            tags=['test', 'end-to-end']
        )
        
        assert model_id is not None
        
        # Load model
        loaded_model = registry.load_model('test_classifier')
        
        # Test predictions match
        pred_original = model.predict(X)
        pred_loaded = loaded_model.predict(X)
        
        assert np.array_equal(pred_original, pred_loaded)
    
    def test_model_versioning(self, tmp_path):
        """Test model versioning in registry."""
        registry = ModelRegistry(registry_path=str(tmp_path / 'models'))
        
        # Train two different models
        np.random.seed(42)
        X = pd.DataFrame({
            'feature_1': np.random.randn(100),
            'feature_2': np.random.randn(100),
        })
        y = np.random.choice([0, 1], 100)
        
        # Version 1
        model_v1 = RandomForestClassifier(n_estimators=10, random_state=42)
        model_v1.fit(X, y)
        id_v1 = registry.save_model(
            model=model_v1,
            name='classifier',
            problem_type='classification',
            metrics={'accuracy': 0.80},
            tags=['v1']
        )
        
        # Version 2
        model_v2 = RandomForestClassifier(n_estimators=20, random_state=42)
        model_v2.fit(X, y)
        id_v2 = registry.save_model(
            model=model_v2,
            name='classifier',
            problem_type='classification',
            metrics={'accuracy': 0.85},
            tags=['v2']
        )
        
        # Both versions should be saved
        assert id_v1 != id_v2
        
        # List models - returns a DataFrame
        models_df = registry.list_models(name='classifier')
        assert len(models_df) >= 2
        assert all(models_df['name'] == 'classifier')


class TestProductionPipelineIntegration:
    """Test complete production pipeline matching production_pipeline_with_registry.py example."""
    
    def test_complete_production_workflow(self, tmp_path):
        """
        Test the EXACT workflow from production_pipeline_with_registry.py:
        - FeatureMath (ratio + arithmetic)
        - SmartBinning with custom bins
        - OneHotEncoder
        - StandardScaler
        - RandomForestClassifier
        - Save/load entire pipeline
        - Make predictions
        """
        # Create sample data similar to production example
        np.random.seed(42)
        n_samples = 200
        
        df = pd.DataFrame({
            'customer_id': range(1000, 1000 + n_samples),
            'age': np.random.randint(18, 70, n_samples),
            'tenure_months': np.random.randint(1, 60, n_samples),
            'monthly_charges': np.random.uniform(20, 150, n_samples),
            'total_charges': np.random.uniform(100, 5000, n_samples),
            'contract_type': np.random.choice(['Month-to-Month', 'One Year', 'Two Year'], n_samples),
            'payment_method': np.random.choice(['Electronic Check', 'Credit Card'], n_samples),
            'num_services': np.random.randint(1, 8, n_samples),
        })
        
        # Create target
        churn_prob = 0.1 + 0.2 * (df['tenure_months'] < 12)
        df['churn'] = (np.random.random(n_samples) < churn_prob).astype(int)
        
        # Split data
        train_df = df.sample(frac=0.8, random_state=42)
        test_df = df.drop(train_df.index)
        
        X_train = train_df.drop(columns=['churn'])
        y_train = train_df['churn']
        X_test = test_df.drop(columns=['churn'])
        y_test = test_df['churn']
        
        # Build pipeline components
        # Step 1: Feature Engineering - FeatureMath
        feature_operations = [
            {
                'type': 'ratio',
                'numerator': ['total_charges'],
                'denominator': ['tenure_months'],
                'output': 'avg_monthly_spend'
            },
            {
                'type': 'arithmetic',
                'method': 'multiply',
                'columns': ['monthly_charges', 'num_services'],
                'output': 'service_value'
            }
        ]
        
        feature_math = FeatureMath(operations=feature_operations)
        X_train_eng = feature_math.fit_transform(X_train.drop(columns=['customer_id']))
        X_test_eng = feature_math.transform(X_test.drop(columns=['customer_id']))
        
        assert 'avg_monthly_spend' in X_train_eng.columns
        assert 'service_value' in X_train_eng.columns
        
        # Step 2: Binning
        binner = SmartBinning(
            strategy='custom',
            columns=['age'],
            bins={'age': [18, 30, 45, 60, 70]},
            labels={'age': ['Young', 'Adult', 'Middle', 'Senior']},
            suffix='_group'
        )
        X_train_binned = binner.fit_transform(X_train_eng)
        X_test_binned = binner.transform(X_test_eng)
        
        assert 'age_group' in X_train_binned.columns
        
        # Step 3: Encoding
        categorical_cols = ['contract_type', 'payment_method', 'age_group']
        encoder = OneHotEncoder(columns=categorical_cols)
        X_train_encoded = encoder.fit_transform(X_train_binned)
        X_test_encoded = encoder.transform(X_test_binned)
        
        # Step 4: Scaling
        numerical_cols = ['monthly_charges', 'total_charges']
        scaler = StandardScaler(columns=numerical_cols)
        X_train_scaled = scaler.fit_transform(X_train_encoded)
        X_test_scaled = scaler.transform(X_test_encoded)
        
        # Step 5: Train model
        model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Step 6: Evaluate
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)
        
        metrics_calc = MetricsCalculator(problem_type='classification')
        metrics = metrics_calc.calculate(y_test, y_pred, y_proba)
        
        assert 'accuracy' in metrics
        assert 'f1' in metrics
        assert metrics['accuracy'] > 0.5  # Should be better than random
        
        # Step 7: Save to registry
        registry = ModelRegistry(registry_path=str(tmp_path / 'registry'))
        
        # Create a pipeline dict to save
        pipeline_artifact = {
            'feature_math': feature_math,
            'binner': binner,
            'encoder': encoder,
            'scaler': scaler,
            'model': model,
            'feature_operations': feature_operations,
            'categorical_cols': categorical_cols,
            'numerical_cols': numerical_cols
        }
        
        model_id = registry.save_model(
            model=pipeline_artifact,
            name='complete_pipeline',
            problem_type='classification',
            metrics=metrics,
            description='Complete production pipeline with all transformers',
            tags=['production', 'integration-test']
        )
        
        assert model_id is not None
        
        # Step 8: Load and predict
        loaded_pipeline = registry.load_model('complete_pipeline')
        
        # Reconstruct transformations on new data
        X_new = X_test.iloc[:5].copy()
        X_new_eng = loaded_pipeline['feature_math'].transform(X_new.drop(columns=['customer_id']))
        X_new_binned = loaded_pipeline['binner'].transform(X_new_eng)
        X_new_encoded = loaded_pipeline['encoder'].transform(X_new_binned)
        X_new_scaled = loaded_pipeline['scaler'].transform(X_new_encoded)
        
        y_new_pred = loaded_pipeline['model'].predict(X_new_scaled)
        
        assert len(y_new_pred) == 5
        assert all(pred in [0, 1] for pred in y_new_pred)
    
    def test_model_version_comparison(self, tmp_path):
        """Test comparing multiple versions of models with different metrics."""
        registry = ModelRegistry(registry_path=str(tmp_path / 'registry'))
        
        # Create simple dataset
        np.random.seed(42)
        X = pd.DataFrame({
            'feature_1': np.random.randn(100),
            'feature_2': np.random.randn(100),
        })
        y = np.random.choice([0, 1], 100)
        
        # Save multiple versions with different metrics
        for i in range(3):
            model = RandomForestClassifier(
                n_estimators=10 * (i + 1),
                random_state=42
            )
            model.fit(X, y)
            
            # Simulate different performance
            metrics = {
                'accuracy': 0.70 + (i * 0.05),
                'f1': 0.65 + (i * 0.05),
                'roc_auc': 0.72 + (i * 0.03)
            }
            
            registry.save_model(
                model=model,
                name='churn_model',
                problem_type='classification',
                metrics=metrics,
                tags=[f'version_{i+1}']
            )
        
        # List all versions
        models_df = registry.list_models(name='churn_model')
        
        assert len(models_df) == 3
        assert all(models_df['name'] == 'churn_model')
        
        # Check metrics are preserved
        accuracies = [m['accuracy'] for m in models_df['metrics'].values]
        assert len(accuracies) == 3
        assert max(accuracies) >= 0.79  # Best model should have good accuracy
        
        # Verify we can compare versions
        best_model_row = models_df.loc[models_df['metrics'].apply(lambda m: m.get('accuracy', 0)).idxmax()]
        assert best_model_row['metrics']['accuracy'] >= 0.79


class TestDataLeakagePrevention:
    """Tests to ensure no data leakage in workflows."""
    
    def test_scaler_no_leakage(self):
        """Verify that scaler fit on train doesn't use test data."""
        np.random.seed(42)
        
        # Create data with different distributions
        train_data = pd.DataFrame({
            'value': np.random.normal(100, 10, 100)
        })
        test_data = pd.DataFrame({
            'value': np.random.normal(150, 20, 50)  # Different distribution
        })
        
        # Fit on train
        scaler = StandardScaler(columns=['value'])
        train_scaled = scaler.fit_transform(train_data.copy())
        
        # Get train statistics
        train_mean = scaler.metadata['mean'][0]
        train_std = np.sqrt(scaler.metadata['var'][0])
        
        # Should be close to train data statistics
        assert abs(train_mean - 100) < 5
        assert abs(train_std - 10) < 5
        
        # Transform test data
        test_scaled = scaler.transform(test_data.copy())
        
        # Test data should be scaled using TRAIN statistics
        # So test mean should NOT be 0 (because test distribution is different)
        test_scaled_mean = test_scaled['value'].mean()
        assert abs(test_scaled_mean) > 1  # Should not be centered at 0
    
    def test_encoder_handles_unseen_categories(self):
        """Verify encoder handles categories in test not seen in train."""
        train_data = pd.DataFrame({
            'city': ['NYC', 'LA', 'NYC', 'LA', 'Chicago'] * 20
        })
        
        test_data = pd.DataFrame({
            'city': ['NYC', 'LA', 'Boston', 'Seattle']  # Boston and Seattle not in train
        })
        
        encoder = OneHotEncoder(columns=['city'])
        train_encoded = encoder.fit_transform(train_data.copy())
        
        # Should not raise error on unseen categories
        test_encoded = encoder.transform(test_data.copy())
        
        # Test should have same columns as train (unseen categories handled gracefully)
        assert set(train_encoded.columns).issubset(set(test_encoded.columns))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
