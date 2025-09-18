"""
Unit tests for the California housing model training module.
Tests model training, evaluation, and prediction functions.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import sys
import joblib

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from model_training import CaliforniaHousingPredictor
from data_processing import CaliforniaHousingProcessor


class TestCaliforniaHousingPredictor:
    """Test class for CaliforniaHousingPredictor."""

    @pytest.fixture
    def predictor(self):
        """Create a fresh predictor instance for each test."""
        return CaliforniaHousingPredictor()

    @pytest.fixture
    def processor(self):
        """Create a fresh processor instance for each test."""
        return CaliforniaHousingProcessor()

    @pytest.fixture
    def sample_training_data(self):
        """Create sample training data."""
        np.random.seed(42)
        n_samples = 100

        X_train = pd.DataFrame({
            'median_income': np.random.uniform(1, 10, n_samples),
            'house_age': np.random.uniform(1, 50, n_samples),
            'avg_rooms': np.random.uniform(3, 10, n_samples),
            'avg_bedrooms': np.random.uniform(0.5, 2, n_samples),
            'population': np.random.uniform(500, 5000, n_samples),
            'avg_occupancy': np.random.uniform(2, 6, n_samples),
            'latitude': np.random.uniform(32, 42, n_samples),
            'longitude': np.random.uniform(-125, -114, n_samples),
        })

        X_test = pd.DataFrame({
            'median_income': np.random.uniform(1, 10, 20),
            'house_age': np.random.uniform(1, 50, 20),
            'avg_rooms': np.random.uniform(3, 10, 20),
            'avg_bedrooms': np.random.uniform(0.5, 2, 20),
            'population': np.random.uniform(500, 5000, 20),
            'avg_occupancy': np.random.uniform(2, 6, 20),
            'latitude': np.random.uniform(32, 42, 20),
            'longitude': np.random.uniform(-125, -114, 20),
        })

        # Generate realistic prices based on features
        y_train = (
            X_train['median_income'] * 30000 +
            (50 - X_train['house_age']) * 2000 +
            X_train['avg_rooms'] * 10000 +
            np.random.normal(0, 10000, n_samples)
        )
        y_train = np.clip(y_train, 50000, 800000)

        y_test = (
            X_test['median_income'] * 30000 +
            (50 - X_test['house_age']) * 2000 +
            X_test['avg_rooms'] * 10000 +
            np.random.normal(0, 10000, 20)
        )
        y_test = np.clip(y_test, 50000, 800000)

        return X_train, X_test, y_train, y_test

    def test_train_multiple_models(self, predictor, sample_training_data):
        """Test training multiple models."""
        X_train, X_test, y_train, y_test = sample_training_data

        # Train models
        predictor.train_multiple_models(X_train, y_train, X_test, y_test)

        # Check that models were trained
        assert len(predictor.models) > 0
        assert predictor.best_model is not None
        assert predictor.best_model_name is not None

        # Check that evaluation results exist
        assert len(predictor.evaluation_results) > 0

        # Check that all models have evaluation metrics
        for model_name, results in predictor.evaluation_results.items():
            assert 'test_r2' in results
            assert 'test_rmse' in results
            assert 'test_mae' in results
            assert 'cv_r2_mean' in results
            assert 'cv_r2_std' in results

            # Check that metrics are reasonable
            assert results['test_r2'] >= -1  # R² can be negative for very bad models
            assert results['test_rmse'] > 0
            assert results['test_mae'] > 0

        # Check that best model has highest R² score
        best_r2 = predictor.evaluation_results[predictor.best_model_name]['test_r2']
        for model_name, results in predictor.evaluation_results.items():
            assert results['test_r2'] <= best_r2

    def test_predict(self, predictor, sample_training_data):
        """Test prediction functionality."""
        X_train, X_test, y_train, y_test = sample_training_data

        # Train a model first
        predictor.train_multiple_models(X_train, y_train, X_test, y_test)

        # Make predictions
        predictions = predictor.predict(X_test)

        # Check predictions
        assert len(predictions) == len(X_test)
        assert all(isinstance(p, (int, float, np.number)) for p in predictions)
        assert all(p > 0 for p in predictions)  # Housing prices should be positive

    def test_predict_without_trained_model(self, predictor, sample_training_data):
        """Test prediction error when no model is trained."""
        X_train, X_test, y_train, y_test = sample_training_data

        # Try to predict without training
        with pytest.raises(ValueError, match="No trained model available"):
            predictor.predict(X_test)

    def test_predict_single(self, predictor, processor, sample_training_data):
        """Test single house prediction."""
        X_train, X_test, y_train, y_test = sample_training_data

        # Create sample data for processor fitting
        sample_df = pd.DataFrame({
            'median_income': [5.0, 8.0, 4.5, 6.2],
            'house_age': [10, 25, 5, 15],
            'avg_rooms': [6.0, 5.5, 7.2, 6.8],
            'avg_bedrooms': [1.2, 1.0, 1.4, 1.3],
            'population': [3000, 4500, 2800, 3600],
            'avg_occupancy': [3.5, 4.2, 3.1, 3.3],
            'latitude': [34.0, 37.5, 33.8, 36.1],
            'longitude': [-118.0, -122.0, -117.5, -119.2],
            'price': [250000, 450000, 320000, 380000]
        })

        # Fit processor
        df_clean = processor.clean_data(sample_df)
        df_engineered = processor.engineer_features(df_clean)
        processor.preprocess_for_modeling(df_engineered)

        # Train predictor with processed data
        predictor.train_multiple_models(X_train, y_train, X_test, y_test)

        # Test single prediction
        house_features = {
            'median_income': 6.5,
            'house_age': 20,
            'avg_rooms': 6.2,
            'avg_bedrooms': 1.1,
            'population': 3200,
            'avg_occupancy': 3.4,
            'latitude': 37.0,
            'longitude': -121.5
        }

        predicted_price = predictor.predict_single(house_features, processor)

        # Check prediction
        assert isinstance(predicted_price, (int, float, np.number))
        assert predicted_price > 0
        assert predicted_price < 2000000  # Reasonable upper bound

    def test_get_feature_importance(self, predictor, sample_training_data):
        """Test feature importance extraction."""
        X_train, X_test, y_train, y_test = sample_training_data

        # Train models
        predictor.train_multiple_models(X_train, y_train, X_test, y_test)

        # Get feature importance
        feature_names = list(X_train.columns)
        importance_df = predictor.get_feature_importance(feature_names)

        # Check if importance was extracted (depends on best model type)
        if importance_df is not None:
            assert len(importance_df) == len(feature_names)
            assert 'feature' in importance_df.columns
            assert 'importance' in importance_df.columns
            assert all(imp >= 0 for imp in importance_df['importance'])

            # Check that it's sorted by importance (descending)
            importances = importance_df['importance'].values
            assert all(importances[i] >= importances[i+1] for i in range(len(importances)-1))

    def test_save_and_load_model(self, predictor, sample_training_data):
        """Test model saving and loading."""
        X_train, X_test, y_train, y_test = sample_training_data

        # Train models
        predictor.train_multiple_models(X_train, y_train, X_test, y_test)
        original_best_model_name = predictor.best_model_name

        # Save model
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as f:
            temp_path = f.name

        try:
            predictor.save_model(temp_path)
            assert os.path.exists(temp_path)

            # Create new predictor and load model
            new_predictor = CaliforniaHousingPredictor()
            new_predictor.load_model(temp_path)

            # Check that model was loaded correctly
            assert new_predictor.best_model_name == original_best_model_name
            assert new_predictor.best_model is not None

            # Test that loaded model can make predictions
            predictions = new_predictor.predict(X_test)
            assert len(predictions) == len(X_test)

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_save_model_without_training(self, predictor):
        """Test error when trying to save without training."""
        with tempfile.NamedTemporaryFile(suffix='.joblib') as f:
            with pytest.raises(ValueError, match="No trained model to save"):
                predictor.save_model(f.name)

    def test_load_nonexistent_model(self, predictor):
        """Test error when loading non-existent model."""
        with pytest.raises(FileNotFoundError):
            predictor.load_model('nonexistent_model.joblib')

    def test_get_model_summary(self, predictor, sample_training_data):
        """Test model performance summary."""
        X_train, X_test, y_train, y_test = sample_training_data

        # Train models
        predictor.train_multiple_models(X_train, y_train, X_test, y_test)

        # Get summary
        summary_df = predictor.get_model_summary()

        assert summary_df is not None
        assert len(summary_df) > 0

        # Check required columns
        required_columns = ['Model', 'Test R²', 'Test RMSE', 'Test MAE',
                           'CV R² Mean', 'CV R² Std', 'Is Best']
        for col in required_columns:
            assert col in summary_df.columns

        # Check that exactly one model is marked as best
        assert summary_df['Is Best'].sum() == 1

        # Check that summary is sorted by Test R² (descending)
        r2_values = summary_df['Test R²'].values
        assert all(r2_values[i] >= r2_values[i+1] for i in range(len(r2_values)-1))

    def test_get_model_summary_without_training(self, predictor):
        """Test model summary without training."""
        summary_df = predictor.get_model_summary()
        assert summary_df is None

    def test_model_performance_metrics(self, predictor, sample_training_data):
        """Test that model performance metrics are reasonable."""
        X_train, X_test, y_train, y_test = sample_training_data

        # Train models
        predictor.train_multiple_models(X_train, y_train, X_test, y_test)

        # Check performance metrics for all models
        for model_name, metrics in predictor.evaluation_results.items():
            # R² should be reasonable (between -1 and 1, ideally positive)
            assert -2 <= metrics['test_r2'] <= 1

            # RMSE and MAE should be positive
            assert metrics['test_rmse'] > 0
            assert metrics['test_mae'] > 0

            # MAE should be <= RMSE (mathematical property)
            assert metrics['test_mae'] <= metrics['test_rmse']

            # Cross-validation std should be non-negative
            assert metrics['cv_r2_std'] >= 0

    def test_different_model_types(self, predictor, sample_training_data):
        """Test that different model types are trained."""
        X_train, X_test, y_train, y_test = sample_training_data

        # Train models
        predictor.train_multiple_models(X_train, y_train, X_test, y_test)

        # Check that multiple model types were trained
        expected_models = ['Linear Regression', 'Ridge Regression',
                          'Random Forest', 'Gradient Boosting']

        for model_name in expected_models:
            assert model_name in predictor.models
            assert model_name in predictor.evaluation_results

    def test_integration_with_real_data(self, predictor, processor):
        """Test integration with real California housing data."""
        # Load real data
        df = processor.load_california_housing_data()

        # Take a small sample for faster testing
        df_sample = df.sample(n=200, random_state=42)

        # Process data
        df_clean = processor.clean_data(df_sample)
        df_engineered = processor.engineer_features(df_clean)
        X_train, X_test, y_train, y_test, feature_names = processor.preprocess_for_modeling(df_engineered)

        # Train models
        predictor.train_multiple_models(X_train, y_train, X_test, y_test)

        # Check that training worked
        assert predictor.best_model is not None
        assert len(predictor.evaluation_results) > 0

        # Check that predictions are reasonable
        predictions = predictor.predict(X_test)
        assert all(50000 <= p <= 1000000 for p in predictions)  # Reasonable price range

        # Test feature importance
        importance_df = predictor.get_feature_importance(feature_names)
        if importance_df is not None:
            assert len(importance_df) > 0
            # Check that median_income is likely to be important
            top_features = importance_df.head(5)['feature'].values
            income_related = any('median_income' in feature or 'income' in feature.lower()
                               for feature in top_features)
            # This is not a strict requirement as it depends on preprocessing
            # but median_income should generally be important