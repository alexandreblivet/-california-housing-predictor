import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
import logging
from data_processing import CaliforniaHousingProcessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CaliforniaHousingPredictor:
    """
    A class to handle model training, evaluation, and prediction for housing prices.
    """

    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_importance = None
        self.evaluation_results = {}

    def train_multiple_models(self, X_train, y_train, X_test, y_test):
        """
        Train multiple machine learning models and compare their performance.
        """
        # Define models to train
        model_configs = {
            'Linear Regression': {
                'model': LinearRegression(),
                'params': {}
            },
            'Ridge Regression': {
                'model': Ridge(),
                'params': {
                    'alpha': [0.1, 1.0, 10.0, 100.0]
                }
            },
            'Random Forest': {
                'model': RandomForestRegressor(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10]
                }
            },
            'Gradient Boosting': {
                'model': GradientBoostingRegressor(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.05, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                }
            }
        }

        best_score = -np.inf

        for model_name, config in model_configs.items():
            logger.info(f"Training {model_name}...")

            if config['params']:
                # Hyperparameter tuning with GridSearchCV
                grid_search = GridSearchCV(
                    config['model'],
                    config['params'],
                    cv=5,
                    scoring='neg_mean_squared_error',
                    n_jobs=-1
                )
                grid_search.fit(X_train, y_train)
                model = grid_search.best_estimator_
                logger.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
            else:
                # Train without hyperparameter tuning
                model = config['model']
                model.fit(X_train, y_train)

            # Store the trained model
            self.models[model_name] = model

            # Evaluate the model
            train_predictions = model.predict(X_train)
            test_predictions = model.predict(X_test)

            # Calculate metrics
            train_r2 = r2_score(y_train, train_predictions)
            test_r2 = r2_score(y_test, test_predictions)
            train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
            test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
            test_mae = mean_absolute_error(y_test, test_predictions)

            # Cross-validation score
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()

            # Store evaluation results
            self.evaluation_results[model_name] = {
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'test_mae': test_mae,
                'cv_r2_mean': cv_mean,
                'cv_r2_std': cv_std
            }

            logger.info(f"{model_name} - Test R²: {test_r2:.4f}, Test RMSE: {test_rmse:.2f}")

            # Update best model based on test R² score
            if test_r2 > best_score:
                best_score = test_r2
                self.best_model = model
                self.best_model_name = model_name

        logger.info(f"Best model: {self.best_model_name} with R² score: {best_score:.4f}")

        # Extract feature importance from the best model if available
        if hasattr(self.best_model, 'feature_importances_'):
            self.feature_importance = self.best_model.feature_importances_
        elif hasattr(self.best_model, 'coef_'):
            self.feature_importance = np.abs(self.best_model.coef_)

    def get_feature_importance(self, feature_names):
        """Get feature importance from the best model."""
        if self.feature_importance is not None and feature_names:
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': self.feature_importance
            }).sort_values('importance', ascending=False)
            return importance_df
        return None

    def predict(self, X):
        """Make predictions using the best model."""
        if self.best_model is not None:
            return self.best_model.predict(X)
        else:
            raise ValueError("No trained model available. Please train a model first.")

    def predict_single(self, features_dict, processor):
        """Make a prediction for a single house."""
        # Create DataFrame from features
        df = pd.DataFrame([features_dict])

        # Transform the data
        X_processed = processor.transform_new_data(df)

        # Make prediction
        prediction = self.predict(X_processed)
        return prediction[0]

    def save_model(self, filepath):
        """Save the best model to disk."""
        if self.best_model is not None:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            model_data = {
                'model': self.best_model,
                'model_name': self.best_model_name,
                'feature_importance': self.feature_importance,
                'evaluation_results': self.evaluation_results
            }
            joblib.dump(model_data, filepath)
            logger.info(f"Model saved to {filepath}")
        else:
            raise ValueError("No trained model to save.")

    def load_model(self, filepath):
        """Load a saved model from disk."""
        try:
            model_data = joblib.load(filepath)
            self.best_model = model_data['model']
            self.best_model_name = model_data['model_name']
            self.feature_importance = model_data.get('feature_importance')
            self.evaluation_results = model_data.get('evaluation_results', {})
            logger.info(f"Model loaded from {filepath}")
        except FileNotFoundError:
            logger.error(f"Model file not found: {filepath}")
            raise
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def get_model_summary(self):
        """Get a summary of all trained models' performance."""
        if not self.evaluation_results:
            return None

        summary_data = []
        for model_name, metrics in self.evaluation_results.items():
            summary_data.append({
                'Model': model_name,
                'Test R²': metrics['test_r2'],
                'Test RMSE': metrics['test_rmse'],
                'Test MAE': metrics['test_mae'],
                'CV R² Mean': metrics['cv_r2_mean'],
                'CV R² Std': metrics['cv_r2_std'],
                'Is Best': model_name == self.best_model_name
            })

        return pd.DataFrame(summary_data).sort_values('Test R²', ascending=False)


def main():
    """Main function to demonstrate the model training pipeline."""
    # Initialize components
    processor = CaliforniaHousingProcessor()
    predictor = CaliforniaHousingPredictor()

    # Load and process data
    logger.info("Loading and processing California housing data...")
    raw_data_path = "data/raw/california_housing.csv"

    # Load the data (will fetch from sklearn if file doesn't exist)
    df = processor.load_data(raw_data_path)

    # Process data
    df_clean = processor.clean_data(df)
    df_engineered = processor.engineer_features(df_clean)
    X_train, X_test, y_train, y_test, feature_names = processor.preprocess_for_modeling(df_engineered)

    # Train models
    logger.info("Training models...")
    predictor.train_multiple_models(X_train, y_train, X_test, y_test)

    # Display results
    logger.info("\n" + "="*50)
    logger.info("MODEL PERFORMANCE SUMMARY")
    logger.info("="*50)

    summary_df = predictor.get_model_summary()
    print(summary_df.to_string(index=False))

    # Feature importance
    importance_df = predictor.get_feature_importance(feature_names)
    if importance_df is not None:
        logger.info(f"\nTop 10 Most Important Features for {predictor.best_model_name}:")
        print(importance_df.head(10).to_string(index=False))

    # Save the best model
    model_path = "models/california_housing_model.joblib"
    predictor.save_model(model_path)

    # Save the processor
    processor_path = "models/data_processor.joblib"
    os.makedirs(os.path.dirname(processor_path), exist_ok=True)
    joblib.dump(processor, processor_path)
    logger.info(f"Data processor saved to {processor_path}")

    # Test prediction with sample data
    sample_house = {
        'median_income': 5.0,
        'house_age': 10.0,
        'avg_rooms': 6.0,
        'avg_bedrooms': 1.2,
        'population': 3000.0,
        'avg_occupancy': 3.5,
        'latitude': 34.0,
        'longitude': -118.0
    }

    predicted_price = predictor.predict_single(sample_house, processor)
    logger.info(f"\nSample prediction for California house: ${predicted_price:,.2f}")

    logger.info("Model training completed successfully!")


if __name__ == "__main__":
    main()