"""
Data processing module for California housing price prediction.
Uses the classic California housing dataset from sklearn.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CaliforniaHousingProcessor:
    """
    A class to handle all data processing operations for California housing price prediction.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = []

    def load_california_housing_data(self, save_path=None):
        """
        Load the California housing dataset from sklearn.

        Args:
            save_path (str): Path to save the raw data

        Returns:
            pd.DataFrame: California housing data
        """
        # Load the dataset
        california_housing = fetch_california_housing(as_frame=True)

        # Combine features and target
        df = california_housing.data.copy()
        df['price'] = california_housing.target * 100000  # Convert to actual price scale

        # Add more descriptive column names
        column_mapping = {
            'MedInc': 'median_income',
            'HouseAge': 'house_age',
            'AveRooms': 'avg_rooms',
            'AveBedrms': 'avg_bedrooms',
            'Population': 'population',
            'AveOccup': 'avg_occupancy',
            'Latitude': 'latitude',
            'Longitude': 'longitude'
        }
        df = df.rename(columns=column_mapping)

        logger.info(f"California housing data loaded: {df.shape}")
        logger.info(f"Features: {list(df.columns[:-1])}")

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            df.to_csv(save_path, index=False)
            logger.info(f"Data saved to {save_path}")

        return df

    def load_data(self, file_path):
        """
        Load housing data from CSV file.

        Args:
            file_path (str): Path to the CSV file

        Returns:
            pd.DataFrame: Loaded data
        """
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Data loaded successfully from {file_path}")
            logger.info(f"Data shape: {df.shape}")
            return df
        except FileNotFoundError:
            logger.warning(f"File not found: {file_path}. Loading from sklearn instead.")
            return self.load_california_housing_data(save_path=file_path)
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def clean_data(self, df):
        """
        Clean the California housing data.

        Args:
            df (pd.DataFrame): Raw data

        Returns:
            pd.DataFrame: Cleaned data
        """
        df_clean = df.copy()

        # The California housing dataset is already quite clean, but let's add some validation
        initial_count = len(df_clean)

        # Remove any rows with missing values (though there shouldn't be any)
        df_clean = df_clean.dropna()

        # Remove extreme outliers using IQR method for price
        if 'price' in df_clean.columns:
            Q1 = df_clean['price'].quantile(0.01)  # Use 1st and 99th percentiles instead of IQR
            Q3 = df_clean['price'].quantile(0.99)
            df_clean = df_clean[(df_clean['price'] >= Q1) & (df_clean['price'] <= Q3)]

        # Remove outliers for median_income (extremely high values might be errors)
        if 'median_income' in df_clean.columns:
            income_99th = df_clean['median_income'].quantile(0.99)
            df_clean = df_clean[df_clean['median_income'] <= income_99th]

        # Ensure positive values for certain columns
        positive_columns = ['avg_rooms', 'avg_bedrooms', 'population', 'avg_occupancy']
        for col in positive_columns:
            if col in df_clean.columns:
                df_clean = df_clean[df_clean[col] > 0]

        removed_count = initial_count - len(df_clean)
        logger.info(f"Data cleaning completed. Removed {removed_count} rows ({removed_count/initial_count*100:.2f}%)")

        return df_clean

    def engineer_features(self, df):
        """
        Create new features from existing ones.

        Args:
            df (pd.DataFrame): Cleaned data

        Returns:
            pd.DataFrame: Data with engineered features
        """
        df_engineered = df.copy()

        # Bedroom ratio (proportion of bedrooms to total rooms)
        if 'avg_bedrooms' in df_engineered.columns and 'avg_rooms' in df_engineered.columns:
            df_engineered['bedroom_ratio'] = df_engineered['avg_bedrooms'] / df_engineered['avg_rooms']
            df_engineered['bedroom_ratio'] = df_engineered['bedroom_ratio'].fillna(0)

        # Rooms per person
        if 'avg_rooms' in df_engineered.columns and 'avg_occupancy' in df_engineered.columns:
            df_engineered['rooms_per_person'] = df_engineered['avg_rooms'] / df_engineered['avg_occupancy']
            df_engineered['rooms_per_person'] = df_engineered['rooms_per_person'].replace([np.inf, -np.inf], 0)

        # Population density (relative)
        if 'population' in df_engineered.columns:
            df_engineered['population_density'] = np.log1p(df_engineered['population'])

        # Income categories
        if 'median_income' in df_engineered.columns:
            df_engineered['income_category'] = pd.cut(
                df_engineered['median_income'],
                bins=[0, 2.5, 4.5, 6.0, float('inf')],
                labels=['Low', 'Medium', 'High', 'Very High']
            )

        # Age categories
        if 'house_age' in df_engineered.columns:
            df_engineered['age_category'] = pd.cut(
                df_engineered['house_age'],
                bins=[0, 10, 25, 40, float('inf')],
                labels=['New', 'Recent', 'Mature', 'Old']
            )

        # Location-based features (simplified geographic regions)
        if 'latitude' in df_engineered.columns and 'longitude' in df_engineered.columns:
            # Create rough geographic regions
            df_engineered['coastal'] = (df_engineered['longitude'] > -121.0).astype(int)
            df_engineered['northern'] = (df_engineered['latitude'] > 36.0).astype(int)

        # Price per income (affordability ratio) - only for training data
        if 'price' in df_engineered.columns and 'median_income' in df_engineered.columns:
            df_engineered['price_income_ratio'] = df_engineered['price'] / (df_engineered['median_income'] * 10000)

        logger.info("Feature engineering completed")
        logger.info(f"New features added: bedroom_ratio, rooms_per_person, population_density, income_category, age_category, coastal, northern")

        return df_engineered

    def preprocess_for_modeling(self, df, target_column='price', test_size=0.2, random_state=42):
        """
        Prepare data for machine learning modeling.

        Args:
            df (pd.DataFrame): Engineered data
            target_column (str): Name of the target column
            test_size (float): Proportion of test set
            random_state (int): Random state for reproducibility

        Returns:
            tuple: X_train, X_test, y_train, y_test, feature_names
        """
        # Separate features and target
        if target_column in df.columns:
            X = df.drop(columns=[target_column])
            y = df[target_column]
        else:
            X = df
            y = None

        # Handle categorical variables using one-hot encoding
        X_processed = pd.get_dummies(X, columns=['income_category', 'age_category'], drop_first=True)

        # Store feature names
        self.feature_columns = X_processed.columns.tolist()

        # Scale numerical features
        X_scaled = self.scaler.fit_transform(X_processed)
        X_scaled = pd.DataFrame(X_scaled, columns=self.feature_columns, index=X_processed.index)

        if y is not None:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=test_size, random_state=random_state, stratify=None
            )
            logger.info(f"Data split - Train: {X_train.shape}, Test: {X_test.shape}")
            return X_train, X_test, y_train, y_test, self.feature_columns
        else:
            return X_scaled, self.feature_columns

    def transform_new_data(self, df):
        """
        Transform new data using fitted preprocessors.

        Args:
            df (pd.DataFrame): New data to transform

        Returns:
            pd.DataFrame: Transformed data
        """
        # Apply the same preprocessing steps
        df_clean = self.clean_data(df)
        df_engineered = self.engineer_features(df_clean)

        # Remove target column if present
        if 'price' in df_engineered.columns:
            df_engineered = df_engineered.drop(columns=['price'])

        # Apply one-hot encoding (need to handle missing categories)
        df_processed = pd.get_dummies(df_engineered, columns=['income_category', 'age_category'], drop_first=True)

        # Ensure all feature columns are present
        for col in self.feature_columns:
            if col not in df_processed.columns:
                df_processed[col] = 0

        # Reorder columns to match training data
        df_processed = df_processed[self.feature_columns]

        # Scale features
        df_scaled = self.scaler.transform(df_processed)
        df_scaled = pd.DataFrame(df_scaled, columns=self.feature_columns, index=df_processed.index)

        return df_scaled


def main():
    """
    Main function to demonstrate the data processing pipeline.
    """
    processor = CaliforniaHousingProcessor()

    # Load California housing data
    raw_data_path = "data/raw/california_housing.csv"
    df = processor.load_california_housing_data(save_path=raw_data_path)

    # Display basic statistics
    logger.info(f"\nDataset Statistics:")
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Price range: ${df['price'].min():,.0f} - ${df['price'].max():,.0f}")
    logger.info(f"Median price: ${df['price'].median():,.0f}")

    # Process data
    df_clean = processor.clean_data(df)
    df_engineered = processor.engineer_features(df_clean)

    # Prepare for modeling
    X_train, X_test, y_train, y_test, feature_names = processor.preprocess_for_modeling(df_engineered)

    # Save processed data
    processed_data_path = "data/processed/processed_california_housing.csv"
    os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
    df_engineered.to_csv(processed_data_path, index=False)

    logger.info("Data processing completed successfully!")
    logger.info(f"Total features: {len(feature_names)}")
    logger.info(f"Training set size: {len(X_train)}")


if __name__ == "__main__":
    main()