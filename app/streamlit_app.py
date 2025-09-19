import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append('src')

st.title("California Housing Price Predictor")

# Simple data loading function
@st.cache_data
def load_data():
    try:
        from data_processing import CaliforniaHousingProcessor
        processor = CaliforniaHousingProcessor()
        df = processor.load_california_housing_data()
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Load and display data
df = load_data()

if df is not None:
    st.success(f"Data loaded successfully! {len(df)} districts")

    # Basic statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Districts", len(df))
    with col2:
        avg_price = df['price'].mean()
        st.metric("Average Price", f"${avg_price:,.0f}")
    with col3:
        avg_income = df['median_income'].mean()
        st.metric("Average Income", f"${avg_income*10000:,.0f}")

    # Show data sample
    st.subheader("Data Sample")
    st.dataframe(df.head())

    # Simple prediction form
    st.subheader("Price Prediction")

    with st.form("predict"):
        col1, col2 = st.columns(2)

        with col1:
            median_income = st.slider("Median Income (tens of thousands)", 0.5, 15.0, 5.0)
            house_age = st.slider("House Age (years)", 1, 52, 10)
            avg_rooms = st.slider("Average Rooms", 2.0, 15.0, 6.0)
            avg_bedrooms = st.slider("Average Bedrooms", 0.5, 5.0, 1.2)

        with col2:
            population = st.number_input("Population", 100, 50000, 3000)
            avg_occupancy = st.slider("Average Occupancy", 1.0, 10.0, 3.5)
            latitude = st.slider("Latitude", 32.0, 42.0, 34.0)
            longitude = st.slider("Longitude", -125.0, -114.0, -118.0)

        submitted = st.form_submit_button("Predict Price")

        if submitted:
            # Try to load and use model
            try:
                from model_training import CaliforniaHousingPredictor
                import joblib

                # Check if model exists
                if os.path.exists("models/california_housing_model.joblib") and os.path.exists("models/data_processor.joblib"):
                    # Load models
                    predictor = CaliforniaHousingPredictor()
                    predictor.load_model("models/california_housing_model.joblib")
                    processor = joblib.load("models/data_processor.joblib")

                    # Make prediction
                    input_data = {
                        'median_income': median_income,
                        'house_age': house_age,
                        'avg_rooms': avg_rooms,
                        'avg_bedrooms': avg_bedrooms,
                        'population': population,
                        'avg_occupancy': avg_occupancy,
                        'latitude': latitude,
                        'longitude': longitude
                    }

                    predicted_price = predictor.predict_single(input_data, processor)
                    st.success(f"Predicted Price: ${predicted_price:,.2f}")

                else:
                    st.warning("Models not found. Please train the model first.")

            except Exception as e:
                st.error(f"Error making prediction: {e}")
                # Fallback simple calculation
                simple_price = (median_income * 30000 +
                              avg_rooms * 10000 +
                              (52 - house_age) * 1000)
                st.info(f"Fallback estimate: ${simple_price:,.2f}")

else:
    st.error("Failed to load data")