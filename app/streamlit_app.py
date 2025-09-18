import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import sys

# Add src directory to path for imports
sys.path.append('src')
from data_processing import CaliforniaHousingProcessor
from model_training import CaliforniaHousingPredictor

# Page configuration
st.set_page_config(
    page_title="California Housing Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1e3d59;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .prediction-result {
        font-size: 2rem;
        color: #28a745;
        text-align: center;
        padding: 1rem;
        background-color: #e8f5e8;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load the California housing dataset."""
    try:
        if os.path.exists("data/processed/processed_california_housing.csv"):
            df = pd.read_csv("data/processed/processed_california_housing.csv")
        elif os.path.exists("data/raw/california_housing.csv"):
            df = pd.read_csv("data/raw/california_housing.csv")
        else:
            # Load data directly from sklearn
            processor = CaliforniaHousingProcessor()
            df = processor.load_california_housing_data()
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

@st.cache_resource
def load_models():
    """Load the trained models and processor."""
    try:
        # Load the predictor model
        predictor = CaliforniaHousingPredictor()
        if os.path.exists("models/california_housing_model.joblib"):
            predictor.load_model("models/california_housing_model.joblib")
        else:
            st.warning("Pre-trained model not found. Please train the model first.")
            return None, None

        # Load the data processor
        if os.path.exists("models/data_processor.joblib"):
            processor = joblib.load("models/data_processor.joblib")
        else:
            st.warning("Data processor not found. Please train the model first.")
            return None, None

        return predictor, processor
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None

def main():
    """Main application function."""
    # Title
    st.markdown('<h1 class="main-header">🏠 California Housing Price Predictor</h1>', unsafe_allow_html=True)

    # Load data and models
    df = load_data()
    predictor, processor = load_models()

    if df is None:
        st.error("Failed to load data. Please check your data files.")
        return

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["Home", "Price Prediction", "Data Exploration", "Model Performance"]
    )

    if page == "Home":
        show_home_page(df, predictor)
    elif page == "Price Prediction":
        show_prediction_page(predictor, processor)
    elif page == "Data Exploration":
        show_exploration_page(df)
    elif page == "Model Performance":
        show_model_performance(predictor, df)

def show_home_page(df, predictor):
    """Display the home page with overview information."""
    st.write("## Welcome to the California Housing Price Predictor!")

    st.write("""
    This application uses machine learning to predict housing prices in California based on the classic
    California housing dataset. Our model analyzes factors such as median income, location, house age,
    and neighborhood characteristics to provide accurate price estimates.
    """)

    # Key statistics
    if df is not None:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Districts", f"{len(df):,}")

        with col2:
            avg_price = df['price'].mean() if 'price' in df.columns else 0
            st.metric("Average Price", f"${avg_price:,.0f}")

        with col3:
            avg_income = df['median_income'].mean() if 'median_income' in df.columns else 0
            st.metric("Average Income", f"${avg_income*10000:,.0f}")

        with col4:
            if predictor and predictor.evaluation_results:
                best_r2 = max([result['test_r2'] for result in predictor.evaluation_results.values()])
                st.metric("Model Accuracy (R²)", f"{best_r2:.3f}")

    # Feature overview
    st.write("### Key Features Used in Prediction:")

    feature_descriptions = {
        "Median Income": "Median household income in the district (in tens of thousands)",
        "House Age": "Median age of houses in the district",
        "Room Statistics": "Average rooms and bedrooms per household",
        "Population": "Total population and occupancy rates in the district",
        "Location": "Latitude and longitude coordinates (coastal vs inland)",
        "Engineered Features": "Bedroom ratios, population density, and regional indicators"
    }

    for feature, description in feature_descriptions.items():
        st.write(f"**{feature}**: {description}")

    # Quick stats chart
    if df is not None and 'price' in df.columns:
        st.write("### California Housing Price Distribution")
        fig = px.histogram(df, x='price', nbins=50, title="Distribution of Housing Prices in California")
        fig.update_layout(xaxis_title="Price ($)", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)
def show_prediction_page(predictor, processor):
    """Display the price prediction interface."""
    st.write("## 🔮 Predict California Housing Price")

    if predictor is None or processor is None:
        st.error("Models not loaded. Please ensure the model has been trained.")
        return

    st.write("Enter the district characteristics below to get a median housing price prediction:")

    # Create input form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.write("### Economic & Demographic Data")
            median_income = st.slider("Median Income (tens of thousands)", min_value=0.5, max_value=15.0, value=5.0, step=0.1)
            population = st.number_input("Population", min_value=100, max_value=50000, value=3000, step=100)
            avg_occupancy = st.slider("Average Occupancy (people per household)", min_value=1.0, max_value=10.0, value=3.5, step=0.1)

        with col2:
            st.write("### Housing Characteristics")
            house_age = st.slider("Median House Age (years)", min_value=1, max_value=52, value=10)
            avg_rooms = st.slider("Average Rooms per House", min_value=2.0, max_value=15.0, value=6.0, step=0.1)
            avg_bedrooms = st.slider("Average Bedrooms per House", min_value=0.5, max_value=5.0, value=1.2, step=0.1)

        st.write("### Location")
        col3, col4 = st.columns(2)
        with col3:
            latitude = st.slider("Latitude", min_value=32.0, max_value=42.0, value=34.0, step=0.1)
        with col4:
            longitude = st.slider("Longitude", min_value=-125.0, max_value=-114.0, value=-118.0, step=0.1)

        # Submit button
        submitted = st.form_submit_button("Predict Price", type="primary")

        if submitted:
            # Prepare input data
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

            try:
                # Make prediction
                predicted_price = predictor.predict_single(input_data, processor)

                # Display result
                st.markdown(f"""
                <div class="prediction-result">
                    Predicted Median House Value: ${predicted_price:,.2f}
                </div>
                """, unsafe_allow_html=True)

                # Additional insights
                st.write("### District Analysis")

                # Income-to-price ratio
                price_income_ratio = predicted_price / (median_income * 10000)
                st.write(f"**Price-to-Income Ratio**: {price_income_ratio:.1f}x")

                # Room analysis
                bedroom_ratio = avg_bedrooms / avg_rooms
                st.write(f"**Bedroom Ratio**: {bedroom_ratio:.2f} ({bedroom_ratio*100:.0f}% of rooms are bedrooms)")

                # Location analysis
                is_coastal = longitude > -121.0
                is_northern = latitude > 36.0

                location_desc = "Northern" if is_northern else "Southern"
                location_desc += " California, "
                location_desc += "Coastal region" if is_coastal else "Inland region"

                st.write(f"**Location**: {location_desc}")

                # Provide context
                if predicted_price < 100000:
                    st.info("💡 This district has below-average housing prices for California.")
                elif predicted_price > 400000:
                    st.info("💡 This district has premium housing prices, likely in a desirable area.")
                else:
                    st.info("💡 This district has moderate housing prices for California.")

                # Show input summary
                with st.expander("Input Summary"):
                    st.json(input_data)

            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")

def show_exploration_page(df):
    """Display data exploration and visualizations."""
    st.write("## 📊 Data Exploration")

    if df is None:
        st.error("No data available for exploration.")
        return

    # Dataset overview
    st.write("### Dataset Overview")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Features", len(df.columns))
    with col3:
        missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        st.metric("Missing Data %", f"{missing_pct:.2f}%")

    # Data sample
    st.write("### Data Sample")
    st.dataframe(df.head(), use_container_width=True)

    # Visualizations
    st.write("### Visualizations")

    # Price distribution
    if 'price' in df.columns:
        col1, col2 = st.columns(2)

        with col1:
            fig = px.histogram(df, x='price', nbins=30, title="Price Distribution")
            fig.update_layout(xaxis_title="Price ($)", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.box(df, y='price', title="Price Box Plot")
            fig.update_layout(yaxis_title="Price ($)")
            st.plotly_chart(fig, use_container_width=True)

    # Correlation with price
    if 'price' in df.columns:
        st.write("### Feature Correlations with Price")

        numerical_cols = df.select_dtypes(include=[np.number]).columns
        correlations = df[numerical_cols].corr()['price'].sort_values(ascending=False)

        # Remove price correlation with itself
        correlations = correlations.drop('price')

        fig = px.bar(x=correlations.values, y=correlations.index, orientation='h',
                     title="Correlation with Price")
        fig.update_layout(xaxis_title="Correlation Coefficient", yaxis_title="Feature")
        st.plotly_chart(fig, use_container_width=True)

    # Interactive scatter plots
    st.write("### Interactive Scatter Plots")

    numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'price' in numerical_columns and len(numerical_columns) > 1:
        col1, col2 = st.columns(2)

        with col1:
            x_axis = st.selectbox("X-axis", numerical_columns, index=0)
        with col2:
            y_axis = st.selectbox("Y-axis", numerical_columns,
                                index=numerical_columns.index('price') if 'price' in numerical_columns else 1)

        if x_axis != y_axis:
            fig = px.scatter(df, x=x_axis, y=y_axis, title=f"{y_axis} vs {x_axis}")
            if 'neighborhood_type' in df.columns:
                fig = px.scatter(df, x=x_axis, y=y_axis, color='neighborhood_type',
                               title=f"{y_axis} vs {x_axis} by Neighborhood Type")
            st.plotly_chart(fig, use_container_width=True)

def show_model_performance(predictor, df):
    """Display model performance metrics and analysis."""
    st.write("## 🎯 Model Performance")

    if predictor is None or not predictor.evaluation_results:
        st.error("No model performance data available. Please train the model first.")
        return

    # Model comparison
    st.write("### Model Comparison")

    summary_df = predictor.get_model_summary()
    if summary_df is not None:
        # Format the dataframe for better display
        display_df = summary_df.copy()
        display_df['Test R²'] = display_df['Test R²'].round(4)
        display_df['Test RMSE'] = display_df['Test RMSE'].round(2)
        display_df['Test MAE'] = display_df['Test MAE'].round(2)
        display_df['CV R² Mean'] = display_df['CV R² Mean'].round(4)
        display_df['CV R² Std'] = display_df['CV R² Std'].round(4)

        st.dataframe(display_df, use_container_width=True)

        # Best model highlight
        best_model = display_df[display_df['Is Best']]['Model'].iloc[0]
        best_r2 = display_df[display_df['Is Best']]['Test R²'].iloc[0]
        st.success(f"🏆 Best Model: **{best_model}** with R² Score: **{best_r2:.4f}**")

    # Feature importance
    if predictor.best_model_name and hasattr(predictor, 'feature_importance'):
        st.write("### Feature Importance")

        # Get feature names from processor
        if df is not None:
            try:
                # This is a simplified approach - in production, you'd store feature names
                numerical_cols = ['square_feet', 'bedrooms', 'bathrooms', 'year_built',
                                'neighborhood_score', 'has_garage', 'has_garden',
                                'distance_to_school', 'distance_to_transport']

                if predictor.feature_importance is not None and len(predictor.feature_importance) > 0:
                    # Create feature importance plot
                    feature_names = numerical_cols[:len(predictor.feature_importance)]
                    importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': predictor.feature_importance
                    }).sort_values('Importance', ascending=True)

                    fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                               title=f"Feature Importance - {predictor.best_model_name}")
                    st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.warning(f"Could not display feature importance: {str(e)}")

    # Performance metrics explanation
    st.write("### Metrics Explanation")

    metrics_info = {
        "R² Score": "Coefficient of determination. Values closer to 1.0 indicate better fit.",
        "RMSE": "Root Mean Square Error. Lower values indicate better predictions.",
        "MAE": "Mean Absolute Error. Average absolute difference between predictions and actual values.",
        "CV Score": "Cross-validation score. Indicates model stability across different data splits."
    }

    for metric, explanation in metrics_info.items():
        st.write(f"**{metric}**: {explanation}")

if __name__ == "__main__":
    main()