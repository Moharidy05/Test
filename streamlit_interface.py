import streamlit as st
import pandas as pd
import numpy as np
from tensorflow import keras
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Wind Power Prediction", layout="wide")
st.title("Wind Power Prediction Dashboard")

# Define features based on Book1.xlsx
FEATURES = [
    'Wind Speed (m/s)', 'Theoretical_Power_Curve (KWh)', 'Wind Direction (°)', 
    'hour', 'day_of_week', 'month', 'day_of_year', 
    'wind_direction_sin', 'wind_direction_cos', 'power_deviation'
]

@st.cache_resource
def load_model():
    """Load the pre-trained LSTM model."""
    try:
        return keras.models.load_model('best_model.keras')
    except FileNotFoundError:
        st.error("Model file 'best_model.keras' not found. Please ensure it exists in the working directory.")
        return None

model = load_model()

if model is None:
    st.stop()

# Input Method Selection
input_method = st.radio("Choose Input Method:", ("Upload CSV", "Manual Entry"))

X_input = pd.DataFrame()
y_actual = None

if input_method == "Upload CSV":
    uploaded_file = st.file_uploader("Upload wind data (CSV)", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Validate that all required features exist
            missing_features = [f for f in FEATURES if f not in df.columns]
            if missing_features:
                st.error(f"Missing required features: {', '.join(missing_features)}")
            else:
                target_col = st.selectbox("Select Actual Power Column (for evaluation):", df.columns)
                X_input = df[FEATURES]
                y_actual = df[target_col].values
                st.write("Data Preview:", X_input.head())
        except Exception as e:
            st.error(f"Error reading CSV: {e}")

else:
    st.write("### Enter Feature Values")
    st.info("Enter values for a single data point to make a prediction.")
    
    manual_data = {}
    cols = st.columns(2)
    for i, feature in enumerate(FEATURES):
        with cols[i % 2]:
            manual_data[feature] = st.number_input(f"{feature}", value=0.0)
    
    # Create a single row DataFrame
    X_input = pd.DataFrame([manual_data])

if st.button("Run Prediction"):
    if X_input.empty:
        st.error("No input data provided. Please upload a CSV or enter values manually.")
    else:
        try:
            # Handle single entry (manual) or multiple entries (CSV)
            if input_method == "Manual Entry":
                # For manual entry, reshape single row to (1, 1, 10) for LSTM
                X_reshaped = X_input.values.reshape(1, 1, len(FEATURES))
                raw_preds = model.predict(X_reshaped, verbose=0)
                predictions = np.asarray(raw_preds).flatten()
                
                st.success(f"Predicted Power: {predictions[0]:.2f} kW")
                
            else:
                # For CSV upload, maintain the 144-sequence logic
                num_sequences = len(X_input) // 144
                
                if num_sequences == 0:
                    st.error("Insufficient data. CSV requires 144+ rows for prediction.")
                else:
                    X_trimmed = X_input.iloc[:num_sequences * 144]
                    X_reshaped = X_trimmed.values.reshape(num_sequences, 144, len(FEATURES))
                    
                    # Make predictions
                    raw_preds = model.predict(X_reshaped, verbose=0)
                    predictions = np.asarray(raw_preds).flatten()
                    
                    st.success(f"Predicted Power Values: {len(predictions)} predictions made")
                    
                    if y_actual is not None and len(y_actual) >= num_sequences:
                        # Align actuals with predictions (take every 144th value starting from index 143)
                        y_matched = y_actual[143::144][:num_sequences]
                        
                        if len(y_matched) == len(predictions):
                            df_res = pd.DataFrame({'Actual': y_matched, 'Predicted': predictions})
                            
                            mae = mean_absolute_error(df_res['Actual'], df_res['Predicted'])
                            mse = mean_squared_error(df_res['Actual'], df_res['Predicted'])
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Mean Absolute Error (MAE)", f"{mae:.2f}")
                            with col2:
                                st.metric("Mean Squared Error (MSE)", f"{mse:.2f}")
                            
                            st.line_chart(df_res)
                        else:
                            st.warning("Could not align actual and predicted values properly.")
                    else:
                        st.info("No actual values provided for comparison.")

        except Exception as e:
            st.error(f"Error during prediction: {e}")