import streamlit as st
import pandas as pd
import numpy as np
import keras
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Wind Power Prediction", layout="wide")
st.title("Wind Power Prediction Dashboard")

# Feature list based on Book1.xlsx metadata[cite: 1, 5]
FEATURES = [
    'Wind Speed (m/s)', 'Theoretical_Power_Curve (KWh)', 'Wind Direction (°)', 
    'hour', 'day_of_week', 'month', 'day_of_year', 
    'wind_direction_sin', 'wind_direction_cos', 'power_deviation'
]

@st.cache_resource
def load_assets():
    """Load model and the provided scaler with explicit type hints[cite: 1, 5]."""
    # Adding '-> keras.Model' and using a local variable helps clear Pylance errors
    loaded_model: keras.Model = keras.models.load_model('best_model.keras') # type: ignore
    loaded_scaler = joblib.load('scaler_X.pkl') 
    return loaded_model, loaded_scaler

# Ensure you are calling the function with parentheses () to get the Model object[cite: 3]
model, scaler = load_assets()

# Input Method Selection
input_method = st.radio("Choose Input Method:", ("Upload CSV", "Manual Entry"))

X_processed = None
y_actual = None

if input_method == "Upload CSV":
    uploaded_file = st.file_uploader("Upload wind data (CSV)", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        target_col = st.selectbox("Select Actual Power Column:", df.columns)
        
        num_seq = len(df) // 144
        if num_seq > 0:
            # Scaling: Apply your .pkl scaler to the 10 features[cite: 4, 5]
            X_raw = df[FEATURES].iloc[:num_seq * 144]
            X_scaled = scaler.transform(X_raw) 
            
            # Reshaping for LSTM[cite: 1]
            X_processed = X_scaled.reshape(num_seq, 144, 10)
            y_actual = df[target_col].values[143::144][:num_seq]
        else:
            st.error("CSV must have at least 144 rows[cite: 1].")

else:
    st.write("### Manual Entry")
    manual_data = {f: st.number_input(f, value=0.0) for f in FEATURES}
    
    # Scale and replicate for 144 steps[cite: 1, 4]
    # We use pd.DataFrame to ensure column names match what the scaler expects
    single_row = pd.DataFrame([manual_data])[FEATURES]
    single_scaled = scaler.transform(single_row)
    X_processed = np.repeat(single_scaled[np.newaxis, :, :], 144, axis=1)

if st.button("Run Prediction"):
    # Guard against 'None' to satisfy Pylance[cite: 3]
    if model is not None and X_processed is not None:
        try:
            # Generate raw prediction
            raw_preds = model.predict(X_processed, verbose="0")
            
            # Rescaling Back: Multiplying by the max power (approx. 3600)
            predictions = np.array(raw_preds).flatten() * 3600 
            
            if input_method == "Manual Entry":
                st.success(f"Rescaled Predicted Power: {predictions[0]:.2f} kW")
            else:
                res_df = pd.DataFrame({'Actual': y_actual, 'Predicted': predictions})
                mae = mean_absolute_error(res_df['Actual'], res_df['Predicted'])
                st.metric("MAE (Rescaled)", f"{mae:.2f}")
                st.line_chart(res_df)
        except Exception as e:
            st.error(f"Prediction Error: {e}")