import streamlit as st
import pandas as pd
import numpy as np
import keras
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

st.set_page_config(page_title="Wind Power Prediction", layout="wide")
st.title("Wind Power Prediction Dashboard")

# Define features based on Book1.xlsx
FEATURES = [
    'Wind Speed (m/s)', 'Theoretical_Power_Curve (KWh)', 'Wind Direction (°)', 
    'hour', 'day_of_week', 'month', 'day_of_year', 
    'wind_direction_sin', 'wind_direction_cos', 'power_deviation'
]

@st.cache_resource
def load_model(model_path):
    try:
        return keras.models.load_model(model_path) # type: ignore
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Sidebar Configuration
st.sidebar.header("⚙️ Configuration")
model_path = st.sidebar.text_input("Model Path", value="best_model.keras")
model = load_model(model_path)

if model is None:
    st.warning("⚠️ Please provide a valid model file in the sidebar.")
    st.stop()

# Input Selection
input_method = st.radio("Choose Input Method:", ("Upload CSV", "Manual Entry"))

X_final = None
y_actual = None

if input_method == "Upload CSV":
    uploaded_file = st.file_uploader("Upload wind data (CSV)", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        missing = [f for f in FEATURES if f not in df.columns]
        if missing:
            st.error(f"Missing features: {', '.join(missing)}")
        else:
            target_col = st.selectbox("Select Actual Power Column:", df.columns)
            # Use multiples of 144[cite: 1]
            num_seq = len(df) // 144
            if num_seq > 0:
                X_final = df[FEATURES].iloc[:num_seq*144].values.reshape(num_seq, 144, 10)
                y_actual = df[target_col].values[143::144][:num_seq]
                st.info(f"Loaded {num_seq} sequences of 144 steps.")
            else:
                st.error("CSV needs at least 144 rows.")

else:
    st.write("### Manual Entry")
    manual_data = {}
    cols = st.columns(2)
    for i, f in enumerate(FEATURES):
        with cols[i % 2]:
            manual_data[f] = st.number_input(f, value=0.0)
    
    # Simulate a sequence of 144 identical steps to satisfy the model[cite: 1]
    single_entry = np.array([list(manual_data.values())])
    X_final = np.repeat(single_entry[np.newaxis, :, :], 144, axis=1)

if st.button("Run Prediction"):
    if X_final is not None:
        try:
            preds = model.predict(X_final, verbose=0).flatten() # type: ignore
            
            if input_method == "Manual Entry":
                st.success(f"Predicted Power: {preds[0]:.2f} kW")
            else:
                st.success(f"Generated {len(preds)} predictions.")
                if y_actual is not None:
                    res = pd.DataFrame({'Actual': y_actual, 'Predicted': preds})
                    mae = mean_absolute_error(res['Actual'], res['Predicted'])
                    st.metric("MAE", f"{mae:.2f}")
                    st.line_chart(res)
        except Exception as e:
            st.error(f"Prediction Error: {e}")