import streamlit as st
import pandas as pd
import joblib

def clean_input_data(X):
    X = X.copy()
    if 'cargo_volume_l' in X.columns:
        X['cargo_volume_l'] = pd.to_numeric(X['cargo_volume_l'], errors='coerce')
   
    cats = ['drivetrain', 'segment', 'car_body_type', 'brand', 'model']
    for col in cats:
        if col in X.columns:
            X[col] = X[col].astype(str)
            
    return X


@st.cache_resource
def load_pipeline():
    return joblib.load('ev_range_prediction_pipeline.joblib')

model_pipeline = load_pipeline()

st.title("Electric Vehicle Range Prediction Tool")
st.write("Input full vehicle specifications to compute estimated driving range.")

# Group inputs into UI columns for a clean layout
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Vehicle Identity")
    brand = st.text_input("Brand", value="Tesla")
    model_name = st.text_input("Model", value="Model 3")
    car_body_type = st.selectbox("Body Type", ["Sedan", "SUV", "Hatchback", "Coupe", "Wagon"])
    segment = st.selectbox("Segment", ["A", "B", "C", "D", "E", "F", "J"])
    drivetrain = st.selectbox("Drivetrain", ["AWD", "FWD", "RWD"])

with col2:
    st.subheader("Battery & Performance")
    battery_capacity_kWh = st.number_input("Battery Capacity (kWh)", min_value=5.0, max_value=250.0, value=60.0, step=1.0)
    efficiency_wh_per_km = st.number_input("Efficiency (Wh/km)", min_value=50.0, max_value=500.0, value=160.0, step=5.0)
    fast_charging_power_kw_dc = st.number_input("Fast Charge Power (kW DC)", min_value=0.0, max_value=400.0, value=150.0, step=5.0)
    acceleration_0_100_s = st.number_input("0-100 km/h Acceleration (s)", min_value=1.5, max_value=20.0, value=5.5, step=0.1)
    top_speed_kmh = st.number_input("Top Speed (km/h)", min_value=80.0, max_value=350.0, value=200.0, step=5.0)
    torque_nm = st.number_input("Torque (Nm)", min_value=50.0, max_value=1500.0, value=350.0, step=5.0)

with col3:
    st.subheader("Dimensions & Capacity")
    length_mm = st.number_input("Length (mm)", min_value=2000.0, max_value=6000.0, value=4500.0, step=10.0)
    width_mm = st.number_input("Width (mm)", min_value=1000.0, max_value=2500.0, value=1800.0, step=10.0)
    height_mm = st.number_input("Height (mm)", min_value=1000.0, max_value=2500.0, value=1500.0, step=10.0)
    seats = st.number_input("Seats", min_value=1, max_value=9, value=5, step=1)
    number_of_cells = st.number_input("Number of Cells", min_value=10, max_value=10000, value=4000, step=100)
    towing_capacity_kg = st.number_input("Towing Capacity (kg)", min_value=0.0, max_value=5000.0, value=1000.0, step=50.0)
    cargo_volume_l = st.number_input("Cargo Volume (L)", min_value=50.0, max_value=3000.0, value=450.0, step=10.0)

if st.button("Predict Range"):
    # Calculate custom engineered features
    volumetric_footprint = length_mm * width_mm * height_mm
    battery_to_torque_ratio = battery_capacity_kWh / torque_nm if torque_nm > 0 else 0.0

    # Build DataFrame matching all exact column names from training data
    input_df = pd.DataFrame([{
        'brand': brand,
        'model': model_name,
        'car_body_type': car_body_type,
        'segment': segment,
        'drivetrain': drivetrain,
        'battery_capacity_kWh': battery_capacity_kWh,
        'efficiency_wh_per_km': efficiency_wh_per_km,
        'fast_charging_power_kw_dc': fast_charging_power_kw_dc,
        'acceleration_0_100_s': acceleration_0_100_s,
        'top_speed_kmh': top_speed_kmh,
        'torque_nm': torque_nm,
        'number_of_cells': number_of_cells,
        'towing_capacity_kg': towing_capacity_kg,
        'cargo_volume_l': cargo_volume_l,
        'seats': seats,
        'length_mm': length_mm,
        'width_mm': width_mm,
        'height_mm': height_mm,
        'volumetric_footprint': volumetric_footprint,
        'battery_to_torque_ratio': battery_to_torque_ratio
    }])

    # Predict using the exported joblib pipeline
    prediction = model_pipeline.predict(input_df)[0]
    st.success(f"Predicted EV Range: {prediction:.2f} km")