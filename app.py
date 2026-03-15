import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# Load model
model = load_model("sales_lstm_model.keras")

# Load scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Page config
st.set_page_config(page_title="Sales Forecasting", page_icon="📈")

# Title
st.title("📈 Sales Forecasting (LSTM)")
st.write("Enter the last 30 sales values to predict the next sales value.")

# Input box
user_input = st.text_area(
    "Enter 30 values separated by commas",
    placeholder="41, 53, 39, 35, 51, ..."
)

# Prediction button
if st.button("Predict Next Sales"):

    if user_input.strip() == "":
        st.warning("Please enter values.")
    else:
        try:
            values = [float(x.strip()) for x in user_input.split(",")]

            if len(values) != 30:
                st.error("You must enter exactly 30 values.")
            else:
                input_data = np.array(values).reshape(-1, 1)

                # Scale input
                scaled_input = scaler.transform(input_data)
                X_input = scaled_input.reshape(1, 30, 1)

                # Predict
                prediction = model.predict(X_input, verbose=0)
                predicted_value = scaler.inverse_transform(prediction)[0][0]

                st.success(f"Predicted Next Sales Value: {predicted_value:.2f}")

        except:
            st.error("Invalid input. Please enter numeric values only.")