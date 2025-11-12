import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import plotly.graph_objects as go
import os

# ------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------
st.set_page_config(
    page_title="AI Sales Forecasting Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------
# LOAD MODEL AND SCALER
# ------------------------------------------------------------
MODEL_PATH = "lstm_model.keras"
SCALER_PATH = "scaler.pkl"

model, scaler = None, None

try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    scaler = joblib.load(SCALER_PATH)
    st.success("Model and Scaler loaded successfully!")
except Exception as e:
    st.error(f"Error loading model/scaler: {e}")

# ------------------------------------------------------------
# SIDEBAR INPUT PARAMETERS
# ------------------------------------------------------------
st.sidebar.title("Input Parameters")
st.sidebar.write("Adjust parameters below to predict future sales:")

price = st.sidebar.number_input("Product Price", min_value=0.0, value=250.0)
freight = st.sidebar.number_input("Freight Value", min_value=0.0, value=50.0)
payment = st.sidebar.number_input("Payment Value", min_value=0.0, value=500.0)

# ------------------------------------------------------------
# MAIN PAGE CONTENT
# ------------------------------------------------------------
st.title("AI-Powered Sales Forecasting Dashboard")
st.markdown("Predict and visualize business growth trends using AI-based forecasting models.")

# ------------------------------------------------------------
# SALES PREDICTION SECTION
# ------------------------------------------------------------
st.subheader("Sales Prediction")

predicted_value = None

if st.button("Predict Sales") and model and scaler:
    try:
        input_data = np.array([[price, freight, payment]])
        scaled_input = scaler.transform(input_data)
        reshaped_input = scaled_input.reshape((1, 1, scaled_input.shape[1]))

        prediction_scaled = model.predict(reshaped_input)
        padding = np.zeros((1, scaled_input.shape[1] - 1))
        padded_output = np.concatenate([prediction_scaled, padding], axis=1)
        predicted_value = scaler.inverse_transform(padded_output)[0][0]

        st.success(f"Predicted Sales Value: ₹ {predicted_value:.2f}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ------------------------------------------------------------
# DATA UPLOAD SECTION
# ------------------------------------------------------------
st.subheader("Data Insights Visualization")
uploaded_file = st.file_uploader("Upload a dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"File uploaded successfully: {uploaded_file.name}")
        st.dataframe(df.head())

        # Try to find a valid sales-related column or compute one
        sales_col = None
        possible_sales_cols = ["total_amount", "payment_value", "sales_value", "revenue", "amount"]

        for col in possible_sales_cols:
            if col in df.columns:
                sales_col = col
                break

        # If no direct sales column, create one if price + freight_value exist
        if sales_col is None and all(c in df.columns for c in ["price", "freight_value"]):
            df["total_sales"] = df["price"] + df["freight_value"]
            sales_col = "total_sales"
            st.info("Computed 'total_sales' from price + freight_value.")

        if sales_col:
            st.write(f"Detected sales column: **{sales_col}**")

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=df[sales_col].head(1000),  # limit for performance
                mode="lines+markers",
                name="Actual Sales",
                line=dict(color="lightgreen", width=2)
            ))

            if predicted_value is not None:
                fig.add_trace(go.Scatter(
                    x=[len(df)],
                    y=[predicted_value],
                    mode="markers+text",
                    text=["Predicted"],
                    textposition="top center",
                    name="Predicted Sales",
                    marker=dict(color="magenta", size=10)
                ))

            fig.update_layout(
                title="Actual vs Predicted Sales",
                xaxis_title="Order Index",
                yaxis_title="Sales Value (₹)",
                template="plotly_dark",
                height=450
            )
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.warning(
                "No valid sales-related column found. "
                "Try uploading a dataset containing either 'total_amount', 'payment_value', 'price', or 'freight_value'."
            )

    except Exception as e:
        st.error(f"Error loading or visualizing file: {e}")
else:
    st.info("Upload a CSV file to view the visualization.")

# ------------------------------------------------------------
# ABOUT SECTION
# ------------------------------------------------------------
st.markdown("---")
st.subheader("About this Project")

st.markdown("""
AI-Powered Sales Forecasting Dashboard uses an LSTM neural network  
to predict and visualize future sales based on historical data.

Frameworks: TensorFlow · Streamlit · Pandas · Plotly  
Developer: Prathmesh Yadav  
Version: 2.0  
""")

st.markdown("""
<hr style="border: 0.5px solid #666;">
<div style="text-align:center; font-size:13px; color:#999;">
© 2025 Prathmesh Yadav | AI-Powered Sales Analysis Dashboard
</div>
""", unsafe_allow_html=True)
