"""
This module implements a Streamlit web application for estimating the 
annual spending of a customer based on various input features using
a machine learning model.
"""

# Import necessary libraries
from os.path import normpath

import pandas as pd
import streamlit as st
from PIL import Image

from src.components.model_prediction import ModelPrediction

# Configure the Streamlit page
st.set_page_config(
    page_title="Spend Estimator",
    page_icon="💰",
    layout="wide",
    menu_items=None,
    initial_sidebar_state="collapsed",
)

st.title("Customer Annual Spend Estimator")

# Define paths for images
main_image = Image.open(normpath("./resources/app-cover-image.png"))

# Create two columns for layout
col1, col2 = st.columns([0.45, 0.55], gap="medium")

# Populate col1 with image and project information
with col1:
    st.image(main_image, use_column_width=True)
    st.write(
        """The application utilizes a machine learning model
        to estimate the annual spending of a customer based on a range of input features
        """
    )
    st.write(
        """To obtain the desired outcome, input the appropriate values into
        the designated field.
        """
    )

# working on col2 section
with col2:
    with st.form("user_inputs"):
        avg_session_length = st.slider(
            label="Avg. Session Length:", min_value=1.00, max_value=100.00, value=30.00
        )
        time_on_app = st.slider(
            label="Time on App:", min_value=1.00, max_value=100.00, value=15.00
        )
        time_on_website = st.slider(
            label="Time on Website:", min_value=1.00, max_value=100.00, value=35.00
        )
        membership_length = st.slider(
            label="Length of Membership:", min_value=0.1, max_value=15.00, value=3.00
        )

        if st.form_submit_button():
            user_data = [
                {
                    "avg_session_length": avg_session_length,
                    "time_on_app": time_on_app,
                    "time_on_website": time_on_website,
                    "membership_length": membership_length,
                }
            ]
            user_df = pd.DataFrame(user_data)
            model_pred = ModelPrediction()
            est_annual_spend = round(model_pred.predict(user_df)[0], 2)
            st.write(
                f"With the given input features, the annual spent of the customer"
                f" as estimated by the model is: **:green[${est_annual_spend:0.2f}]**"
            )
