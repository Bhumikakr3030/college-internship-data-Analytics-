import streamlit as st

import joblib
import pandas as pd

rescaler = joblib.load('models/serial_minmax_scaler.pkl')
encoder = joblib.load('models/serial_ordinal_encoder.pkl')
model = joblib.load('models/serial_rf_regressor.pkl')

carat = st.number_input("Enter Carat", value=None, placeholder="Type a number...")
cut = st.selectbox("Enter Cut", ('Ideal', 'Premium', 'Good', 'Very Good', 'Fair'), index=None, placeholder="Select Cut...",)
color = st.selectbox("Enter Color", ('E', 'I', 'J', 'H', 'F', 'G', 'D'), index=None, placeholder="Select Color...",)
clarity = st.selectbox("Enter Clarity", ('SI2', 'SI1', 'VS1', 'VS2', 'VVS2', 'VVS1', 'I1', 'IF'), index=None, placeholder="Select Clarity...",)

btn_click = st.button("Predict")

if btn_click:
    data = {
        'carat': [carat],
        'cut': [cut],
        'color': [color],
        'clarity': [clarity]
    }

    data_point = pd.DataFrame(data)

    num_features = ['carat']
    cat_features = ['cut', 'color', 'clarity']

    data_point[cat_features] = encoder.transform(data_point[cat_features])
    data_point[num_features] = rescaler.transform(data_point[num_features])

    prediction = model.predict(data_point)

    st.write("💎 Predicted Diamond Price:", prediction[0])
