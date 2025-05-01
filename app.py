import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load trained model
with open('trained_model.pkl', 'rb') as f:
    model = pickle.load(f)
    trained_feature_names = model.feature_names_in_

st.title("🧠 Mental Health Treatment Prediction App")
st.write("This app predicts whether a person is likely to seek mental health treatment based on key workplace and personal factors.")

# Feature Inputs
st.header("📝 Please Fill the Following Details")

work_interfere = st.selectbox("How often does your mental health interfere with work?",
                              ['Never', 'Rarely', 'Sometimes', 'Often', "Don't know", "NA"])

family_history = st.selectbox("Do you have a family history of mental illness?", ['Yes', 'No'])
benefits = st.selectbox("Does your employer provide mental health benefits?", ['Yes', 'No'])
care_options = st.selectbox("Does your employer provide mental health care options?", ['Yes', 'No'])
country = st.selectbox("Country", ['India', 'United States', 'Canada', 'Other'])
state = st.selectbox("State (if in the U.S.)", ['TN', 'MD', 'Other'])

# Map Inputs to One-Hot Encoding Format used in Model
def preprocess_input():
    # Initialize all known feature columns to 0
    input_dict = {
        'work_interfere_Sometimes': 0,
        'work_interfere_Often': 0,
        'work_interfere_Rarely': 0,
        'work_interfere_Unknown': 0,
        'family_history_Yes': 0,
        'benefits_Yes': 0,
        'Country_India': 0,
        'care_options_Yes': 0,
        'state_MD': 0,
        'state_TN': 0
    }

    # Fill the relevant columns
    if work_interfere == 'Sometimes':
        input_dict['work_interfere_Sometimes'] = 1
    elif work_interfere == 'Often':
        input_dict['work_interfere_Often'] = 1
    elif work_interfere == 'Rarely':
        input_dict['work_interfere_Rarely'] = 1
    elif work_interfere in ["Don't know", "NA"]:
        input_dict['work_interfere_Unknown'] = 1

    if family_history == 'Yes':
        input_dict['family_history_Yes'] = 1
    if benefits == 'Yes':
        input_dict['benefits_Yes'] = 1
    if care_options == 'Yes':
        input_dict['care_options_Yes'] = 1
    if country == 'India':
        input_dict['Country_India'] = 1
    if state == 'MD':
        input_dict['state_MD'] = 1
    elif state == 'TN':
        input_dict['state_TN'] = 1

    # Create DataFrame with only used features
    input_df = pd.DataFrame([input_dict])

    # Add missing columns with 0
    for col in trained_feature_names:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder columns to match training
    input_df = input_df[trained_feature_names]

    return input_df

# Predict
if st.button("🚀 Predict Treatment Need"):
    input_df = preprocess_input()
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]  # Probability for class 1

    st.subheader("📊 Prediction Result:")
    if prediction == 1:
        st.error(f"⛔ The model predicts this person is **likely to seek mental health treatment**. (Probability: {proba:.2f})")
    else:
        st.success(f"✅ The model predicts this person is **unlikely to seek treatment**. (Probability: {proba:.2f})")

