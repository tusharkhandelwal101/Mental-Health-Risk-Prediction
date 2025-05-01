# 🧠 Mental Health Treatment Prediction App

## 📌 Project Overview
This project aims to predict whether an individual is likely to seek mental health treatment based on various workplace and personal factors. It leverages logistic regression and LIME-based feature explanations and is deployed as an interactive Streamlit web application.

---

## 📂 Dataset Information
- **Dataset Name:** Mental_Health_Dataset.csv
- **Source:** [Open Source Survey Data]
- **Records:** ~1,300
- **Features:** Work interference, Family history, Benefits, Care options, Country, State, etc.

---

## 🔍 CRISP-ML(Q) Process Followed

1. **Business & Data Understanding:**
   - Goal: Predict treatment likelihood and understand mental health support drivers.
   - Data includes anonymized responses on workplace environment and mental health.

2. **Data Dictionary:**
   - Documented each feature and its significance.

3. **Exploratory Data Analysis (EDA):**
   - Visualized feature distributions and correlations.
   - Identified key features using LIME explainability.

4. **Data Preprocessing:**
   - Missing value handling
   - One-hot encoding
   - Feature selection

5. **Model Building:**
   - Logistic Regression
   - Evaluation using accuracy and probability

6. **Explainability:**
   - Used LIME to identify top contributing features.

7. **Deployment:**
   - Built a user-friendly Streamlit app for prediction.

---

## 🔑 Top 10 Features (LIME-Based)
1. work_interfere_Sometimes
2. work_interfere_Often
3. work_interfere_Rarely
4. work_interfere_Unknown
5. family_history_Yes
6. benefits_Yes
7. Country_India
8. care_options_Yes
9. state_MD
10. state_TN

---

## 🚀 How to Run the App
1. Clone the repo:
```bash
git clone https://github.com/tusharkhandelwal101/Mental-Health-Risk-Prediction.git
cd Mental-Health-Risk-Prediction
```

2. Install dependencies:
```bash
pip install streamlit
pip install scikit-learn
pip install pandas
pip install numpy
pip install lime
```

3. Run Streamlit app:
```bash
streamlit run app.py
```

---


## 📬 Contact
Feel free to connect on [LinkedIn] (https://www.linkedin.com/in/tushar-khandelwal-webdev/) for collaboration or feedback!

