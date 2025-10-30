# app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

st.set_page_config(page_title="Procurement ML App", layout="wide")

st.title("📊 Procurement Strategy ML App (Kraljic Matrix Style)")
st.markdown("Upload your dataset and train a machine learning model instantly.")

# --- 1. Upload CSV file ---
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Dataset loaded successfully!")
    st.write("### Dataset Preview:")
    st.dataframe(df.head())

    # --- 2. Select Target Column ---
    st.subheader("🎯 Select Target Variable")
    target_col = st.selectbox("Select the target column to predict:", df.columns)

    # --- 3. Optional Encoding ---
    st.subheader("🔤 Encode Categorical Columns Automatically")
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    if len(cat_cols) > 0:
        st.write(f"Categorical columns detected: {cat_cols}")
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    # --- 4. Feature Selection ---
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # --- 5. Split Data ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # --- 6. Scale Numeric Data ---
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # --- 7. Train Model ---
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # --- 8. Evaluation ---
    st.subheader("📈 Model Evaluation")
    st.text("Confusion Matrix:")
    st.write(confusion_matrix(y_test, y_pred))
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    # --- 9. Single Prediction ---
    st.subheader("🔮 Try a Single Prediction")
    input_data = {}
    for i, col in enumerate(df.drop(columns=[target_col]).columns):
        val = st.text_input(f"Enter value for {col}")
        input_data[col] = val

    if st.button("Predict"):
        input_df = pd.DataFrame([input_data])
        # Ensure numeric columns are converted properly
        for c in input_df.columns:
            input_df[c] = pd.to_numeric(input_df[c], errors='ignore')
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)
        st.success(f"✅ Predicted {target_col}: **{prediction[0]}**")

else:
    st.warning("👆 Please upload a CSV file to continue.")
