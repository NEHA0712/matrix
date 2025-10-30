# app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 1. Load data
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    return df

st.title("Procurement Strategy – Kraljic Matrix Classification")

data_path = st.text_input("Enter path or URL of dataset CSV", "/content/realistic_kraljic_dataset.csv")
if data_path:
    df = load_data(data_path)
    st.write("Dataset preview:", df.head())

    # 2. Basic cleaning
    st.subheader("Data Cleaning")
    # Example: Drop unwanted columns
    cols_to_drop = st.multiselect("Columns to drop", df.columns.tolist(), default=[])
    if st.button("Drop selected columns"):
        df = df.drop(columns=cols_to_drop)
        st.write("After drop:", df.head())

    # 3. Pre‐process / feature engineering
    st.subheader("Feature Engineering")
    # Example: encode categorical columns
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    st.write("Categorical columns:", cat_cols)
    cols_encode = st.multiselect("Columns to encode", cat_cols)
    if st.button("Encode selected columns"):
        for col in cols_encode:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
        st.write("After encoding:", df.head())

    # 4. Select target and features
    st.subheader("Select Features & Target")
    target_col = st.selectbox("Select target variable (to predict)", df.columns.tolist())
    feature_cols = st.multiselect("Select feature columns", [c for c in df.columns if c != target_col], default=[c for c in df.columns if c != target_col][:5])
    if st.button("Proceed to model"):
        X = df[feature_cols]
        y = df[target_col]

        # train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # scale numeric features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 5. Train model
        st.subheader("Model Training")
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train_scaled, y_train)
        st.write("Model trained.")

        # 6. Metrics
        y_pred = rf.predict(X_test_scaled)
        st.subheader("Evaluation")
        st.text("Confusion Matrix:")
        st.write(confusion_matrix(y_test, y_pred))
        st.text("Classification Report:")
        st.write(classification_report(y_test, y_pred))

        # 7. Single prediction
        st.subheader("Make a single prediction")
        input_data = {}
        for feat in feature_cols:
            val = st.text_input(f"Enter value for {feat}")
            input_data[feat] = val
        if st.button("Predict"):
            # convert into proper shape & types
            X_new = pd.DataFrame([input_data])
            X_new = X_new.astype({feat: X[feat].dtype for feat in feature_cols})
            X_new_scaled = scaler.transform(X_new)
            pred = rf.predict(X_new_scaled)
            st.write(f"Predicted {target_col}: {pred[0]}")

