# ============================================================
# 🧠 Kraljic Matrix Procurement Classification (Streamlit App)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# ------------------------------------------------------------
# 🌐 Streamlit UI
# ------------------------------------------------------------
st.set_page_config(page_title="Kraljic Matrix ML App", layout="wide")
st.title("📦 Kraljic Matrix Procurement Classification App")
st.write("Upload your dataset to train multiple machine learning models and predict procurement categories.")

# ------------------------------------------------------------
# 📤 Upload CSV
# ------------------------------------------------------------
uploaded_file = st.file_uploader("📁 Upload your training CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Dataset loaded successfully!")
    st.dataframe(df.head())

    # ------------------------------------------------------------
    # 🧹 Basic EDA
    # ------------------------------------------------------------
    st.subheader("🔍 Dataset Overview")
    st.write("**Shape:**", df.shape)
    st.write("**Missing Values:**")
    st.write(df.isnull().sum())

    if "Kraljic_Category" in df.columns:
        target_col = "Kraljic_Category"
    elif "Category" in df.columns:
        target_col = "Category"
    else:
        st.error("❌ Target column 'Kraljic_Category' or 'Category' not found!")
        st.stop()

    st.write("**Class Distribution:**")
    st.bar_chart(df[target_col].value_counts())

    # ------------------------------------------------------------
    # 🧩 Data Preprocessing
    # ------------------------------------------------------------
    st.subheader("⚙️ Data Preprocessing")

    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        if col != target_col:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le

    # Features and target
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    st.success("✅ Data preprocessing completed!")

    # ------------------------------------------------------------
    # 🧠 Train Multiple Models
    # ------------------------------------------------------------
    st.subheader("🤖 Training Models")

    models = {
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "SVM": SVC(kernel='rbf', probability=True),
        "Naive Bayes": GaussianNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc

    st.write("### 📊 Model Accuracy Comparison")
    results_df = pd.DataFrame(list(results.items()), columns=["Model", "Accuracy"])
    st.bar_chart(results_df.set_index("Model"))

    best_model_name = max(results, key=results.get)
    st.success(f"🏆 Best Model: {best_model_name} ({results[best_model_name]:.2f} accuracy)")

    best_model = models[best_model_name]

    # ------------------------------------------------------------
    # 📥 Upload Test Data for Prediction
    # ------------------------------------------------------------
    st.subheader("📤 Upload Test Data for Prediction")

    test_file = st.file_uploader("Upload your test CSV file", type=["csv"])

    if test_file:
        test_df = pd.read_csv(test_file)
        st.write("✅ Test data preview:")
        st.dataframe(test_df.head())

        # Encode test data using fitted encoders
        for col, le in label_encoders.items():
            if col in test_df.columns:
                test_df[col] = le.transform(test_df[col].astype(str))

        # Standardize using same scaler
        test_scaled = scaler.transform(test_df)

        # Predict
        predictions = best_model.predict(test_scaled)
        test_df["Predicted_Category"] = predictions

        st.success("✅ Predictions complete!")
        st.dataframe(test_df.head())

        # Download predictions
        csv = test_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📩 Download Predictions CSV",
            data=csv,
            file_name="Kraljic_Predictions.csv",
            mime="text/csv"
        )

else:
    st.info("👆 Upload your training CSV file to begin.")
