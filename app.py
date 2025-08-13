import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px

st.set_page_config(page_title="Titanic Survival Predictor", layout="wide")

# ---------------------------
# Helpers
# ---------------------------
@st.cache_data
def load_data(path="titanic.csv"):
    if not os.path.exists(path):
        st.error(f"Dataset not found at {path}. Please run model_training.py first.")
        return None
    df = pd.read_csv(path)
    if 'survived' in df.columns and 'Survived' not in df.columns:
        df.rename(columns={'survived': 'Survived'}, inplace=True)
    return df

@st.cache_resource
def load_model(path="model.pkl"):
    if not os.path.exists(path):
        return None
    return joblib.load(path)

df = load_data()
model = load_model()

# ---------------------------
# Sidebar navigation
# ---------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data", "Visualize", "Predict"])

# ---------------------------
# Home
# ---------------------------
if page == "Home":
    st.title("🚢 Titanic Survival Prediction")
    st.write("""
    This app predicts whether a passenger would have survived the Titanic disaster,
    based on their characteristics.
    """)
    if model is None:
        st.warning("Model not found. Please run model_training.py first.")
    if df is not None:
        st.write("Dataset sample:")
        st.dataframe(df.sample(5))

# ---------------------------
# Data view
# ---------------------------
elif page == "Data":
    st.title("📊 Dataset Overview")
    if df is None:
        st.error("Dataset missing.")
    else:
        st.write("Shape:", df.shape)
        st.dataframe(df.head(20))
        st.write("Missing values:")
        st.table(df.isnull().sum())

# ---------------------------
# Visualize
# ---------------------------
elif page == "Visualize":
    st.title("📈 Visualizations")
    if df is None:
        st.error("Dataset missing.")
    else:
        # Survival by Pclass
        st.subheader("Survival by Passenger Class")
        fig1 = px.histogram(df, x='Pclass', color='Survived', barmode='group')
        st.plotly_chart(fig1, use_container_width=True)

        # Survival by Sex
        st.subheader("Survival by Sex")
        fig2 = px.histogram(df, x='Sex', color='Survived', barmode='group')
        st.plotly_chart(fig2, use_container_width=True)

        # Age distribution
        st.subheader("Age Distribution by Survival")
        fig3 = px.box(df, x='Survived', y='Age', points="all")
        st.plotly_chart(fig3, use_container_width=True)

# ---------------------------
# Predict
# ---------------------------
elif page == "Predict":
    st.title("🤖 Make a Prediction")
    if model is None:
        st.error("Model not found. Run model_training.py to create model.pkl")
    else:
        st.write("Enter passenger details:")

        col1, col2, col3 = st.columns(3)
        with col1:
            pclass = st.selectbox("Passenger Class", [1, 2, 3], index=0)
            sex = st.selectbox("Sex", ["male", "female"], index=0)
            age = st.slider("Age", 0, 100, 30)
        with col2:
            sibsp = st.number_input("Siblings/Spouses aboard", min_value=0, max_value=10, value=0)
            parch = st.number_input("Parents/Children aboard", min_value=0, max_value=10, value=0)
            fare = st.number_input("Fare", min_value=0.0, value=32.0, format="%.2f")
        with col3:
            embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"], index=0)

        input_df = pd.DataFrame([{
            "Pclass": int(pclass),
            "Sex": sex,
            "Age": float(age),
            "SibSp": int(sibsp),
            "Parch": int(parch),
            "Fare": float(fare),
            "Embarked": embarked
        }])

        if st.button("Predict"):
            try:
                pred = model.predict(input_df)[0]
                proba = None
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(input_df)[0][1]
                st.subheader("Prediction Result")
                st.write("✅ Survived" if pred == 1 else "❌ Did not survive")
                if proba is not None:
                    st.write(f"Survival Probability: **{proba*100:.2f}%**")
            except Exception as e:
                st.error("Prediction failed: " + str(e))
