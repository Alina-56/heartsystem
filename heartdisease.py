import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import streamlit as st

# Load dataset
data = pd.read_csv('heart.csv')

# Data preprocessing
data['Sex'] = data['Sex'].map({'M': 0, 'F': 1})  # Encode gender

# Streamlit app
def main():
    st.title("Heart Disease Prediction Based on Cholesterol Level")

    # Sidebar inputs
    age = st.sidebar.number_input("Enter your age", min_value=1, max_value=120, value=30)
    bp = st.sidebar.number_input("Enter your resting blood pressure", min_value=50, max_value=200, value=120)
    chest_pain = st.sidebar.selectbox("Do you experience chest pain?", ["No", "Yes"])
    chest_pain_encoded = 1 if chest_pain == "Yes" else 0

    # Train regression model separately for males and females
    male_data = data[data['Sex'] == 0]
    female_data = data[data['Sex'] == 1]

    models = {}
    for gender, gender_data in zip(["Male", "Female"], [male_data, female_data]):
        X = gender_data[['Cholesterol']]
        y = gender_data['HeartDisease']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        models[gender] = model

        # Plot regression line
        plt.figure(figsize=(8, 5))
        sns.scatterplot(x='Cholesterol', y='HeartDisease', data=gender_data, label='Data Points')
        plt.plot(X, model.predict(X), color='red', label='Regression Line')
        plt.title(f"Simple Linear Regression for {gender}")
        plt.xlabel("Cholesterol Level")
        plt.ylabel("Heart Disease (0 = No, 1 = Yes)")
        plt.legend()
        st.pyplot(plt)

    # User input prediction
    user_cholesterol = st.number_input("Enter your cholesterol level", min_value=100, max_value=500, value=200)
    user_gender = st.radio("Select your gender", ["Male", "Female"])
