
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Load the Titanic dataset
train = pd.read_csv(r"C:\Users\dell\DATA SCIENCE_ExcelR\Assignments\Assignment - 7\Titanic_train.csv")

# Preprocess the data (same as in the notebook)
train["Embarked"].fillna(train["Embarked"].mode()[0], inplace=True)
train["Age"].fillna(train["Age"].mean(), inplace=True)

# Initialize LabelEncoders
le_sex = LabelEncoder()
train["Sex"] = le_sex.fit_transform(train["Sex"])

le_embarked = LabelEncoder()
train["Embarked"] = le_embarked.fit_transform(train["Embarked"])

# Prepare input data for prediction
X = train[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]
Y = train["Survived"]

# Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X, Y)

# Streamlit app
st.title('Titanic Survival Prediction')
pclass = st.selectbox('Passenger Class', [1, 2, 3])
sex = st.selectbox('Sex', ['Male', 'Female'])
age = st.slider('Age', 1, 100)
sibsp = st.slider('Number of Siblings/Spouses Aboard', 0, 8)
parch = st.slider('Number of Parents/Children Aboard', 0, 6)
fare = st.slider('Fare Paid', 0, 500)
embarked = st.selectbox('Port of Embarkation', ['S', 'C', 'Q'])

# Convert input to the encoded format for the model
input_data = [
    pclass,
    le_sex.transform([sex.lower()])[0],
    age,
    sibsp,
    parch,
    fare,
    le_embarked.transform([embarked])[0]
]

if st.button('Predict Survival'):
    prediction = model.predict([input_data])[0]
    probability = model.predict_proba([input_data])[0][1]
    st.write(f"Prediction: {'Survived' if prediction == 1 else 'Not Survive'}")
    st.write(f"Probability of Survival: {probability:.2f}")
