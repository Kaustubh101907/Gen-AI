import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import pickle


## loading the trained model
model = tf.keras.models.load_model('model.h5')


## load the encoders and scaler
with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)

with open('onehot_encoder_geo.pkl', 'rb') as f:
    onehot_encoder_geo = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)


## Streamlit app
st.title("Customer Churn Prediction")

## user input
Geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
Gender = st.selectbox('Gender', label_encoder_gender.classes_)
Age = st.slider('Age', 18, 100, 30)
Balance = st.number_input('Balance', min_value=0.0, step=100.0)
credit_score = st.number_input('Credit Score', min_value=0, max_value=1000, step=1)
estimated_salary = st.number_input('Estimated Salary', min_value=0.0, step=100.0)
tenure = st.slider('Tenure', 0, 10, 1)
num_of_products = st.slider('Number of Products', 1, 4, 1)
has_cr_card = st.selectbox('Has Credit Card', ['Yes', 'No'])
is_active_member = st.selectbox('Is Active Member', [0, 1])


## Prepare input data
input_data = {
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([Gender])[0]],
    'Age': [Age],
    'Tenure': [tenure],
    'Balance': [Balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [1 if has_cr_card == 'Yes' else 0],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
}


## one hot encode Geography
geo_encoded = onehot_encoder_geo.transform([[Geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))


## combine onehot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)