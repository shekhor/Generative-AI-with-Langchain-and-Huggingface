import streamlit as st
import pandas as pd
import tensorflow as tf
import pickle

model = tf.keras.models.load_model('model.h5')

with open('one_hot_encoder_geo.pkl', 'rb') as f:
    one_hot_encoder_geo = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)

st.title('Customer Churn Prediction')

geography = st.selectbox('Geography', one_hot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
credit_score = st.slider('Credit Score', min_value=300, max_value=900, value=650)
age = st.slider('Age', min_value=18, max_value=100, value=40)
tenure = st.slider('Tenure', min_value=0, max_value=10, value=3)
balance = st.number_input('Balance', min_value=0.0, value=0.0)
num_of_products = st.slider('Number of Products', min_value=1, max_value=4, value=1)
has_cr_card = st.selectbox('Has Credit Card', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
is_active_member = st.selectbox('Is Active Member', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
estimated_salary = st.slider('Estimated Salary', min_value=0, max_value=200000, value=50000)

input_data = {
    'CreditScore': credit_score,
    'Geography': geography,
    'Gender': gender,
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': num_of_products,
    'HasCrCard': has_cr_card,
    'IsActiveMember': is_active_member,
    'EstimatedSalary': estimated_salary,
}

input_df = pd.DataFrame([input_data])
input_df['Gender'] = label_encoder_gender.transform(input_df['Gender'])

geo_encoded = one_hot_encoder_geo.transform(input_df[['Geography']]).toarray()
geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=one_hot_encoder_geo.get_feature_names_out(['Geography'])
)

input_df = pd.concat([input_df.drop('Geography', axis=1), geo_encoded_df], axis=1)
input_scaled = scaler.transform(input_df)

prediction_prob = model.predict(input_scaled, verbose=0)[0][0]
prediction = 1 if prediction_prob >= 0.5 else 0

if prediction == 0:
    st.success(f'The customer is not likely to churn (probability: {prediction_prob:.2%})')
else:
    st.error(f'The customer is likely to churn (probability: {prediction_prob:.2%})')
