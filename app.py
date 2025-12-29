import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import pandas as pd
import pickle

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load external CSS
def load_css():
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css()

# Load the trained model
@st.cache_resource
def load_model_and_encoders():
    model = tf.keras.models.load_model('model.h5')
    
    with open('label_encoder_gender.pkl', 'rb') as file:
        label_encoder_gender = pickle.load(file)
    
    with open('one_hot_encoder_geography.pkl', 'rb') as file:
        onehot_encoder_geography = pickle.load(file)
    
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    
    return model, label_encoder_gender, onehot_encoder_geography, scaler

model, label_encoder_gender, onehot_encoder_geography, scaler = load_model_and_encoders()

# Header
st.title("🏦 Customer Churn Prediction")
st.markdown('<p class="subtitle">AI-powered analytics to predict customer churn probability</p>', unsafe_allow_html=True)

# Info box
st.markdown("""
    <div class="info-box">
        <b>ℹ️</b> Predict if a customer will leave the bank based on their profile and account data
    </div>
""", unsafe_allow_html=True)

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Customer Info")
    geography = st.selectbox('🌍 Geography', onehot_encoder_geography.categories_[0])
    gender = st.selectbox('👫 Gender', label_encoder_gender.classes_)
    age = st.slider('🎂 Age', 18, 92, 35)
    tenure = st.slider('📅 Tenure', 0, 10, 5)

with col2:
    st.subheader("💳 Account Info")
    credit_score = st.number_input('📊 Credit Score', min_value=0, max_value=850, value=650)
    balance = st.number_input('💰 Balance ($)', min_value=0.0, value=50000.0, step=1000.0)
    estimated_salary = st.number_input('💵 Salary ($)', min_value=0.0, value=60000.0, step=1000.0)
    num_of_products = st.slider('🛍️ Products', 1, 4, 2)

# Additional features
col3, col4 = st.columns(2)

with col3:
    has_cr_card = st.radio('💳 Credit Card?', options=[1, 0], format_func=lambda x: 'Yes' if x == 1 else 'No', horizontal=True)

with col4:
    is_active_member = st.radio('✅ Active?', options=[1, 0], format_func=lambda x: 'Yes' if x == 1 else 'No', horizontal=True)

st.markdown("---")

# Predict button
if st.button('🔮 Predict Churn'):
    with st.spinner('Analyzing...'):
        # Prepare the input data
        input_data = pd.DataFrame({
            'CreditScore': [credit_score],
            'Gender': [label_encoder_gender.transform([gender])[0]],
            'Age': [age],
            'Tenure': [tenure],
            'Balance': [balance],
            'NumOfProducts': [num_of_products],
            'HasCrCard': [has_cr_card],
            'IsActiveMember': [is_active_member],   
            'EstimatedSalary': [estimated_salary]
        })

        # One-hot encode the Geography feature
        geography_encoded = onehot_encoder_geography.transform([[geography]]).toarray()
        geography_df = pd.DataFrame(geography_encoded, columns=onehot_encoder_geography.get_feature_names_out(['Geography']))

        # Combine all features
        input_data = pd.concat([input_data.reset_index(drop=True), geography_df], axis=1)

        # Scale the input data
        input_scaled = scaler.transform(input_data)

        # Predict churn
        prediction = model.predict(input_scaled)
        prediction_prob = prediction[0][0]

        # Display results
        col_a, col_b, col_c = st.columns([1, 2, 1])
        
        with col_b:
            # Result box with conditional styling
            if prediction_prob > 0.5:
                st.markdown(f"""
                    <div class="prediction-box churn-high">
                        <h2>⚠️ High Churn Risk</h2>
                        <h1>{prediction_prob*100:.1f}%</h1>
                        <p><b>Customer likely to leave</b></p>
                        <p>Action: Implement retention strategy</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="prediction-box churn-low">
                        <h2>✅ Low Churn Risk</h2>
                        <h1>{prediction_prob*100:.1f}%</h1>
                        <p><b>Customer likely to stay</b></p>
                        <p>Status: Retention is strong</p>
                    </div>
                """, unsafe_allow_html=True)

# Footer
st.markdown("""
    <div style="text-align: center; color: #95a5a6; padding: 0.5rem; font-size: 0.85rem;">
        🤖 Powered by Deep Learning<br>
        © 2025 Imansha Dilshan. All rights reserved.
    </div>
""", unsafe_allow_html=True)