import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# 페이지 설정
st.set_page_config(page_title='Insurance Premium Prediction', page_icon=':money_with_wings:', layout='wide')

# 페이지 헤더
st.title('Insurance Premium Prediction')

# 절대 경로 설정
BASE_DIR = r'D:\python_project'

# 모델 로드
@st.cache_resource
def load_prediction_model():
    model_path = os.path.join(BASE_DIR, 'model', 'optimized_rf_model.pkl')
    return joblib.load(model_path)

model = load_prediction_model()
st.write("### Model Loaded Successfully!")

# 사용자 입력 폼
st.sidebar.header('User Input Features')
def user_input():
    age = st.sidebar.slider('Age', 18, 100, 30)
    bmi = st.sidebar.slider('BMI', 10.0, 50.0, 25.0)
    children = st.sidebar.slider('Children', 0, 10, 1)
    gender_male = st.sidebar.selectbox('Gender', ['Male', 'Female'])
    smoker_yes = st.sidebar.selectbox('Smoker', ['Yes', 'No'])
    
    # 데이터 프레임 생성
    data = {
        'age': age,
        'bmi': bmi,
        'children': children,
        'gender_male': 1 if gender_male == 'Male' else 0,
        'smoker_yes': 1 if smoker_yes == 'Yes' else 0   
    }
    return pd.DataFrame(data, index=[0])

input_data = user_input()

# 사용자 입력 데이터 표시
st.subheader('User Input Features')
st.write(input_data)

# 예측 수행
if st.button('Predict'):
    prediction = model.predict(input_data)
    st.subheader('Predicted Insurance Charges')
    st.write(f"${prediction[0]:.2f}")

