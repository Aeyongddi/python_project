from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import streamlit as st
import matplotlib.pyplot as plt

# Streamlit 애플리케이션 시작
st.title('KOSPI 종가 예측 애플리케이션')

# 모델 로드
@st.cache_resource
def load_prediction_model():
    return load_model('./python_project/model/kospi_prediction_model.h5')

model = load_prediction_model()
st.write("### Model Loaded Successfully!")

# 데이터 로드
@st.cache_data
def load_data():
    data = pd.read_csv(
        './python_project/dataset/kospi_data.csv',
        skiprows=2,
        parse_dates=['Date'],
        index_col='Date'
    )
    data.columns = ['Close', 'High', 'Low', 'Open', 'Volume']
    return data

kospi_data = load_data()
st.write("### KOSPI Historical Data", kospi_data)

# 데이터 전처리
scaler = MinMaxScaler()
close_prices = kospi_data['Close'].values.reshape(-1, 1)
scaled_data = scaler.fit_transform(close_prices)

# `n_steps` 정의
n_steps = 30

# **과거 분석**
st.subheader("과거 데이터 분석")
start_date_past = st.date_input(
    "과거 데이터 시작 날짜",
    value=kospi_data.index.min().date(),
    min_value=kospi_data.index.min().date(),
    max_value=kospi_data.index.max().date()
)
end_date_past = st.date_input(
    "과거 데이터 종료 날짜",
    value=kospi_data.index.max().date(),
    min_value=start_date_past,
    max_value=kospi_data.index.max().date()
)

if start_date_past and end_date_past and start_date_past <= end_date_past:
    historical_data = kospi_data[start_date_past:end_date_past]
    st.line_chart(historical_data['Close'])

# **미래 데이터 예측**
st.subheader("미래 데이터 예측")

# 미래 데이터 시작 날짜와 종료 날짜
min_date = pd.Timestamp("2025-01-01").date()
max_date = pd.Timestamp("2025-12-31").date()

start_date_future = st.date_input(
    "예측 시작 날짜",
    value=min_date,
    min_value=min_date,
    max_value=max_date
)
end_date_future = st.date_input(
    "예측 종료 날짜",
    value=min_date + pd.Timedelta(days=1),
    min_value=start_date_future,
    max_value=max_date
)

# 예측 함수
def predict_future_range(model, scaled_data, scaler, start_date, end_date, n_steps):
    predictions = []
    current_input = scaled_data[-n_steps:]
    current_input = np.reshape(current_input, (1, n_steps, 1))

    while start_date <= end_date:
        next_pred = model.predict(current_input)[0][0]
        predictions.append(next_pred)
        current_input = np.append(current_input[:, 1:, :], [[[next_pred]]], axis=1)
        start_date += pd.Timedelta(days=1)

    predicted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predicted_prices

# 예측 실행
if start_date_future and end_date_future and start_date_future < end_date_future:
    try:
        start_timestamp = pd.Timestamp(start_date_future)
        end_timestamp = pd.Timestamp(end_date_future)
        predicted_prices = predict_future_range(model, scaled_data, scaler, start_timestamp, end_timestamp, n_steps)

        # 예측 결과 시각화
        st.write(f"### Predicted Prices from {start_date_future} to {end_date_future}")
        future_dates = pd.date_range(start=start_timestamp, end=end_timestamp)
        plt.figure(figsize=(12, 6))
        plt.plot(future_dates, predicted_prices, label='Predicted Prices', color='red')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title('Future KOSPI Close Price Prediction')
        plt.legend()
        st.pyplot(plt)

        # 예측 성능 지표 출력
        st.write("### 예측 성능 지표")
        if len(predicted_prices) > 1:  # 단일 예측일 경우 지표 계산 불가
            mae = mean_absolute_error(close_prices[-len(predicted_prices):], predicted_prices)
            rmse = np.sqrt(mean_squared_error(close_prices[-len(predicted_prices):], predicted_prices))
            r2 = r2_score(close_prices[-len(predicted_prices):], predicted_prices)
            st.metric("Mean Absolute Error (MAE)", f"{mae:.2f}")
            st.metric("Root Mean Squared Error (RMSE)", f"{rmse:.2f}")
            st.metric("R² Score", f"{r2:.2f}")

    except Exception as e:
        st.error(f"예측 오류: {e}")
