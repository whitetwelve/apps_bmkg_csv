import streamlit as st
import joblib
import pandas as pd

def load_model_and_label_encoder(model_path, label_encoder_path):
    loaded_model = joblib.load(model_path)
    loaded_label_encoder = joblib.load(label_encoder_path)
    return loaded_model, loaded_label_encoder

def predict_weather(model, label_encoder, suhu, kecepatan_angin, kelembapan):
    new_data = pd.DataFrame({
        'suhu': [suhu],
        'kecepatanangin': [kecepatan_angin],
        'kelembapan': [kelembapan]
    })

    predicted_weather = model.predict(new_data)
    predicted_weather_label = label_encoder.inverse_transform(predicted_weather)
    return predicted_weather_label[0]

# Load model and label encoder
loaded_model, loaded_label_encoder = load_model_and_label_encoder('csv.sav', 'label_csv.sav')

# Streamlit App
st.title('Prediksi Cuaca Data CSV')

# Input parameters
suhu = st.slider('Suhu (Celcius)', min_value=0.0, max_value=35.0, value=25.0)
kecepatan_angin = st.slider('Kecepatan Angin (km/h)', min_value=0, max_value=50, value=10)
kelembapan = st.slider('Kelembapan (%)', min_value=0.0, max_value=100.0, value=50.0)

# Predict button
if st.button('Prediksi Cuaca'):
    predicted_result = predict_weather(loaded_model, loaded_label_encoder, suhu, kecepatan_angin, kelembapan)
    st.success(f'Hasil Prediksi Cuaca: {predicted_result}')
