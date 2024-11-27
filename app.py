import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import plotly.express as px
import plotly.figure_factory as ff

# --- Load Pretrained Model, Tokenizer, and Scaler ---
model = load_model("model.h5")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# --- Feature Extraction Functions ---
def special_char_ratio(url):
    return len(re.findall(r"[^a-zA-Z0-9]", url)) / len(url) if len(url) > 0 else 0

def preprocess_single_url(url):
    url_sequence = tokenizer.texts_to_sequences([url])  # Tokenize the URL
    padded_sequence = pad_sequences(url_sequence, maxlen=100, padding='post', truncating='post')
    special_char_ratio_val = special_char_ratio(url)

    # Add a placeholder second feature to match scaler expectations
    numerical_features = [[special_char_ratio_val, 0]]  # Placeholder feature
    scaled_numerical_features = scaler.transform(numerical_features)
    return padded_sequence, scaled_numerical_features

def predict_url(url):
    padded_sequence, scaled_features = preprocess_single_url(url)
    prediction = model.predict([padded_sequence, scaled_features])
    return "bad" if prediction[0][0] > 0.5 else "good"

# --- Model Evaluation ---
def evaluate_model(y_true, y_pred):
    st.write("### Hasil Evaluasi Model")
    st.metric("Akurasi", f"{accuracy_score(y_true, y_pred):.2%}")
    st.metric("Precision (macro)", f"{precision_score(y_true, y_pred, average='macro'):.2%}")
    st.metric("Recall (macro)", f"{recall_score(y_true, y_pred, average='macro'):.2%}")
    st.metric("F1-Score (macro)", f"{f1_score(y_true, y_pred, average='macro'):.2%}")

    # Visualize Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=['good', 'bad'])
    fig = ff.create_annotated_heatmap(
        z=cm,
        x=['good', 'bad'],
        y=['good', 'bad'],
        annotation_text=cm.astype(str),
        colorscale="Blues"
    )
    fig.update_layout(title="Confusion Matrix", xaxis_title="Predicted", yaxis_title="True")
    st.plotly_chart(fig)

# --- Sidebar and Navigation ---
st.sidebar.title("🔍 Deteksi URL Phishing")
st.sidebar.write("Aplikasi untuk mendeteksi URL phishing menggunakan Machine Learning.")
menu = st.sidebar.radio("Navigasi", ["🏠 Home", "📊 EDA", "🔮 Prediksi URL", "📈 Evaluasi Model"])

# --- Home Page ---
if menu == "🏠 Home":
    st.title("🌐 Deteksi URL Phishing")
    st.markdown(
        """
        **Selamat datang di aplikasi Deteksi URL Phishing!**  
        Aplikasi ini menggunakan model Machine Learning untuk:
        - Mengeksplorasi dataset URL.
        - Memprediksi apakah sebuah URL adalah *phishing* atau tidak.
        - Mengevaluasi performa model berdasarkan dataset yang diunggah.

        Mulailah dengan memilih menu di sidebar! 🚀
        """
    )
    st.image("https://miro.medium.com/v2/resize:fit:1200/format:webp/1*4eMO2p28USFfV_CVaHK_rw.png", use_column_width=True)

# --- EDA Page ---
elif menu == "📊 EDA":
    st.title("📊 Eksplorasi Data")
    
    uploaded_file = st.file_uploader("Unggah dataset untuk EDA (CSV/Excel)", type=["csv", "xlsx"])
    if uploaded_file:
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            data = pd.read_excel(uploaded_file)

        # Data Cleaning
        data = data.dropna(subset=['URL', 'Label'])  # Hapus baris dengan nilai kosong
        data = data[data['Label'].isin(['good', 'bad'])]  # Pertahankan hanya label valid

        if data.empty:
            st.error("Dataset kosong setelah pembersihan. Pastikan dataset memiliki nilai 'URL' dan 'Label' yang valid.")
        else:
            st.write("Data yang Diunggah:")
            st.dataframe(data.head())

            st.write("Statistik Dataset:")
            st.write(data.describe())

            st.write("Distribusi Label:")
            st.bar_chart(data['Label'].value_counts())

            st.write("Visualisasi Distribusi Panjang URL:")
            data['url_length'] = data['URL'].apply(len)
            fig = px.histogram(data, x='url_length', color='Label', title="Distribusi Panjang URL")
            st.plotly_chart(fig)

# --- Prediction Page ---
elif menu == "🔮 Prediksi URL":
    st.title("🔮 Prediksi URL Phishing")
    
    url_input = st.text_input("Masukkan URL", "")
    if st.button("Prediksi"):
        if url_input.strip() == "":
            st.warning("Silakan masukkan URL yang valid.")
        else:
            result = predict_url(url_input)
            if result == "bad":
                st.error(f"❌ Prediksi: **{result.upper()}** (Phishing)")
            else:
                st.success(f"✅ Prediksi: **{result.upper()}** (Aman)")

    st.write("---")
    st.subheader("🔢 Prediksi Batch")
    uploaded_predict_file = st.file_uploader("Unggah file CSV untuk prediksi batch", type=["csv"])
    if uploaded_predict_file:
        batch_data = pd.read_csv(uploaded_predict_file)
        if 'URL' not in batch_data.columns:
            st.error("File harus memiliki kolom bernama 'URL'.")
        else:
            batch_data['prediction'] = batch_data['URL'].apply(predict_url)
            st.write("Hasil Prediksi:")
            st.dataframe(batch_data)
            csv = batch_data.to_csv(index=False).encode('utf-8')
            st.download_button("Unduh Hasil Prediksi", data=csv, file_name="batch_predictions.csv")

# --- Model Evaluation Page ---
elif menu == "📈 Evaluasi Model":
    st.title("📈 Evaluasi Model")
    
    uploaded_eval_file = st.file_uploader("Unggah dataset untuk evaluasi model (CSV/Excel)", type=["csv", "xlsx"])
    if uploaded_eval_file:
        if uploaded_eval_file.name.endswith('.csv'):
            eval_data = pd.read_csv(uploaded_eval_file)
        elif uploaded_eval_file.name.endswith('.xlsx'):
            eval_data = pd.read_excel(uploaded_eval_file)

        # Data Cleaning
        eval_data = eval_data.dropna(subset=['URL', 'Label'])  # Hapus baris dengan nilai kosong
        eval_data = eval_data[eval_data['Label'].isin(['good', 'bad'])]  # Pertahankan hanya label valid

        if eval_data.empty:
            st.error("Dataset kosong setelah pembersihan. Pastikan dataset memiliki nilai 'URL' dan 'Label' yang valid.")
        else:
            y_true = eval_data['Label']
            y_pred = eval_data['URL'].apply(predict_url)

            st.write("Evaluasi Model:")
            evaluate_model(y_true, y_pred)
