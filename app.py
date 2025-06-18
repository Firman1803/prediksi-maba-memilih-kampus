import streamlit as st
import pickle
import numpy as np

# Load model dan encoder
with open('model.pkl', 'rb') as f:
    model, le_akreditasi = pickle.load(f)

st.title("Prediksi Pemilihan Perguruan Tinggi")

st.markdown("Masukkan kriteria pilihan kamu, dan aplikasi akan memprediksi apakah kamu akan memilih PTN ini atau tidak.")

# Input pengguna
akreditasi = st.selectbox("Akreditasi", ['A', 'B', 'C'])
uang_kuliah = st.number_input("Uang Kuliah (Rp)", min_value=0)
fasilitas = st.slider("Fasilitas (1-5)", 1, 5, 3)
pelayanan = st.slider("Pelayanan (1-5)", 1, 5, 3)
lokasi = st.slider("Lokasi (1-5)", 1, 5, 3)

# Proses prediksi
if st.button("Prediksi"):
    akreditasi_encoded = le_akreditasi.transform([akreditasi])[0]
    data = np.array([[akreditasi_encoded, uang_kuliah, fasilitas, pelayanan, lokasi]])
    prediction = model.predict(data)

    if prediction[0] == 1:
        st.success("✅ Berdasarkan input kamu, kemungkinan kamu akan memilih PTN ini.")
    else:
        st.warning("❌ Berdasarkan input kamu, kemungkinan kamu *tidak akan memilih* PTN ini.")
