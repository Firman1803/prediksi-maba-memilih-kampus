import streamlit as st
import numpy as np
import pickle

st.title("🤖 Prediksi Pemilihan PTN")

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

akreditasi = st.selectbox("Akreditasi PTN", ["A", "B", "C"])
uang_kuliah = st.number_input("Uang Kuliah per semester (Rp)", min_value=1000000, step=500000)
fasilitas = st.slider("Fasilitas Kampus (1–5)", 1, 5, 3)
pelayanan = st.slider("Pelayanan Mahasiswa (1–5)", 1, 5, 3)
lokasi = st.slider("Kedekatan Lokasi (1 = dekat, 5 = jauh)", 1, 5, 3)

akreditasi_val = {"A": 0, "B": 1, "C": 2}[akreditasi]

if st.button("Prediksi"):
    input_data = np.array([[akreditasi_val, uang_kuliah, fasilitas, pelayanan, lokasi]])
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.success("✅ Kamu kemungkinan besar akan memilih PTN ini.")
    else:
        st.error("❌ Kamu kemungkinan tidak memilih PTN ini.")

