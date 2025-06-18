import streamlit as st
import pickle
import numpy as np

st.title("🤖 Prediksi Pemilihan PTN")

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

akreditasi = st.selectbox("Akreditasi PTN", [1, 2, 3])
uang_kuliah = st.number_input("Uang Kuliah (Rp)", min_value=0)
fasilitas = st.slider("Fasilitas", 1, 3, 2)
pelayanan = st.slider("Pelayanan", 1, 3, 2)
lokasi = st.slider("Lokasi", 1, 3, 2)

if st.button("Prediksi"):
    input_data = np.array([[akreditasi, uang_kuliah, fasilitas, pelayanan, lokasi]])
    result = model.predict(input_data)
    if result[0] == 1:
        st.success("✅ Kamu cenderung memilih PTN ini.")
    else:
        st.error("❌ Kamu cenderung tidak memilih PTN ini.")
