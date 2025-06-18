import streamlit as st
import pandas as pd
import pickle

# Judul halaman
st.title("🔮 Prediksi Calon Mahasiswa Memilih PTN")

# Load model
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# Form input prediksi
st.subheader("Masukkan Data Calon Mahasiswa")

akreditasi = st.slider("Akreditasi (1-3)", 1, 2, 3)
uang_kuliah = st.number_input("Uang Kuliah (Rp)", min_value=3000000, value=7000000, step=1000000)
fasilitas = st.slider("Fasilitas (1-3)", 1, 2, 3)
pelayanan = st.slider("Pelayanan (1-10)", 1, 2, 3)
lokasi = st.slider("Lokasi (1-10)", 1, 2, 3)

# Prediksi saat tombol ditekan
if st.button("🔍 Prediksi Sekarang"):
    input_data = pd.DataFrame([[akreditasi, uang_kuliah, fasilitas, pelayanan, lokasi]],
                              columns=['Akreditasi', 'Uang_Kuliah', 'Fasilitas', 'Pelayanan', 'Lokasi'])
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success("✅ Calon mahasiswa kemungkinan besar akan memilih PTN ini.")
    else:
        st.warning("❌ Calon mahasiswa kemungkinan tidak akan memilih PTN ini.")
