import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.title("🔮 Prediksi Pilihan Mahasiswa")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("dataset.csv", sep=";")
    df.columns = ['Akreditasi', 'Uang_Kuliah', 'Fasilitas', 'Pelayanan', 'Lokasi', 'Pilih_PTN']
    df['Pilih_PTN'] = df['Pilih_PTN'].map({'Ya': 1, 'Tidak': 0})
    return df

df = load_data()

# Preprocessing dan pelatihan model langsung
X = df[['Akreditasi', 'Uang_Kuliah', 'Fasilitas', 'Pelayanan', 'Lokasi']]
y = df['Pilih_PTN']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# UI Input untuk Prediksi
st.subheader("🧪 Masukkan Data Mahasiswa")

akreditasi = st.slider("Akreditasi Kampus", 1, 10, 7)
uang_kuliah = st.number_input("Uang Kuliah", min_value=0, value=5000000)
fasilitas = st.slider("Fasilitas Kampus", 1, 10, 6)
pelayanan = st.slider("Pelayanan Akademik", 1, 10, 7)
lokasi = st.slider("Lokasi Kampus", 1, 10, 8)

# Prediksi
if st.button("Prediksi Pilihan Mahasiswa"):
    input_data = pd.DataFrame([[akreditasi, uang_kuliah, fasilitas, pelayanan, lokasi]],
                              columns=['Akreditasi', 'Uang_Kuliah', 'Fasilitas', 'Pelayanan', 'Lokasi'])
    prediction = model.predict(input_data)[0]
    hasil = "✅ Memilih Kampus Ini" if prediction == 1 else "❌ Tidak Memilih Kampus Ini"
    st.success(f"Hasil Prediksi: {hasil}")
