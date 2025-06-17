import streamlit as st

# Judul aplikasi
st.title("Prediksi Pilihan Kampus")

# Input pengguna
nilai = st.number_input("Masukkan nilai rata-rata (0-100)", min_value=0.0, max_value=100.0, step=0.1)
jurusan = st.selectbox("Pilih jurusan yang diminati", ["Teknik Informatika", "Kedokteran", "Manajemen", "Hukum", "Sastra"])

# Fungsi prediksi sederhana
def prediksi_kampus(nilai, jurusan):
    if jurusan == "Teknik Informatika":
        if nilai >= 85:
            return "Institut Teknologi Bandung (ITB)"
        elif nilai >= 75:
            return "Universitas Gadjah Mada (UGM)"
        else:
            return "Universitas Negeri lokal"

    elif jurusan == "Kedokteran":
        if nilai >= 90:
            return "Universitas Indonesia (UI)"
        elif nilai >= 80:
            return "Universitas Airlangga"
        else:
            return "Sekolah Tinggi Kesehatan Daerah"

    elif jurusan == "Manajemen":
        if nilai >= 80:
            return "Universitas Gadjah Mada (UGM)"
        else:
            return "Universitas Swasta Terakreditasi A"

    elif jurusan == "Hukum":
        if nilai >= 78:
            return "Universitas Indonesia (UI)"
        else:
            return "Universitas Islam Negeri (UIN)"

    elif jurusan == "Sastra":
        if nilai >= 70:
            return "Universitas Diponegoro"
        else:
            return "Universitas Negeri Yogyakarta"

# Tombol untuk prediksi
if st.button("Prediksi Kampus"):
    hasil = prediksi_kampus(nilai, jurusan)
    st.success(f"Kampus yang direkomendasikan: {hasil}")
