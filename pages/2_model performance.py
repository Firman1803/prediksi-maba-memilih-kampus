import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Judul halaman
st.title("📈 Model Performance")

# Load dataset dan model
@st.cache_data
def load_data():
    return pd.read_csv("dataset.csv", sep=";")

def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

df = load_data()
model = load_model()

# Preprocessing dataset
df.columns = ['Akreditasi', 'Uang_Kuliah', 'Fasilitas', 'Pelayanan', 'Lokasi', 'Pilih_PTN']
df['Pilih_PTN'] = df['Pilih_PTN'].map({'Ya': 1, 'Tidak': 0})

X = df[['Akreditasi', 'Uang_Kuliah', 'Fasilitas', 'Pelayanan', 'Lokasi']]
y = df['Pilih_PTN']

# Prediksi
y_pred = model.predict(X)

# Akurasi
st.subheader("🎯 Akurasi Model")
acc = accuracy_score(y, y_pred)
st.metric(label="Akurasi", value=f"{acc*100:.2f}%")

# Confusion Matrix
st.subheader("🧩 Confusion Matrix")
cm = confusion_matrix(y, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Tidak", "Ya"], yticklabels=["Tidak", "Ya"])
plt.xlabel("Prediksi")
plt.ylabel("Aktual")
st.pyplot(fig)

# Classification Report
st.subheader("📋 Classification Report")
report = classification_report(y, y_pred, target_names=["Tidak", "Ya"], output_dict=True)
report_df = pd.DataFrame(report).transpose()
st.dataframe(report_df.style.format({"precision": "{:.2f}", "recall": "{:.2f}", "f1-score": "{:.2f}"}))

# Percobaan Prediksi Manual
st.subheader("🧪 Coba Prediksi Manual")
akreditasi = st.slider("Akreditasi", 1, 10, 8)
uang_kuliah = st.number_input("Uang Kuliah", min_value=0, value=5000000)
fasilitas = st.slider("Fasilitas", 1, 10, 7)
pelayanan = st.slider("Pelayanan", 1, 10, 8)
lokasi = st.slider("Lokasi", 1, 10, 9)

if st.button("Prediksi"):
    input_data = pd.DataFrame([[akreditasi, uang_kuliah, fasilitas, pelayanan, lokasi]],
                              columns=['Akreditasi', 'Uang_Kuliah', 'Fasilitas', 'Pelayanan', 'Lokasi'])
    prediction = model.predict(input_data)[0]
    hasil = "Memilih PTN Ini ✅" if prediction == 1 else "Tidak Memilih PTN Ini ❌"
    st.success(f"Hasil Prediksi: {hasil}")
