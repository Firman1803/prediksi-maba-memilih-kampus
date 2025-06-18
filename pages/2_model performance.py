import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

st.title("📈 Model Performance")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("dataset.csv", sep=";")
    df.columns = ['Akreditasi', 'Uang_Kuliah', 'Fasilitas', 'Pelayanan', 'Lokasi', 'Pilih_PTN']
    df['Pilih_PTN'] = df['Pilih_PTN'].map({'Ya': 1, 'Tidak': 0})
    return df

df = load_data()

# Split data
X = df[['Akreditasi', 'Uang_Kuliah', 'Fasilitas', 'Pelayanan', 'Lokasi']]
y = df['Pilih_PTN']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Akurasi
st.subheader("🎯 Akurasi Model")
acc = accuracy_score(y_test, y_pred)
st.metric("Akurasi", f"{acc*100:.2f}%")

# Confusion Matrix
st.subheader("🧩 Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Tidak", "Ya"], yticklabels=["Tidak", "Ya"])
plt.xlabel("Prediksi")
plt.ylabel("Aktual")
st.pyplot(fig)

# Classification Report
st.subheader("📋 Classification Report")
report = classification_report(y_test, y_pred, target_names=["Tidak", "Ya"], output_dict=True)
report_df = pd.DataFrame(report).transpose()
st.dataframe(report_df.style.format({"precision": "{:.2f}", "recall": "{:.2f}", "f1-score": "{:.2f}"}))

# Prediksi manual
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
    hasil = "✅ Memilih PTN Ini" if prediction == 1 else "❌ Tidak Memilih PTN Ini"
    st.success(f"Hasil Prediksi: {hasil}")
