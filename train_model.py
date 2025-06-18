import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# Baca dataset
df = pd.read_csv("dataset.csv", sep=";")
df.columns = ['Akreditasi', 'Uang_Kuliah', 'Fasilitas', 'Pelayanan', 'Lokasi', 'Pilih_PTN']
df['Pilih_PTN'] = df['Pilih_PTN'].map({'Ya': 1, 'Tidak': 0})

# Siapkan data
X = df[['Akreditasi', 'Uang_Kuliah', 'Fasilitas', 'Pelayanan', 'Lokasi']]
y = df['Pilih_PTN']

# Split dan latih model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Simpan model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model berhasil disimpan sebagai model.pkl")
