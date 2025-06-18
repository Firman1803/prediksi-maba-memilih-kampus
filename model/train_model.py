import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
df = pd.read_csv("dataset.csv", sep=";")
df.columns = ['Akreditasi', 'Uang_Kuliah', 'Fasilitas', 'Pelayanan', 'Lokasi', 'Pilih_PTN']
df['Pilih_PTN'] = df['Pilih_PTN'].map({'Ya': 1, 'Tidak': 0})

# Features & Target
X = df[['Akreditasi', 'Uang_Kuliah', 'Fasilitas', 'Pelayanan', 'Lokasi']]
y = df['Pilih_PTN']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
