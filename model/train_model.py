import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# Load data
df = pd.read_csv('dataset.csv')

# Encode fitur kategorikal
le_akreditasi = LabelEncoder()
df['Akreditasi'] = le_akreditasi.fit_transform(df['Akreditasi'])
df['Target'] = df['Target'].map({'Ya': 1, 'Tidak': 0})

# Fitur dan target
X = df[['Akreditasi', 'Uang Kuliah', 'Fasilitas', 'Pelayanan', 'Lokasi']]
y = df['Target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Simpan model dan label encoder
with open('model.pkl', 'wb') as f:
    pickle.dump((model, le_akreditasi), f)
