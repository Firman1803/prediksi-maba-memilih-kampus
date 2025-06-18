import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("dataset.csv", sep=";")
df.columns = ['Akreditasi', 'Uang_Kuliah', 'Fasilitas', 'Pelayanan', 'Lokasi', 'Pilih_PTN']

st.title("📊 Dashboard Data")
st.dataframe(df.head())

st.subheader("Distribusi Pilihan PTN")
st.bar_chart(df['Pilih_PTN'].value_counts())

st.subheader("Korelasi")
df['Pilih_PTN'] = df['Pilih_PTN'].map({'Ya': 1, 'Tidak': 0})
corr = df.corr(numeric_only=True)
fig, ax = plt.subplots()
sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)
