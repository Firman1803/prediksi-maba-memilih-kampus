import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("📊 Dashboard Data")

# Load dataset
df = pd.read_csv("dataset.csv")

st.subheader("Preview Data")
st.dataframe(df.head())

st.subheader("Distribusi Akreditasi")
st.bar_chart(df['Akreditasi'].value_counts())

st.subheader("Distribusi Kolom Numerik")
num_cols = ['Uang_Kuliah', 'Fasilitas', 'Pelayanan', 'Lokasi']
st.line_chart(df[num_cols])

st.subheader("Korelasi Antar Fitur")
corr = df[num_cols + ['Pilih_PTN']].corr()
fig, ax = plt.subplots()
sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

