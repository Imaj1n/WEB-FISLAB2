import streamlit as st
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Fungsi simulasi osilasi pegas
def osilasi_pegas(massa, konstanta_pegas, amplitudo, waktu):
    waktu = np.linspace(0, waktu, 1000)
    frekuensi = np.sqrt(konstanta_pegas / massa)
    posisi = amplitudo * np.sin(2 * np.pi * frekuensi * waktu)
    return waktu, posisi

def app():
    # Judul aplikasi
    st.title('Simulasi Osilasi Pegas')

    # Widget untuk mengatur parameter
    massa = st.slider('Massa (kg)', min_value=0.1, max_value=10.0, value=1.0, step=0.1)
    konstanta_pegas = st.slider('Konstanta Pegas (N/m)', min_value=1, max_value=100, value=10, step=1)
    amplitudo = st.slider('Amplitudo (m)', min_value=0.1, max_value=5.0, value=1.0, step=0.1)
    waktu = st.slider('Waktu (s)', min_value=1, max_value=10, value=5, step=1)

    # Memanggil fungsi simulasi
    waktu_sim, posisi_sim = osilasi_pegas(massa, konstanta_pegas, amplitudo, waktu)

    # Visualisasi dengan Matplotlib
    fig, ax = plt.subplots()
    ax.plot(waktu_sim, posisi_sim)
    ax.set_xlabel('Waktu (s)')
    ax.set_ylabel('Posisi (m)')
    ax.set_title('Osilasi Pegas')
    st.pyplot(fig)