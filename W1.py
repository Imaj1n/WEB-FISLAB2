import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def sub_judul(teks):
    st.divider()
    st.subheader(teks)
    st.divider()
def Indeks_tabel(tabel,indeks):
    st.caption(indeks)
    st.table(tabel)


def app():
    st.title("Sifat Gelombang Cahaya : Hukum snell, Difraksi dan Intenferensi")
    st.caption("25 April 2024")
    text1 = r"""
Persamaan yang digunakan dalam Hukum snellius ini adalah :

$n_i \sin{\theta_i}=n_f \sin{\theta_f}$

dimana pada kasus percobaan ini, $n_i = n_{udara}=1$ sehingga persamaan diatas menjadi 

$n_f = \frac{\sin{\theta_i}}{\sin{\theta_f}}$

dengan $\theta_i$ bervariasi dari $10°,20°,30°,40°,50°$ dan $60°$
    """
    st.markdown(text1,unsafe_allow_html=True)
    text2 = """
Import Modul
"""
    st.divider()
    st.subheader(text2)
    st.divider()
    modul = """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

    """
    st.code(modul)
    sub_judul('Menghitung Indeks Bias')
    kode = """
def menghitung_indeks_bias(theta2):
  indeks_bias = []
  for i in range(len(theta2)):
    indeksi = np.sin(np.deg2rad((i+1)*10))/np.sin(np.deg2rad(theta2[i]))
    indeks_bias.append(indeksi)
  rata2 = np.mean(indeks_bias)
  Tabel1 = pd.DataFrame({
      "Pengulangan":[1,2,3,4,5,6,''],
      "theta1":[i for i in range(10,70,10)]+[''],
      "theta2":[6,10,17,23,27,31,""],
      "indeks_bias":indeks_bias+[rata2]
  })
  return Tabel1
"""
    theta2=[6,10,17,23,27,31]
    def menghitung_indeks_bias(theta2):
        indeks_bias = []
        for i in range(len(theta2)):
            indeksi = np.sin(np.deg2rad((i+1)*10))/np.sin(np.deg2rad(theta2[i]))
            indeks_bias.append(indeksi)
        rata2 = np.mean(indeks_bias)
        Tabel1 = pd.DataFrame({
            "Pengulangan":[1,2,3,4,5,6,'rata rata'],
            "theta1":[i for i in range(10,70,10)]+[''],
            "theta2":[6,10,17,23,27,31,""],
            "indeks_bias":indeks_bias+[rata2]
        })
        return Tabel1
    Tabel1 = menghitung_indeks_bias(theta2)
    Indeks_tabel(Tabel1,'Tabel menghitung Indeks Bias')
    sub_judul("Menghitung Panjang gelombang dengan kisi difraksi")
    teks2 = r"""
Menghitung Panjang gelombang dengan kisi Difraksi

$λ = \frac{d}{n}\frac{x}{\sqrt{x^2+l^2}}$

"""
    kode3 = """
kisi1 = 100 #kisi/mm
kisi2 = 300
kisi3 = 600
l = 30
x1 = [1.6,3.225,5.2,6.7,8.5,10.2]
x2 = [4.9,10.65]
x3 = [9.75]
def menghitung_panjang_gelombang(x,l,kisi):
  d = 2/(kisi*10)
  panjang_gelombang = []
  for i in range(len(x)):
    panjang_gelombang.append((d/(i+1))*(x[i]/((x[i])**2+l**2)**(0.5))*10**(7))

  Tabel2 = pd.DataFrame({
      'orde':[i for i in range(len(x))]+['rata-rata'],
      'jarak x (cm)':x+[''],
      'panjang_gelombang (nm)':panjang_gelombang+[np.mean(panjang_gelombang)]
  })
  return Tabel2
Tabel2 = menghitung_panjang_gelombang(x1,l,kisi1)
Tabel2
"""
    st.code(kode3)
    kisi1 = 100 #kisi/mm
    kisi2 = 300
    kisi3 = 600
    l = 30
    x1 = [1.6,3.225,5.2,6.7,8.5,10.2]
    x2 = [4.9,10.65]
    x3 = [9.75]
    def menghitung_panjang_gelombang(x,l,kisi):
        d = 2/(kisi*10)
        panjang_gelombang = []
        for i in range(len(x)):
            panjang_gelombang.append((d/(i+1))*(x[i]/((x[i])**2+l**2)**(0.5))*10**(7))

        Tabel2 = pd.DataFrame({
            'orde':[i for i in range(len(x))]+['rata-rata'],
            'jarak x (cm)':x+[''],
            'panjang_gelombang (nm)':panjang_gelombang+[np.mean(panjang_gelombang)]
        })
        return Tabel2
    Tabela = menghitung_panjang_gelombang(x1,l,kisi1)
    Tabelb = menghitung_panjang_gelombang(x2,l,kisi2)
    Tabelc = menghitung_panjang_gelombang(x3,l,kisi3)
    Indeks_tabel(Tabela,'Tabel 1 Hasil dari 100 kisi/mm ')
    Indeks_tabel(Tabelb,'Tabel 1 Hasil dari 100 kisi/mm ')
    Indeks_tabel(Tabelc,'Tabel 1 Hasil dari 100 kisi/mm ')


    sub_judul("Menghitung Diameter Senar Gitar Dengan Prinsip Babinet")
    teks3 = r"""
Prinsip Babinet menyatakan bahwa sebuah celah pada kisi difraksi dapat diganti 
dengan bahan dengan ukuran yang sama pada peghalang celah tadi. 
Untuk Mengukur diamater senar menggunakan Prinsip Babinet

$D_s=\frac{λln}{x}$
"""

    st.write(teks2)
    kode2 = """
def menghitung_tebal_senar(y,l):
  panjang_gelombang_datasheet = 932e-6 #mm
  Ds = []
  for i in range(len(y)):
    Ds.append(panjang_gelombang_datasheet*l*(i+1)/((y[i])*(10)))

  Tabel3 = pd.DataFrame({
      'orde':[i for i in range(len(y))]+['rata rata'],
      'jarak orde terang (cm)':y+[''],
      # 'Diamater_senar(mm)':Ds+[np.mean(Ds)]
  })
  return Tabel3
Tabel3 = menghitung_tebal_senar(y,l)
"""
    y = [2,3.3,4.5,5.7,7,8.2,9.7,10.9,12.2,13.7]
    l = 6735 #meter
    def menghitung_tebal_senar(y,l):
        panjang_gelombang_datasheet = 932e-6 #mm
        Ds = []
        for i in range(len(y)):
            Ds.append(panjang_gelombang_datasheet*l*(i+1)/((y[i])*(10)))

        Tabel3 = pd.DataFrame({
            'orde':[i for i in range(len(y))]+['rata rata'],
            'jarak orde terang (cm)':y+[''],
            # 'Diamater_senar(mm)':Ds+[np.mean(Ds)]
        })
        return Tabel3
    Tabel3 = menghitung_tebal_senar(y,l)
    Indeks_tabel(Tabel3,"Hasil perhitungan Panjang Gelombang dari kisi difraksi")
