import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go

def app():
    st.title("Intenferensi Pada cincin Newton")
    st.caption("20 April 2024")
    text1 = """ 
    <style>
        p {
            text-align: justify;
        }
    </style>
Beberapa langkah dalam praktikum ini yaitu :
- mengkonversi jarak antara dua buah titik untuk mengukur skalanya dalam function ukur_jarak dan mencari dua buah titik lain untuk mengukur diameter dalam function perbandingan
- setelah itu mencari $\lambda$ dengan persamaan $d^2=4m{\lambda}R$ dengan regresi linier
- dan terakhir mencari indeks bias pada medium air dan larutan gula
    """
    st.markdown(text1,unsafe_allow_html=True)
    text2 = """
Import modul

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

    text3 = """
Untuk membuat tabel diberikan dalam bentuk kode sebagai berikut

"""
    kode2 = """

def ukur_jarak(x,y):
  return (x**2+y**2)**0.5

def perbandingan(x,y,a):
  dn_p = (x**2+y**2)**0.5
  return (dn_p/a)*1

"""
# Teks yang ingin disembunyikan atau ditampilkan
    data = """
# a1 = (ukur_jarak(402,769))/5 # panjang acuan dalam satuan pixel (setara dengan 1 mm)
a1 = 189.36800152084828
a2 = 151.29864506994107
a3 = 116.76043850551437
a4 = 179.394537263541
a5 = 183.4017448117656



Tabel = np.array([
    ["","Udara_1","Udara_2","Udara_3","Air","Larutan gula"],
    ["d1 ",perbandingan(158,4,a1),perbandingan(184,2,a2),perbandingan(224,2,a3),perbandingan(196,2,a4),perbandingan(190,4,a5)],
    ["d2 ",perbandingan(262,44,a1),perbandingan(278,4,a2),perbandingan(300,4,a3),perbandingan(268,4,a4),perbandingan(270,4,a5)],
    ["d3 ",perbandingan(334,60,a1),perbandingan(348,4,a2),perbandingan(358,2,a3),perbandingan(314,6,a4),perbandingan(316,4,a5)],
    ["d4 ",perbandingan(496,78,a1),perbandingan(386,2,a2),perbandingan(410,4,a3),perbandingan(362,8,a4),perbandingan(366,6,a5)],
    ["d5 ",perbandingan(450,84,a1),perbandingan(442,12,a2),perbandingan(456,6,a3),perbandingan(398,8,a4),perbandingan(414,4,a5)],
    ["d6 ",perbandingan(496,92,a1),perbandingan(492,16,a2),perbandingan(500,4,a3),perbandingan(434,8,a4),perbandingan(448,10,a5)],
    ["d7 ",perbandingan(540,98,a1),perbandingan(528,20,a2),perbandingan(538,6,a3),perbandingan(470,10,a4),perbandingan(482,6,a5)],
    ["d8 ",perbandingan(578,104,a1),perbandingan(568,28,a2),perbandingan(578,8,a3),perbandingan(502,12,a4),perbandingan(514,6,a5)],
    ["d9 ",perbandingan(620,112,a1),perbandingan(604,34,a2),perbandingan(612,8,a3),perbandingan(526,12,a4),perbandingan(548,10,a5)],
    ["d10 ",perbandingan(656,114,a1),perbandingan(638,38,a2),perbandingan(644,10,a3),perbandingan(554,12,a4),perbandingan(576,10,a5)],
])
pd.DataFrame(data=Tabel[1:,1:],
            index=Tabel[1:,0],
            columns=Tabel[0,1:])

    """
    with st.expander("Berikut Data Hasil percobaan"):
        st.code(data)
    st.code(kode2)
    def ukur_jarak(x,y):
        return (x**2+y**2)**0.5

# a1 = (ukur_jarak(402,769))/5 # panjang acuan dalam satuan pixel (setara dengan 1 mm)
    a1 = 189.36800152084828
    a2 = 151.29864506994107
    a3 = 116.76043850551437
    a4 = 179.394537263541
    a5 = 183.4017448117656
    R = 1200


    def perbandingan(x,y,a):
        dn_p = (x**2+y**2)**0.5
        return (dn_p/a)*1

    st.write("Tabel hasil pengukuran diameter (mm)")
    Tabel = np.array([
        ["","Udara_1","Udara_2","Udara_3","Air","Larutan gula"],
        ["d1 ",perbandingan(158,4,a1),perbandingan(184,2,a2),perbandingan(224,2,a3),perbandingan(196,2,a4),perbandingan(190,4,a5)],
        ["d2 ",perbandingan(262,44,a1),perbandingan(278,4,a2),perbandingan(300,4,a3),perbandingan(268,4,a4),perbandingan(270,4,a5)],
        ["d3 ",perbandingan(334,60,a1),perbandingan(348,4,a2),perbandingan(358,2,a3),perbandingan(314,6,a4),perbandingan(316,4,a5)],
        ["d4 ",perbandingan(496,78,a1),perbandingan(386,2,a2),perbandingan(410,4,a3),perbandingan(362,8,a4),perbandingan(366,6,a5)],
        ["d5 ",perbandingan(450,84,a1),perbandingan(442,12,a2),perbandingan(456,6,a3),perbandingan(398,8,a4),perbandingan(414,4,a5)],
        ["d6 ",perbandingan(496,92,a1),perbandingan(492,16,a2),perbandingan(500,4,a3),perbandingan(434,8,a4),perbandingan(448,10,a5)],
        ["d7 ",perbandingan(540,98,a1),perbandingan(528,20,a2),perbandingan(538,6,a3),perbandingan(470,10,a4),perbandingan(482,6,a5)],
        ["d8 ",perbandingan(578,104,a1),perbandingan(568,28,a2),perbandingan(578,8,a3),perbandingan(502,12,a4),perbandingan(514,6,a5)],
        ["d9 ",perbandingan(620,112,a1),perbandingan(604,34,a2),perbandingan(612,8,a3),perbandingan(526,12,a4),perbandingan(548,10,a5)],
        ["d10 ",perbandingan(656,114,a1),perbandingan(638,38,a2),perbandingan(644,10,a3),perbandingan(554,12,a4),perbandingan(576,10,a5)],
    ])
    Tabelx=pd.DataFrame(data=Tabel[1:,1:],
                index=Tabel[1:,0],
                columns=Tabel[0,1:])
    st.table(Tabelx.head())

    st.divider()
    st.subheader("Membuat Grafik")
    st.write("Grafik ini didapat dari persamaan $d^2=4m{\lambda}R$ dimana gradiennya sama dengan $4{\lambda}R$ ")
    st.divider()
    
    # Membuat dua kolom
    col1, col2 = st.columns(2)

    # Menambahkan konten ke dalam kolom pertama
    Kode = """
def plot(Tabel,perulangan):
  R = 1
  fungsi_panjang_gelombang = lambda d :d**2/(4*R)
  tinggi_tabel,lebar_tabel = Tabel.shape
  d_kuadrat = np.array([(float(Tabel[m,perulangan]))**2 for m in range(2,tinggi_tabel)])# d^2 sebagai sumbu y
  m = np.array([m for m in range(2,tinggi_tabel)])# orde pita sebagai sumbu x
  model = LinearRegression()# model regresi
  m= m.reshape(-1,1)
  model.fit(m,d_kuadrat)#melatih model regresi
  prediksi = model.predict(m)
  return model.coef_,model.intercept_,m,d_kuadrat,prediksi

def plotting():
  tinggi , lebar = Tabel.shape
  A = []# menyimpan semua data plot
  for i in range(1,lebar):
    A.append(plot(Tabel,i))
  return A
        """
        
    st.code(Kode)
    def plot(Tabel,perulangan):
        R = 1
        fungsi_panjang_gelombang = lambda d :d**2/(4*R)
        tinggi_tabel,lebar_tabel = Tabel.shape
        d_kuadrat = np.array([(float(Tabel[m,perulangan]))**2 for m in range(2,tinggi_tabel)])# d^2 sebagai sumbu y
        m = np.array([m for m in range(2,tinggi_tabel)])# orde pita sebagai sumbu x
        model = LinearRegression()# model regresi
        m= m.reshape(-1,1)
        model.fit(m,d_kuadrat)#melatih model regresi
        prediksi = model.predict(m)
        return model.coef_,model.intercept_,m,d_kuadrat,prediksi
    def plotting():
        tinggi , lebar = Tabel.shape
        A = []# menyimpan semua data plot
        for i in range(1,lebar):
            A.append(plot(Tabel,i))
        return A
        

    tinggi , lebar = Tabel.shape
    i =  3
    # st.pyplot(fig)

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=np.array([i[0] for i in (((plotting())[i])[2])]), y=((plotting())[i])[4], mode='lines', name='Garis'))
    fig.add_trace(go.Scatter(x=np.array([i[0] for i in (((plotting())[i])[2])]), y=((plotting())[i])[4], mode='markers', name='Scatter'))
    
    # Menambahkan judul dan label sumbu
    fig.update_layout(title='Contoh Grafik Pada Udara Perulangan ke 3',
                    xaxis_title="orde m",
                    yaxis_title="diameter kuadrat (mm)")
    st.plotly_chart(fig)
    st.latex(f"y={(np.round((plotting()[i])[0],3))[0]}x+{round(((plotting()[i])[1]),3)}")
    st.write("Menghitung panjang gelombang dan indeks medium")
    st.caption("Tabel hasil Perhitungan panjang gelombang")
    kode3 = """
Hasil = np.array([
    ['','Lambda (nm)','intercept']
])

medium = ["Udara_1","Udara_2","Udara_3","Air","Larutan gula"]

for i in range(0,lebar-1):
  baris_baru = np.array([medium[i],(((plotting()[i])[0])[0])/(4*R)*10**(6),((plotting()[i])[1])])
  Hasil =  np.vstack([Hasil,baris_baru])
pd.DataFrame(data=Hasil[1:,1:],
            index=Hasil[1:,0],
            columns=Hasil[0,1:])
"""
    st.code(kode3)
    Hasil = np.array([
        ['','Lambda (nm)','intercept']
    ])

    medium = ["Udara_1","Udara_2","Udara_3","Air","Larutan gula"]

    for i in range(0,lebar-1):
        baris_baru = np.array([medium[i],(((plotting()[i])[0])[0])/(4*R)*10**(6),((plotting()[i])[1])])
        Hasil =  np.vstack([Hasil,baris_baru])
    
    st.table(pd.DataFrame(data=Hasil[1:,1:],index=Hasil[1:,0],columns=Hasil[0,1:]))
    st.caption("Menghitung indeks medium")
    kode4= """
def mencari_indeks(perulangan,Tabel):
    Tabel_indeks_medium = np.array([["","d1","d2","d3","d4","d5","d6","d7","d8","d9","d10","rata rata"]])
    for m in range(1,lebar-3):
      B = [np.round(((float(Tabel[i,perulangan]))/float(Tabel[i,-m]))**2,4) for i in range(1,tinggi)]
      B = [Tabel[0,3+m]+f" dengan udara{perulangan}"]+B+[np.round(np.mean(B),4)]
      Tabel_indeks_medium = np.vstack([Tabel_indeks_medium,B])
    return pd.DataFrame(data=Tabel_indeks_medium[1:,1:],
            index=Tabel_indeks_medium[1:,0],
            columns=Tabel_indeks_medium[0,1:])
"""
    st.code(kode4)
    def mencari_indeks(perulangan,Tabel):
        Tabel_indeks_medium = np.array([["","d1","d2","d3","d4","d5","d6","d7","d8","d9","d10","rata rata"]])
        for m in range(1,lebar-3):
            B = [np.round(((float(Tabel[i,perulangan]))/float(Tabel[i,-m]))**2,4) for i in range(1,tinggi)]
            B = [Tabel[0,3+m]+f" dengan udara{perulangan}"]+B+[np.round(np.mean(B),4)]
            Tabel_indeks_medium = np.vstack([Tabel_indeks_medium,B])
        return pd.DataFrame(data=Tabel_indeks_medium[1:,1:],
                index=Tabel_indeks_medium[1:,0],
                columns=Tabel_indeks_medium[0,1:])
    Tabel1 = mencari_indeks(1,Tabel)
    Tabel2 = mencari_indeks(2,Tabel)
    Tabel3 = mencari_indeks(3,Tabel)
    st.caption('Menghitung indeks medium pada variasi udara 1')
    st.table(Tabel1)
    st.caption('Menghitung indeks medium pada variasi udara 2')
    st.table(Tabel2)
    st.caption('Menghitung indeks medium pada variasi udara 3')
    st.table(Tabel1)

    