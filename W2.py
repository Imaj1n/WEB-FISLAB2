import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
import plotly.graph_objects as go
import plotly.express as px

def gerak_teredam(t, y,m,b,k):
    """
    Fungsi f(t, y) yang merupakan fungsi diferensial dari gerak osilasi teredam.
    Di sini, kita berikan contoh persamaan diferensial y'' + 2ζω0y' + ω0^2y = 0
    dengan ω0 adalah frekuensi eigen dan ζ adalah rasio redaman.
    """
    omega_0 = 1  # Frekuensi eigen
    zeta = 0.1   # Rasio redaman
    return np.array([y[1], -(b/m)*y[1] - (k/m) * y[0]])



def app():
    st.title("Osilasi Teredam")
    st.caption("20 April 2024")
    text1 = r""" 
Petunjuk simbolik regresi eksponensial
- Popt : Parameter optimalisasi untuk regresi yang paling cocok terutama setiap parameter a dan b
- pcov : Matriks covarian, menunjukkan ketidakpastian dalam paramater regresi
- a : Intercept
- b : Tingkat pertumbuhan
Model regresi Eksponensial ini dimodelkan sebagai :

$y = ae^{bx}$

Persamaan gerak dari osilasi teredam ini diberikan oleh :

$\ddot{x}+\frac{b}{m}\dot{x}+\frac{k}{m}x=0$

Dengan solusi x pada kasus underdamped adalah

$x(t)=Ae^{\frac{-b}{2m}t}\cos{ωt}$

Untuk mencari b, plot $A_0$ dan $t$ dengan persamaan :

$A(t)=A_0 e^{\frac{-b}{2m}t}$

    """
    st.markdown(text1,unsafe_allow_html=True)
    text2 = """
Import modul

"""
    st.divider()
    st.subheader(text2)
    st.divider()
    modul = """
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sympy import *
from scipy.optimize import curve_fit

    """
    st.code(modul)

    text3 = """
Persamaan gerak dari osilasi teredam ini diberikan oleh :

$\ddot{x}+\frac{b}{m}\dot{x}+\frac{k}{m}x=0$

Dengan solusi x pada kasus underdamped adalah

$x(t)=Ae^{\frac{-b}{2m}t}\cos{ωt}$

Untuk mencari b, plot $A_0$ dan $t$ dengan persamaan :

$A(t)=A_0 e^{\frac{-b}{2m}t}$

"""
    kode2 = """

def exponential_regression(x, a, b):
    return a * np.exp(b * x)

def draw_graph(time1,time2,DeltaX1,DeltaX2,massa1,massa2,medium):
  popt1, pcov1 = curve_fit(exponential_regression, time1,DeltaX1)
  popt2, pcov2 = curve_fit(exponential_regression, time2,DeltaX2)
  A01=popt1[0]
  B1=popt1[1]
  b1 = -B1*2*massa1
  A02=popt2[0]
  B2=popt2[1]
  b2 = -B2*2*massa2
  plt.plot(time1, exponential_regression(time1, *popt1), 'b-', label=f'Regresi Eksponensial massa 1')
  plt.scatter(time1,DeltaX1,c='b')
  plt.plot(time2, exponential_regression(time2, *popt2), 'r-', label=f'Regresi Eksponensial massa 2')
  plt.scatter(time2,DeltaX2,c='r')
  plt.xlabel('Waktu (s)')
  plt.ylabel('Perubahan Posisi (m)')
  plt.legend()
  # Membuat string persamaan regresi dengan LaTeX
  equation1 = fr'$\mathrm{{massa 1}} : y = {A01:.2f} \cdot e^{{{B1:.2f}x}}$'
  equation2 = fr'$\mathrm{{massa 2}}:y = {A02:.2f} \cdot e^{{{B2:.2f}x}}$'
  # Menampilkan persamaan regresi di legenda dengan font LaTeX
  plt.legend(fontsize='medium')
  plt.text(0.5, 10.5, equation1, color='black', fontsize='medium', fontname='Latin Modern Math')
  plt.text(0.5, 10, equation2, color='black', fontsize='medium', fontname='Latin Modern Math')
  plt.title(f'Regresi Eksponensial variasi {medium}')
  plt.tight_layout()
  plt.grid(True)
  plt.show()
  return A01,A02,b1,b2
A01,A02,b1,b2 = draw_graph(time1,time2,Deltax1,Deltax2,m1,m2,'udara')

"""
    st.code(kode2)


    Deltax1=np.array([7.5,7.4,7,7.3,7.2,7])#udara1
    time1=np.array([0,0.85,1.72,2.53,3.41,4.22])


    Deltax2 = np.array([11.3,11.5,11.4,11.3,11,11.2])#udara2
    time2 = np.array([0,1.02,2.01,2.97,3.96,4.91])

    Deltax3 = np.array([10,5.5,3.6,2.8,2.3,2])#air1
    time3 = np.array([0,1.03,1.8,2.69,3.57,4.42])

    Deltax4 = np.array([11.3,5.3,3.5,2.4,1.9,1.4])#air2
    time4 = np.array([0,1.1,2.1,3.12,4.13,5.12])


    k = 3 # didapat dari pengukuran video
    m1 = 171e-3
    m2 = 233e-3
    r = 0.014

    def membuat_tabel(Deltax1,time1,Deltax2,time2):
        Tabel = pd.DataFrame({
            '':["",""]+["Deltax"]+["","",""]+["",""]+["waktu"]+["","",""],
            'Udara':np.concatenate((Deltax1,time1)),
            'Air':np.concatenate((Deltax2,time2))
        })
        return Tabel
    Tabel1 = membuat_tabel(Deltax1,time1,Deltax3,time3)
    Tabel2 = membuat_tabel(Deltax3,time3,Deltax4,time4)
    with st.expander("Berikut Data Hasil percobaan"):
        st.table(Tabel1)
        st.caption('Variasi Air')
        st.table(Tabel2)

    def exponential_regression(x, a, b):
        return a * np.exp(b * x)

    def draw_graph(time1,time2,DeltaX1,DeltaX2,massa1,massa2,medium):
        popt1, pcov1 = curve_fit(exponential_regression, time1,DeltaX1)
        popt2, pcov2 = curve_fit(exponential_regression, time2,DeltaX2)
        A01=popt1[0]
        B1=popt1[1]
        b1 = -B1*2*massa1
        A02=popt2[0]
        B2=popt2[1]
        b2 = -B2*2*massa2
        plt.plot(time1, exponential_regression(time1, *popt1), 'b-', label=f'Regresi Eksponensial massa 1')
        plt.scatter(time1,DeltaX1,c='b')
        plt.plot(time2, exponential_regression(time2, *popt2), 'r-', label=f'Regresi Eksponensial massa 2')
        plt.scatter(time2,DeltaX2,c='r')
        plt.xlabel('Waktu (s)')
        plt.ylabel('Perubahan Posisi (m)')
        plt.legend()
        # Membuat string persamaan regresi dengan LaTeX
        equation1 = fr'$\mathrm{{massa 1}} : y = {A01:.2f} \cdot e^{{{B1:.2f}x}}$'
        equation2 = fr'$\mathrm{{massa 2}}:y = {A02:.2f} \cdot e^{{{B2:.2f}x}}$'
        # Menampilkan persamaan regresi di legenda dengan font LaTeX
        plt.legend(fontsize='medium')
        plt.text(0.5, 10.5, equation1, color='black', fontsize='medium', fontname='Latin Modern Math')
        plt.text(0.5, 10, equation2, color='black', fontsize='medium', fontname='Latin Modern Math')
        plt.title(f'Regresi Eksponensial variasi {medium}')
        plt.tight_layout()
        plt.grid(True)
        plt.show()
        return A01,A02,b1,b2,B1,B2
    A03,A04,b3,b4,B1,B2 = draw_graph(time3,time4,Deltax3,Deltax4,m1,m2,'air')
    A01,A02,b1,b2,B1,B2  = draw_graph(time1,time2,Deltax1,Deltax2,m1,m2,'udara')

    st.divider()
    st.subheader("Menghitung viskositas dan frekuensi sudut")
    st.divider()
    st.write("Persamaan untuk menghitung viskositas adalah")
    st.latex(r'\eta=\frac{b}{6{\pi}r}')
    st.write('Untuk menghitung frekuensi sudut natural')
    st.latex(r'\omega=\sqrt{(\frac{k}{m})^2-(\frac{b}{2m})^2}')

    kode3="""
def menghitung_eta(b,r):
  pi = np.pi
  eta = b/(6*pi*r)
  return eta

def menghitung_frekuensi_sudut(k,m,b):
  omega = ((k/m)**2-(b/(2*m))**2)**(0.5)
  return omega
"""
    def menghitung_eta(b,r):
        pi = np.pi
        eta = b/(6*pi*r)
        return eta
    def menghitung_frekuensi_sudut(k,m,b):
        omega = ((k/m)**2-(b/(2*m))**2)**(0.5)
        return omega
    
    
    st.code(kode3)
    # st.write("perhitungan dari kode diatas menghasilkan viskositas dan frekuensi sudut sama dengan")
    # st.latex(f"\eta={np.around(menghitung_eta(b3,r),3)}")
    # st.latex(f"\omega={np.around(menghitung_frekuensi_sudut(k,m1,b3),3)}")
    def menghitung_tabel():
        viskositas = [menghitung_eta(b1,r),menghitung_eta(b2,r),menghitung_eta(b3,r),menghitung_eta(b4,r)]
        omega1 = [menghitung_frekuensi_sudut(k,m1,b1),menghitung_frekuensi_sudut(k,m1,b2),menghitung_frekuensi_sudut(k,m1,b3),menghitung_frekuensi_sudut(k,m1,b4)]
        omega2 = [menghitung_frekuensi_sudut(k,m2,b1),menghitung_frekuensi_sudut(k,m2,b2),menghitung_frekuensi_sudut(k,m2,b3),menghitung_frekuensi_sudut(k,m2,b4)]
        Tabel = pd.DataFrame({
            '':['Udara 1','Udara 2','Air 1','Air 2','rata rata'],
            'viskositas (Pa.s)':viskositas+[np.mean(viskositas)],
            'omega 1(rad/s)':omega1+[np.mean(omega1)],
            'omega 2(rad/s)':omega2+[np.mean(omega2)],
        })
        return Tabel
    Tabela = menghitung_tabel()
    st.caption("Tabel Hasil perhitungan viskositas dan frekuensi osilasi redaman")
    st.table(Tabela)
    st.divider()
    st.subheader("Menentukan gerak osilasi")
    st.divider()
    kode4="""
def draw_graph2(time1,time2,DeltaX1,DeltaX2,m1,m2):
  popt1, pcov1 = curve_fit(exponential_regression, time1, DeltaX1)
  popt2, pcov2 = curve_fit(exponential_regression, time2, DeltaX2)
  e = np.e
  times1 = (np.linspace(0,time1[-1],10000))
  times2 = (np.linspace(0,time1[-1],10000))
  A01=popt1[0]
  B1=popt1[1]
  b1 = -B1*2*m1
  A02=popt2[0]
  B2=popt2[1]
  b2 = -B2*2*m2
  plt.plot(times1, exponential_regression(times1, *popt1)*(np.cos(omega*times1)),'r-', label='osilasi benda 1')
  plt.plot(time1, exponential_regression(time1, *popt1), '--', label='Regresi Eksponensial 1')
  plt.plot(times2, exponential_regression(times2, *popt2)*(np.cos(omega*times1)),color='black', linestyle='-', label='osilasi benda 2')
  plt.plot(time2, exponential_regression(time2, *popt2), 'g--', label='Regresi Eksponensial 2')
  plt.xlabel('Waktu (s)')
  plt.ylabel('Perubahan Posisi (m)')
  plt.legend()
  plt.title('Perbandingan kurva dari DeltaX i terhadap Waktu')
  plt.grid(True)
  plt.show()
""" 
    st.code(kode4)
    def draw_graph2(time,DeltaX,massa,A0,b):
        omega = menghitung_frekuensi_sudut(k,massa,b)
        popt, pcov = curve_fit(exponential_regression, time,DeltaX)
        times = np.linspace(0,time[-1],1000)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=time,y=exponential_regression(time, *popt),mode='lines', name='Regresi'))
        fig.add_trace(go.Scatter(x=time,y=exponential_regression(time, *popt),mode='markers', name='Data'))
        fig.add_trace(go.Scatter(x=times,y=exponential_regression(times, *popt)*(np.cos(omega*times)),mode='markers', name='Osilasi'))
        fig.update_layout(title='Regresi Eksponensial dari DeltaX i terhadap Waktu',
                   xaxis_title='waktu',
                   yaxis_title='posisi')
        return fig
    fig= draw_graph2(time3,Deltax3,m1,A03,b3)
    persamaan_grafik2 = f"y = {np.round(A03,3)}e^{{{np.round(B1,3)}t}}"
    st.caption('contoh grafik hasil variasi pengukuran air')
    st.plotly_chart(fig)
    st.latex(persamaan_grafik2)

    #Simulasi
    st.divider()
    st.subheader("Visualisasi Grafik Osilasi Teredam Sederhana")
    st.divider()
    text4 = r"""
secara general, persamaan osilasi teredam ini diberikan dalam bentuk persamaan

$\frac{d^y}{dx^2}+\frac{b}{m}\frac{dy}{dx}+\frac{k}{m}y=0$

Untuk menyelesaikan persamaan diferensial diatas digunakan metode Runge kutta
"""
    st.write(text4)
    kode5 = """
def gerak_teredam(t, y,m,b,k):
    #
    Fungsi f(t, y) yang merupakan fungsi diferensial dari gerak osilasi teredam.
    Di sini, kita berikan contoh persamaan diferensial y'' + 2ζω0y' + ω0^2y = 0
    dengan ω0 adalah frekuensi eigen dan ζ adalah rasio redaman.
    #
    omega_0 = 1  # Frekuensi eigen
    zeta = 0.1   # Rasio redaman
    return np.array([y[1], -(b/m)*y[1] - (k/m) * y[0]])
def runge_kutta_4(gerak_teredam, a, b, y0, N):
        #
        Implementasi metode Runge-Kutta orde empat untuk menyelesaikan persamaan diferensial
        y'' + 2ζω0y' + ω0^2y = 0, dimana a dan b adalah batas waktu,
        y0 adalah nilai awal [y, y'], dan N adalah jumlah langkah.
        #
        h = (b - a) / N
        t = a
        y = y0
        solusi = [y0]
        for i in range(N):
            k1 = h * gerak_teredam(t, y,m,gamma,konstanta)
            k2 = h * gerak_teredam(t + 0.5*h, y + 0.5*k1,m,b,k)
            k3 = h * gerak_teredam(t + 0.5*h, y + 0.5*k2,m,b,k)
            k4 = h * gerak_teredam(t + h, y + k3,m,gamma,m,b,k)
            y = y + (k1 + 2*k2 + 2*k3 + k4) / 6
            t = a + (i + 1) * h
            solusi.append(y)
        return np.array(solusi)
"""
    st.code(kode5)
    m = st.slider('massa', min_value=0, max_value=100, value=50, step=1)
    gamma = st.slider('koefisien gaya geseka', min_value=0, max_value=100, value=10, step=1)
    konstanta = st.slider("Konstanta Pegas", min_value=0, max_value=100, value=50, step=1)
    def runge_kutta_4(gerak_teredam, a, b, y0, N):
        """
        Implementasi metode Runge-Kutta orde empat untuk menyelesaikan persamaan diferensial
        y'' + 2ζω0y' + ω0^2y = 0, dimana a dan b adalah batas waktu,
        y0 adalah nilai awal [y, y'], dan N adalah jumlah langkah.
        """
        h = (b - a) / N
        t = a
        y = y0
        solusi = [y0]
        for i in range(N):
            k1 = h * gerak_teredam(t, y,m,gamma,konstanta)
            k2 = h * gerak_teredam(t + 0.5*h, y + 0.5*k1,m,gamma,konstanta)
            k3 = h * gerak_teredam(t + 0.5*h, y + 0.5*k2,m,gamma,konstanta)
            k4 = h * gerak_teredam(t + h, y + k3,m,gamma,konstanta)
            y = y + (k1 + 2*k2 + 2*k3 + k4) / 6
            t = a + (i + 1) * h
            solusi.append(y)
        return np.array(solusi)
    # Contoh penggunaan
    a = 0  # Batas bawah waktu
    b = 100  # Batas atas waktu
    y0 = np.array([1, 0])  # Nilai awal [y, y']
    N = 10000  # Jumlah langkah

    solusi = runge_kutta_4(gerak_teredam, a, b, y0, N)
    t = np.linspace(a, b, N+1)

    # Plot hasil
    plt.plot(t, solusi[:, 0], label='Posisi y(t)')
    fig = px.line(x=t,y=solusi[:, 0], labels={'x': 'waktu', 'y': 'Perpindahan arah y'},title='Plot Osilasi teredam')
    st.plotly_chart(fig)
    st.write("untuk kode lebih lengkap,kunjungi google colab dari penulis berikut ini")
    st.markdown("[Tautan Google colab penulis]( https://colab.research.google.com/drive/1uXpXR_uv0NsedrQYDK2ir2UpibKRVi6Q?usp=sharing)")
    
    # Membuat dua kolom
    col1, col2 = st.columns(2)

    