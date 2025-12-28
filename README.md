# Air Quality Analysis Dashboard (PM2.5)

## Deskripsi Projek
Proyek ini bertujuan untuk melakukan analisis data kualitas udara (PM2.5) secara end-to-end
menggunakan **Air Quality Dataset (PRSA – Shunyi Station)**.  
Analisis dilakukan melalui tahapan data wrangling, exploratory data analysis (EDA),
visualisasi data, serta penerapan machine learning (regresi) sebagai alat pendukung
pengambilan keputusan berbasis data, khususnya dalam konteks **kesehatan publik dan lingkungan**.

Hasil analisis disajikan dalam bentuk **Jupyter Notebook** dan **dashboard interaktif menggunakan Streamlit**.

---

## Dataset
- **Nama Dataset:** PRSA Air Quality Dataset (Shunyi Station)
- **Periode:** 2013 – 2017
- **Variabel Utama:** PM2.5, suhu (TEMP), kelembapan (DEWP), tekanan udara (PRES), kecepatan angin (WSPM)

Dataset digunakan untuk menganalisis pola polusi udara serta hubungannya dengan kondisi cuaca.

---

## Tujuan Analisis
1. Menganalisis pola temporal dan distribusi konsentrasi PM2.5.
2. Mengidentifikasi hubungan antara PM2.5 dan variabel meteorologi.
3. Menerapkan model regresi untuk mendukung analisis risiko kualitas udara.
4. Menyajikan insight dalam bentuk visualisasi dan dashboard interaktif.

---

## Fitur Dashboard Streamlit
- Ringkasan kualitas udara (Average, Max, High Risk Days)
- Tren PM2.5 berbasis waktu (interaktif)
- Distribusi PM2.5
- Analisis regresi PM2.5 berbasis variabel cuaca
- Insight berbasis konteks kebijakan dan kesehatan publik
- Filter rentang tahun dan variabel cuaca

