# Dashboard---Analisis-Capil-Kab.-Bener-Meriah
# Dashboard Analisis Kependudukan (Dukcapil) 🇮🇩📊

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)
![Status](https://img.shields.io/badge/Status-Educational%20Project-orange)

Selamat datang di **Dashboard Analisis Kependudukan Kabupaten Bener Meriah**, sebuah Dashboard berbasis web yang dirancang untuk membantu visualisasi, analisis, dan prediksi terkait layanan administrasi kependudukan di Kabupaten Bener Meriah Provinsi Aceh dimulai dari 21/08/2025 sampai dengan 09/02/2026 khususnya untuk Kartu Tanda Penduduk (KTP/).

Aplikasi ini menggabungkan **Data Science** dan **Machine Learning** untuk memberikan wawasan yang lebih dalam daripada sekadar laporan statistik biasa.

---

## 🌟 Fitur Utama

### 1. 🏠 Ringkasan Data (Overview)
- **Monitoring Real-time**: Memantau jumlah permohonan harian.
- **Status Dokumen**: Visualisasi persentase dokumen yang "Sudah Diambil" vs "Belum Diambil".
- **Filter Wilayah**: Analisis data spesifik per Kecamatan.

### 2. 🎯 Prediksi & Pemetaan (AI Powered)
- **🔮 Prediksi Pengambilan (Random Forest)**: 
  - Memasukkan data pemohon (Tanggal, Nama, Kecamatan) untuk memprediksi apakah dokumen akan diambil tepat waktu atau tertunda.
  - Berguna untuk prioritas notifikasi/pengingat kepada warga.
- **🗺️ Clustering Wilayah (K-Means)**:
  - Mengelompokkan kecamatan berdasarkan beban kerja dan efisiensi pengambilan.
  - Membantu alokasi sumber daya petugas di wilayah sibuk.

### 3. 🔮 Proyeksi Masa Depan (Forecasting)
- **Holt-Winters Algorithm**: Meramalkan jumlah permohonan layanan untuk 14 hari ke depan.
- Membantu perencanaan jadwal petugas dan stok blangko.

### 4. 💬 Kritik & Saran
- Fitur interaktif bagi pengguna untuk memberikan masukan.
- Data tersimpan otomatis dalam format `.csv`.

---

## 🛠️ Teknologi yang Digunakan
- **Bahasa Pemrograman**: [Python](https://www.python.org/)
- **Framework Web**: [Streamlit](https://streamlit.io/)
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-Learn (Random Forest, K-Means)
- **Forecasting**: Statsmodels (Holt-Winters)
- **Visualisasi**: Altair, Matplotlib, Seaborn

---

## 🚀 Cara Menjalankan (Lokal)

1.  **Clone Repository**
    ```bash
    git clone https://github.com/username-anda/dashboard-analisis-capil.git
    cd dashboard-analisis-capil
    ```

2.  **Install Dependencies**
    Pastikan Anda sudah menginstall Python, lalu jalankan:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Jalankan Aplikasi**
    ```bash
    streamlit run app.py
    ```
    Aplikasi akan terbuka otomatis di browser Anda (biasanya di `http://localhost:8501`).

---

## 📂 Struktur Proyek
- `app.py`: Kode utama aplikasi Streamlit.
- `requirements.txt`: Daftar pustaka yang dibutuhkan.
- `datacapil2025-2026.csv`: Dataset utama (Dummy/Simulasi untuk edukasi).
- `*.pkl`: Model Machine Learning yang sudah dilatih (Random Forest, KMeans, dll).

---

## 📝 Catatan Penting
> **Educational Purpose Only**: Aplikasi ini dibuat sebagai proyek pembelajaran. Data yang digunakan adalah data simulasi dan prediksi yang dihasilkan mungkin belum sepenuhnya akurat untuk penggunaan produksi skala besar. Kritik dan saran sangat diapresiasi untuk pengembangan lebih lanjut.

---

### Author
**Angga Tamara**  
*Data Science Enthusiast*
