# Analisis Sentimen Ulasan Aplikasi Google Play Store 🇮🇩

Repositori ini berisi proyek akhir untuk materi klasifikasi teks dan analisis sentimen menggunakan kombinasi teknik NLP (Natural Language Processing) dan Deep Learning. Proyek ini memproses dan mengklasifikasikan ulasan aplikasi berbahasa Indonesia dari Google Play Store ke dalam tiga kelas: **Positif**, **Netral**, dan **Negatif**.

## 📌 Deskripsi Proyek
Proyek ini dikembangkan untuk memenuhi standar submission (rating 5) dengan spesifikasi utama:
- **Scraping Dataset**: Lebih dari 10.000 ulasan ditarik secara langsung menggunakan pustaka `google-play-scraper` dari berbagai aplikasi populer di Indonesia (Gojek, Tokopedia, Traveloka, dsb).
- **Pendekatan Pelabelan Leksikon**: Melabeli ulasan berdasarkan kata kunci sentimen untuk mengatasi ambiguitas rating bintang.
- **Balancing Dataset**: Dataset diseimbangkan secara ketat menjadi **5.000 sampel** (1.666 per kelas) untuk mencegah masalah klasifikasi mayoritas.
- **Metrik Keberhasilan**: Semua model mencapai akurasi lebih dari **85%** pada set pengujian (*testing set*).

## 🚀 Skema Pelatihan Model
Tiga skema pelatihan berbeda diimplementasikan dan dievaluasi secara komprehensif:

1. **Skema 1: Conv1D (Convolutional Neural Network) + Embedding**
   - Menggunakan arsitektur 1D CNN yang sangat baik dalam menangkap pola lokal spasial pada teks.
   - **Akurasi Test**: **92.94%**

2. **Skema 2: Bi-LSTM (Bidirectional LSTM) + Word2Vec**
   - Melatih model Word2Vec kustom terlebih dahulu untuk *word embedding*, lalu memasukkannya ke dalam jaringan Bi-LSTM untuk menangkap konteks maju dan mundur.
   - **Akurasi Test**: **90.82%**

3. **Skema 3: SVM (Support Vector Machine) + TF-IDF**
   - Model *Machine Learning* tradisional yang tangguh dan ringan. Ekstraksi fitur menggunakan kombinasi unigram dan bigram (TF-IDF).
   - **Akurasi Test**: **89.81%**

## 📂 Struktur Repositori
- `analisis_sentimen.ipynb`: Notebook Jupyter utama yang berisi keseluruhan *pipeline*, dari EDA, preprocessing, pembuatan model, evaluasi, hingga fungsi inferensi. (Dapat dieksekusi langsung di Google Colab).
- `scraping.py`: Skrip mandiri berbasis Python untuk mengekstrak ribuan ulasan dari Google Play Store.
- `dataset_ulasan.csv`: Dataset mentah dan bersih berukuran >10K baris yang digunakan di proyek.
- `requirements.txt`: Daftar versi *library* Python yang dibutuhkan untuk mereproduksi proyek ini di lingkungan lokal.
- `*.keras` & `*.pkl`: Model deep learning dan machine learning terlatih, beserta objek *tokenizer* dan *TF-IDF Vectorizer*.
- `*.png`: Berbagai infografis (WordCloud, Confusion Matrix, grafik metrik riwayat pelatihan).

## ⚙️ Cara Penggunaan (Local / Google Colab)

**Google Colab (Sangat Disarankan)**
1. Unggah file `analisis_sentimen.ipynb` dan `dataset_ulasan.csv` ke dalam *session storage* Colab Anda.
2. *Notebook* sudah dilengkapi instruksi `!pip install Sastrawi -q` di awal sel.
3. Jalankan menu `Runtime -> Run All`.

**Lokal (macOS / Windows / Linux)**
1. Clone repositori ini:
   ```bash
   git clone https://github.com/wildaafn/analisis-sentimen-playstore.git
   cd analisis-sentimen-playstore
   ```
2. Buat Virtual Environment dan instal dependensi:
   ```bash
   python -m venv venv
   source venv/bin/activate  # (atau venv\Scripts\activate untuk Windows)
   pip install -r requirements.txt
   ```
3. Buka Jupyter Notebook dan jalankan `analisis_sentimen.ipynb`.

## 🧠 Uji Coba Inferensi Kustom
Tersedia satu sel khusus di bagian paling bawah *notebook* (`analisis_sentimen.ipynb`) bernama **"Inference – Prediksi Sentimen Teks Baru"**. Anda cukup mengganti *string* pada variabel teks lalu jalankan selnya untuk melihat klasifikasi teks baru dari ketiga model di atas!

---
*Proyek ini diajukan untuk tugas/submission course Machine Learning / Deep Learning.*
