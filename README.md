# 📊 Sentiment Analysis on Indonesian Google Play Store Reviews

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-lightgrey.svg)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Status-Completed-success.svg)]()

Repositori ini memuat *pipeline* lengkap (dari *scraping* data, prapemrosesan, pelatihan model, hingga fungsi inferensi) untuk memprediksi sentimen ulasan aplikasi berbahasa Indonesia di Google Play Store. Proyek ini membandingkan pendekatan Machine Learning tradisional dengan arsitektur Deep Learning modern, yang dikhususkan untuk memecahkan kasus ketidakseimbangan kelas (*class imbalance*) menggunakan pelabelan berbasis leksikon.

---

## 📌 Ringkasan Proyek
- **Tujuan**: Membangun pengklasifikasi teks otomatis (*Positif*, *Netral*, *Negatif*) pada >10.000 ulasan riil aplikasi populer di Indonesia (Gojek, Tokopedia, Traveloka, dsb).
- **Pendekatan Data**: Melibatkan *Lexicon-based sentiment labeling* untuk menetapkan sentimen sebenarnya, karena penilaian berdasarkan rating bintang (1-5) acapkali tidak konsisten dan tidak merepresentasikan teks dengan baik.
- **Balancing**: Dataset disegmentasi lalu diseimbangkan menjadi presisi **5.000 sampel** (Positif: 1.666, Netral: 1.667, Negatif: 1.667) untuk menunjang metrik klasifikasi berimbang yang tidak bias ke satu kelas mayoritas.

## 📈 Metrik Evaluasi Model
Tiga skema (algoritma & ekstraksi fitur) dibangun menggunakan pemisahan Data Latih (80%) dan Uji (20%). Semua metode berhasil melampaui tolok ukur proyek (>85% akurasi).

| Skema | Algoritma | Ekstraksi Fitur / Embedding | Akurasi Uji (Test Acc) | F1-Score (Macro) |
|---|---|---|:---:|:---:|
| **1** | LSTM | Keras Trainable Embedding | **91.52%** | 0.92 |
| **2** | Bi-LSTM | Word2Vec (Custom) | **90.92%** | 0.91 |
| **3** | Linear SVM | TF-IDF (Unigram & Bigram) | **89.81%** | 0.90 |

*Detail dari Confusion Matrix dan metrik pelatihan per *epoch* (*loss/accuracy curves*) bisa dilihat pada keluaran dalam *notebook* (`analisis_sentimen.ipynb`) atau file berekstensi `.png` di direktori ini.*

---

## 🛠 Instalasi dan Konfigurasi Lokal
Untuk mereplikasi hasil pelatihan dan menguji ulang performa model di sistem komputer Anda (*local machine*), harap ikuti prosedur di bawah ini.

1. **Kloning Repositori**
   ```bash
   git clone https://github.com/wildaafn/analisis-sentimen-playstore.git
   cd analisis-sentimen-playstore
   ```

2. **Pembuatan Virtual Environment** (Sangat Disarankan)
   ```bash
   python3 -m venv venv
   # macOS / Linux
   source venv/bin/activate
   # Windows
   venv\Scripts\activate
   ```

3. **Pemasangan Dependensi**
   Pastikan Anda menginstal semua dependensi proyek menggunakan pip:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Menjalankan Pipeline Proyek

Pilih salah satu dari metode eksekusi berikut sesuai kebutuhan Anda:

### Metode 1: Lewat Google Colab (Paling Praktis)
Metode ini tidak membutuhkan *setup* lokal dan meminimalisir masalah *environment*.
1. Unggah (*upload*) file `analisis_sentimen.ipynb` dan `dataset_ulasan.csv` ke Google Colab Anda.
2. (Opsional) Ganti *runtime* ke **T4 GPU** agar proses pelatihan *Neural Network* berlangsung kilat.
3. Klik **Runtime -> Run all**. Notebook sudah siap dengan sintaks internal untuk menginstal *Sastrawi*.

### Metode 2: Mode Interaktif Lokal (Jupyter Notebook)
Gunakan pendekatan ini apabila ingin bereksperimen dengan model di dalam IDE lokal (VS Code atau Jupyter server).
1. Pastikan Anda telah mengaktifkan *virtual environment*.
2. Jalankan server jupyter:
   ```bash
   jupyter notebook
   ```
3. Buka file `analisis_sentimen.ipynb`, modifikasi nilai atau jumlah *epoch*, kemudian jalankan semua *cells* secara berurutan.

### Metode 3: Ekstraksi Data Baru (Opsional)
Jika Anda ingin menyedot data/ulasan yang lebih baru dari aplikasi lain di Play Store, jalankan *script scraper*:
```bash
python scraping.py
```
*(Script ini akan mengompilasi ulasan dalam format `dataset_ulasan.csv` dan merekam meta datanya).*

---

## 🧠 Fitur Inferensi Langsung (Live Prediction)
Di bagian akhir dari `analisis_sentimen.ipynb`, Anda bisa menggunakan fungsi `predict_sentiment(text, model_name)` untuk menerjemahkan ulasan baru apa pun di luar dataset ke bentuk prediksi label (Positif/Netral/Negatif). Anda dapat membandingkan ketiga model secara berbarengan hanya dengan mengganti argumen kedua: `'lstm'`, `'bilstm'`, atau `'svm'`.

```python
teks_uji = "Aplikasi terus nge-bug setiap mau transfer. Mengecewakan."
print(predict_sentiment(teks_uji, 'lstm')) # Output: Negatif
```

---
*Proyek ini merupakan bukti nyata penerapan teknik pra-pemrosesan kalimat Indonesia (Sastrawi), Penyesuaian Bobot Kelas, Pembelajaran Mesin (ML), dan Pembelajaran Mendalam (DL) terstruktur yang terangkum dalam satu pipeline utuh.*
