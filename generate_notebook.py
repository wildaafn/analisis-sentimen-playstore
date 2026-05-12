import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}

def md(src): return nbf.v4.new_markdown_cell(src)
def code(src): return nbf.v4.new_code_cell(src)

cells = []

cells.append(md("""# Analisis Sentimen Ulasan Aplikasi Google Play Store
**Proyek Submission – Deep Learning & NLP**

| Atribut | Detail |
|---|---|
| Dataset | Ulasan aplikasi populer Indonesia (Gojek, Shopee, Traveloka, Bukalapak) |
| Jumlah Data | ≥ 7.000 ulasan |
| Kelas | 3 kelas: **Positif**, **Netral**, **Negatif** |
| Skema 1 | LSTM + Embedding – split 80/20 |
| Skema 2 | Bi-LSTM + Word2Vec – split 80/20 |
| Skema 3 | SVM + TF-IDF – split 70/30 |
"""))

cells.append(md("## 1. Import Library"))
cells.append(code("""\
import os, re, warnings, joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
warnings.filterwarnings('ignore')

import nltk
for pkg in ['stopwords','punkt','punkt_tab']:
    nltk.download(pkg, quiet=True)
from nltk.corpus import stopwords

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (classification_report, accuracy_score,
                              confusion_matrix, ConfusionMatrixDisplay)
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE

from gensim.models import Word2Vec

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Embedding, LSTM, Bidirectional, Dense,
                                     Dropout, SpatialDropout1D)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

print("TensorFlow:", tf.__version__)
np.random.seed(42)
tf.random.set_seed(42)
"""))

cells.append(md("## 2. Load Dataset"))
cells.append(code("""\
df = pd.read_csv('dataset_ulasan.csv')
print(f"Shape awal: {df.shape}")
df = df.dropna(subset=['review']).reset_index(drop=True)
print(f"Shape setelah dropna: {df.shape}")
print("\\nContoh data:")
df[['app_name','review','rating']].head(5)
"""))

cells.append(md("""## 3. Pelabelan Data

Strategi pelabelan untuk distribusi yang lebih seimbang:
- **Positif** : rating 5
- **Netral**  : rating 3–4  
- **Negatif** : rating 1–2
"""))
cells.append(code("""\
def assign_label(r):
    if r == 5:       return 'Positif'
    elif r in [3,4]: return 'Netral'
    else:            return 'Negatif'

df['label'] = df['rating'].apply(assign_label)

print("Distribusi label:")
print(df['label'].value_counts())

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
colors = ['#e74c3c','#f39c12','#2ecc71']
df['label'].value_counts().plot(kind='bar', ax=axes[0], color=colors)
axes[0].set_title('Distribusi Label', fontsize=13)
axes[0].set_xlabel('Label'); axes[0].set_ylabel('Jumlah')
axes[0].tick_params(axis='x', rotation=0)

df['app_name'].value_counts().plot(kind='bar', ax=axes[1], color='steelblue')
axes[1].set_title('Distribusi per Aplikasi', fontsize=13)
axes[1].tick_params(axis='x', rotation=30)
plt.tight_layout()
plt.savefig('distribusi_label.png', dpi=120)
plt.show()
"""))

cells.append(md("## 4. Preprocessing Teks"))
cells.append(code("""\
factory    = StemmerFactory()
stemmer    = factory.create_stemmer()
stop_words = set(stopwords.words('indonesian'))

slang_dict = {
    'gak':'tidak','ga':'tidak','ngga':'tidak','enggak':'tidak',
    'bgt':'banget','bngt':'banget','bget':'banget',
    'yg':'yang','dgn':'dengan','utk':'untuk','krn':'karena',
    'klo':'kalau','kl':'kalau','tp':'tapi','tpi':'tapi',
    'sy':'saya','gw':'saya','gue':'saya',
    'lg':'lagi','sdh':'sudah','udah':'sudah','udh':'sudah',
    'blm':'belum','jd':'jadi','dpt':'dapat',
    'mantap':'bagus','mantul':'bagus',
    'susah':'sulit','jelek':'buruk',
    'app':'aplikasi','apk':'aplikasi',
}

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'http\\S+|www\\S+', '', text)
    text = re.sub(r'[^a-z\\s]', ' ', text)
    text = re.sub(r'\\s+', ' ', text).strip()
    tokens = text.split()
    tokens = [slang_dict.get(t, t) for t in tokens]
    tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
    tokens = [stemmer.stem(t) for t in tokens]
    return ' '.join(tokens)

print("Memulai preprocessing... (mungkin butuh beberapa menit)")
from tqdm import tqdm
tqdm.pandas()
df['clean_text'] = df['review'].progress_apply(preprocess)
df = df[df['clean_text'].str.strip() != ''].reset_index(drop=True)
print(f"Jumlah data bersih: {len(df)}")
print("\\nContoh hasil preprocessing:")
for i in [0, 1, 2]:
    print(f"  Asli   : {str(df['review'].iloc[i])[:70]}")
    print(f"  Bersih : {df['clean_text'].iloc[i][:70]}")
    print()
"""))

cells.append(code("""\
for label in ['Positif','Netral','Negatif']:
    subset = df[df['label']==label]['clean_text']
    text   = ' '.join(subset)
    if not text.strip():
        continue
    wc = WordCloud(width=800, height=300, background_color='white',
                   max_words=100, colormap='plasma').generate(text)
    plt.figure(figsize=(10, 3))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'WordCloud – {label}', fontsize=13)
    plt.tight_layout()
    plt.savefig(f'wordcloud_{label.lower()}.png', dpi=100)
    plt.show()
print("WordCloud selesai.")
"""))

cells.append(md("## 5. Ekstraksi Fitur & Persiapan Label"))
cells.append(code("""\
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['label_enc'] = le.fit_transform(df['label'])
print("Kelas:", le.classes_)   # urutan alfabet: Negatif=0, Netral=1, Positif=2

# TF-IDF (untuk Skema 3 – SVM)
tfidf = TfidfVectorizer(max_features=15000, ngram_range=(1,2), sublinear_tf=True)
X_tfidf = tfidf.fit_transform(df['clean_text'])
y        = df['label_enc'].values
print("Shape TF-IDF:", X_tfidf.shape)

joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
joblib.dump(le,    'label_encoder.pkl')
"""))

# ── Skema 3: SVM ──────────────────────────────────────────────────────────────
cells.append(md("## 6. Skema 3 – SVM + TF-IDF (Split 70/30) + SMOTE"))
cells.append(code("""\
X_tr3, X_te3, y_tr3, y_te3 = train_test_split(
    X_tfidf, y, test_size=0.30, random_state=42, stratify=y)

# SMOTE untuk oversampling kelas minor
sm = SMOTE(random_state=42)
X_tr3_res, y_tr3_res = sm.fit_resample(X_tr3, y_tr3)
print("Distribusi setelah SMOTE:", dict(zip(*np.unique(y_tr3_res, return_counts=True))))

svm = LinearSVC(max_iter=5000, C=0.5)
svm.fit(X_tr3_res, y_tr3_res)
joblib.dump(svm, 'model_svm.pkl')

acc_tr3 = accuracy_score(y_tr3_res, svm.predict(X_tr3_res))
acc_te3 = accuracy_score(y_te3,     svm.predict(X_te3))
print(f"\\n=== Skema 3 – SVM + TF-IDF (70/30) + SMOTE ===")
print(f"Train Accuracy : {acc_tr3:.4f}")
print(f"Test  Accuracy : {acc_te3:.4f}")
print("\\nClassification Report (Test):")
print(classification_report(y_te3, svm.predict(X_te3), target_names=le.classes_))

cm3 = confusion_matrix(y_te3, svm.predict(X_te3))
disp = ConfusionMatrixDisplay(cm3, display_labels=le.classes_)
fig, ax = plt.subplots(figsize=(5,4))
disp.plot(ax=ax, colorbar=False, cmap='Blues')
ax.set_title('Confusion Matrix – SVM')
plt.tight_layout(); plt.savefig('cm_svm.png', dpi=120); plt.show()
"""))

# ── Deep Learning tokenizer ────────────────────────────────────────────────────
cells.append(md("## 7. Tokenisasi untuk Model Deep Learning"))
cells.append(code("""\
MAX_VOCAB = 20000
MAX_LEN   = 120
EMBED_DIM = 128

tokenizer_dl = Tokenizer(num_words=MAX_VOCAB, oov_token='<OOV>')
tokenizer_dl.fit_on_texts(df['clean_text'])
sequences = tokenizer_dl.texts_to_sequences(df['clean_text'])
X_seq = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')
y_cat = to_categorical(y, num_classes=3)

# Hitung class weight
class_weights_arr = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weights = {i: w for i, w in enumerate(class_weights_arr)}
print("Class weights:", class_weights)

joblib.dump(tokenizer_dl, 'tokenizer_dl.pkl')
print("X_seq shape:", X_seq.shape)
"""))

# ── Skema 1: LSTM ─────────────────────────────────────────────────────────────
cells.append(md("## 8. Skema 1 – LSTM + Embedding (Split 80/20)"))
cells.append(code("""\
X_tr1, X_te1, y_tr1, y_te1 = train_test_split(
    X_seq, y_cat, test_size=0.20, random_state=42, stratify=y)

model_lstm = Sequential([
    Embedding(MAX_VOCAB, EMBED_DIM, input_length=MAX_LEN),
    SpatialDropout1D(0.25),
    LSTM(128, dropout=0.3, recurrent_dropout=0.2, return_sequences=True),
    LSTM(64,  dropout=0.2, recurrent_dropout=0.1),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(3,  activation='softmax'),
], name='LSTM_Model')

model_lstm.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])
model_lstm.summary()

cbs1 = [
    EarlyStopping(patience=5, restore_best_weights=True, monitor='val_accuracy'),
    ReduceLROnPlateau(patience=3, factor=0.5, monitor='val_loss', verbose=0),
]

history1 = model_lstm.fit(
    X_tr1, y_tr1, epochs=25, batch_size=64,
    validation_data=(X_te1, y_te1),
    class_weight=class_weights,
    callbacks=cbs1, verbose=1)

model_lstm.save('model_lstm.keras')
"""))

cells.append(code("""\
y_pred1    = np.argmax(model_lstm.predict(X_te1, verbose=0), axis=1)
y_true1    = np.argmax(y_te1, axis=1)
y_pred1_tr = np.argmax(model_lstm.predict(X_tr1, verbose=0), axis=1)
y_true1_tr = np.argmax(y_tr1, axis=1)

acc_tr1 = accuracy_score(y_true1_tr, y_pred1_tr)
acc_te1 = accuracy_score(y_true1,    y_pred1)
print(f"=== Skema 1 – LSTM (80/20) ===")
print(f"Train Accuracy : {acc_tr1:.4f}")
print(f"Test  Accuracy : {acc_te1:.4f}")
print(classification_report(y_true1, y_pred1, target_names=le.classes_))

fig, axes = plt.subplots(1,2,figsize=(12,4))
axes[0].plot(history1.history['accuracy'],     label='Train')
axes[0].plot(history1.history['val_accuracy'], label='Val')
axes[0].set_title('LSTM – Accuracy'); axes[0].legend()
axes[1].plot(history1.history['loss'],     label='Train')
axes[1].plot(history1.history['val_loss'], label='Val')
axes[1].set_title('LSTM – Loss'); axes[1].legend()
plt.tight_layout(); plt.savefig('history_lstm.png',dpi=120); plt.show()

cm1 = confusion_matrix(y_true1, y_pred1)
disp1 = ConfusionMatrixDisplay(cm1, display_labels=le.classes_)
fig,ax = plt.subplots(figsize=(5,4))
disp1.plot(ax=ax, colorbar=False, cmap='Greens')
ax.set_title('Confusion Matrix – LSTM')
plt.tight_layout(); plt.savefig('cm_lstm.png',dpi=120); plt.show()
"""))

# ── Skema 2: Bi-LSTM + Word2Vec ───────────────────────────────────────────────
cells.append(md("## 9. Skema 2 – Bi-LSTM + Word2Vec (Split 80/20)"))
cells.append(code("""\
token_corpus = [t.split() for t in df['clean_text']]
w2v = Word2Vec(token_corpus, vector_size=EMBED_DIM, window=5,
               min_count=2, workers=4, epochs=15, seed=42)
w2v.save('word2vec.model')
print("Word2Vec vocab:", len(w2v.wv))

vocab_size   = min(MAX_VOCAB, len(tokenizer_dl.word_index)+1)
embed_matrix = np.zeros((vocab_size, EMBED_DIM))
hit = 0
for word, idx in tokenizer_dl.word_index.items():
    if idx < vocab_size and word in w2v.wv:
        embed_matrix[idx] = w2v.wv[word]
        hit += 1
print(f"Embedding hits: {hit}/{vocab_size}")
"""))

cells.append(code("""\
X_tr2, X_te2, y_tr2, y_te2 = train_test_split(
    X_seq, y_cat, test_size=0.20, random_state=42, stratify=y)

vocab_size = embed_matrix.shape[0]
model_bilstm = Sequential([
    Embedding(vocab_size, EMBED_DIM, weights=[embed_matrix],
              input_length=MAX_LEN, trainable=True),
    SpatialDropout1D(0.25),
    Bidirectional(LSTM(128, dropout=0.3, recurrent_dropout=0.2, return_sequences=True)),
    Bidirectional(LSTM(64,  dropout=0.2, recurrent_dropout=0.1)),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(3,  activation='softmax'),
], name='BiLSTM_Model')

model_bilstm.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
model_bilstm.summary()

cbs2 = [
    EarlyStopping(patience=5, restore_best_weights=True, monitor='val_accuracy'),
    ReduceLROnPlateau(patience=3, factor=0.5, monitor='val_loss', verbose=0),
]

history2 = model_bilstm.fit(
    X_tr2, y_tr2, epochs=25, batch_size=64,
    validation_data=(X_te2, y_te2),
    class_weight=class_weights,
    callbacks=cbs2, verbose=1)

model_bilstm.save('model_bilstm.keras')
"""))

cells.append(code("""\
y_pred2    = np.argmax(model_bilstm.predict(X_te2, verbose=0), axis=1)
y_true2    = np.argmax(y_te2, axis=1)
y_pred2_tr = np.argmax(model_bilstm.predict(X_tr2, verbose=0), axis=1)
y_true2_tr = np.argmax(y_tr2, axis=1)

acc_tr2 = accuracy_score(y_true2_tr, y_pred2_tr)
acc_te2 = accuracy_score(y_true2,    y_pred2)
print(f"=== Skema 2 – Bi-LSTM + Word2Vec (80/20) ===")
print(f"Train Accuracy : {acc_tr2:.4f}")
print(f"Test  Accuracy : {acc_te2:.4f}")
print(classification_report(y_true2, y_pred2, target_names=le.classes_))

fig, axes = plt.subplots(1,2,figsize=(12,4))
axes[0].plot(history2.history['accuracy'],     label='Train')
axes[0].plot(history2.history['val_accuracy'], label='Val')
axes[0].set_title('Bi-LSTM – Accuracy'); axes[0].legend()
axes[1].plot(history2.history['loss'],     label='Train')
axes[1].plot(history2.history['val_loss'], label='Val')
axes[1].set_title('Bi-LSTM – Loss'); axes[1].legend()
plt.tight_layout(); plt.savefig('history_bilstm.png',dpi=120); plt.show()

cm2 = confusion_matrix(y_true2, y_pred2)
disp2 = ConfusionMatrixDisplay(cm2, display_labels=le.classes_)
fig,ax = plt.subplots(figsize=(5,4))
disp2.plot(ax=ax, colorbar=False, cmap='Oranges')
ax.set_title('Confusion Matrix – Bi-LSTM')
plt.tight_layout(); plt.savefig('cm_bilstm.png',dpi=120); plt.show()
"""))

# ── Ringkasan ─────────────────────────────────────────────────────────────────
cells.append(md("## 10. Ringkasan Hasil Semua Skema"))
cells.append(code("""\
summary = pd.DataFrame({
    'Skema'         : ['Skema 1 – LSTM (80/20)',
                       'Skema 2 – Bi-LSTM + Word2Vec (80/20)',
                       'Skema 3 – SVM + TF-IDF (70/30)'],
    'Algoritma'     : ['LSTM','Bi-LSTM','SVM'],
    'Fitur'         : ['Embedding','Word2Vec','TF-IDF'],
    'Split'         : ['80/20','80/20','70/30'],
    'Train Acc'     : [f"{acc_tr1:.4f}", f"{acc_tr2:.4f}", f"{acc_tr3:.4f}"],
    'Test  Acc'     : [f"{acc_te1:.4f}", f"{acc_te2:.4f}", f"{acc_te3:.4f}"],
})
print(summary.to_string(index=False))

# Bar chart perbandingan
labels_skema = ['Skema 1\\nLSTM', 'Skema 2\\nBi-LSTM', 'Skema 3\\nSVM']
train_accs = [acc_tr1, acc_tr2, acc_tr3]
test_accs  = [acc_te1, acc_te2, acc_te3]

x = np.arange(len(labels_skema))
w = 0.35
fig, ax = plt.subplots(figsize=(9,5))
bars1 = ax.bar(x - w/2, train_accs, w, label='Train', color='#3498db')
bars2 = ax.bar(x + w/2, test_accs,  w, label='Test',  color='#e74c3c')
ax.axhline(0.85, color='gray', linestyle='--', linewidth=1, label='Target 85%')
ax.axhline(0.92, color='green', linestyle='--', linewidth=1, label='Target 92%')
ax.set_xticks(x); ax.set_xticklabels(labels_skema)
ax.set_ylim(0, 1.05); ax.set_ylabel('Accuracy')
ax.set_title('Perbandingan Akurasi Semua Skema')
ax.legend()
for bar in bars1: ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                           f'{bar.get_height():.3f}', ha='center', fontsize=9)
for bar in bars2: ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                           f'{bar.get_height():.3f}', ha='center', fontsize=9)
plt.tight_layout(); plt.savefig('perbandingan_akurasi.png',dpi=120); plt.show()
"""))

# ── Inference ─────────────────────────────────────────────────────────────────
cells.append(md("## 11. Inference – Prediksi Sentimen Teks Baru"))
cells.append(code("""\
def predict_sentiment(text, model_name='bilstm'):
    \"\"\"
    Prediksi sentimen dari teks input.
    
    Parameters:
        text       : str  – teks ulasan
        model_name : str  – 'bilstm' | 'lstm' | 'svm'
    
    Returns:
        str – label sentimen ('Positif', 'Netral', atau 'Negatif')
    \"\"\"
    clean = preprocess(text)

    if model_name == 'svm':
        vec  = tfidf.transform([clean])
        pred = svm.predict(vec)[0]
    else:
        seq = tokenizer_dl.texts_to_sequences([clean])
        pad = pad_sequences(seq, maxlen=MAX_LEN, padding='post')
        m   = model_lstm if model_name == 'lstm' else model_bilstm
        prob = m.predict(pad, verbose=0)[0]
        pred = int(np.argmax(prob))

    return le.inverse_transform([pred])[0]

# ── Contoh Inferensi ──────────────────────────────────────────────────────────
test_cases = [
    ("Aplikasi ini sangat bagus dan mudah digunakan, saya sangat puas!",       "Positif"),
    ("Lumayan sih, tapi masih ada beberapa fitur yang kurang",                  "Netral"),
    ("Aplikasi sering crash dan sangat lambat, sangat mengecewakan!",           "Negatif"),
    ("Pengiriman cepat, barang sesuai deskripsi, sangat recommended!",          "Positif"),
    ("Biasa saja, tidak ada yang spesial dari aplikasi ini",                    "Netral"),
    ("Update terbaru memperburuk performa, banyak bug bermunculan",             "Negatif"),
    ("Driver ramah dan tepat waktu, pengalaman yang menyenangkan",              "Positif"),
    ("Kadang error kadang tidak, kurang konsisten performanya",                 "Netral"),
]

print("=" * 72)
print(f"{'Teks Ulasan':<42} | {'Label Asli':^10} | {'Bi-LSTM':^8} | {'LSTM':^8} | {'SVM':^8}")
print("=" * 72)
for teks, label_asli in test_cases:
    p1 = predict_sentiment(teks, 'bilstm')
    p2 = predict_sentiment(teks, 'lstm')
    p3 = predict_sentiment(teks, 'svm')
    s  = teks[:39] + '...' if len(teks) > 39 else teks
    print(f"{s:<42} | {label_asli:^10} | {p1:^8} | {p2:^8} | {p3:^8}")
print("=" * 72)
"""))

cells.append(code("""\
# ── Inferensi Interaktif (ubah teks sesuai kebutuhan) ──────────────────────
teks_uji = "Aplikasinya sangat membantu sehari-hari, fitur lengkap dan cepat"

print(f"Input   : {teks_uji}")
print(f"Bi-LSTM : {predict_sentiment(teks_uji, 'bilstm')}")
print(f"LSTM    : {predict_sentiment(teks_uji, 'lstm')}")
print(f"SVM     : {predict_sentiment(teks_uji, 'svm')}")
"""))

nb.cells = cells

with open('analisis_sentimen.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Notebook 'analisis_sentimen.ipynb' berhasil dibuat!")
