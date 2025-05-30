# 📊 Tweet Sentiment Classification using Deep Learning (FNN, RNN, LTSM, GRU & CNN)

- Documentation by Hazwan Adhikara Nasution

Project ini bertujuan untuk mengklasifikasikan sentimen dari tweet berbahasa Inggris ke dalam tiga kelas: **Negative**, **Neutral**, dan **Positive**. Model deep learning dibangun dan dievaluasi untuk memprediksi sentimen secara akurat menggunakan data teks mentah dari tweet.

## 🚀 Tujuan Proyek

- Membangun model klasifikasi teks untuk prediksi sentimen tweet.
- Menggunakan pendekatan deep learning: **FNN**, **RNN**, dan **CNN**.
- Mengevaluasi performa model menggunakan metrik klasifikasi seperti **Confusion Matrix**, **Accuracy**, dan **F1 Score**.
- Menghasilkan file prediksi (`submission.csv`) untuk kebutuhan deployment/evaluasi lebih lanjut.

---

## 📦 Dataset

Dataset terdiri dari:

- `train.csv`: berisi teks tweet dan label sentimen (`sentiment`: Negative, Neutral, Positive).
- `test.csv`: berisi teks tweet tanpa label, digunakan untuk membuat submission akhir.

---

## 🧼 Preprocessing

Data teks mentah diproses menggunakan pipeline berikut:

- Lowercasing
- Menghapus tanda baca dan angka
- Tokenisasi dan padding
- Label encoding (`LabelEncoder`)
- Tokenization: `Tokenizer(num_words=10000)`
- Padding: `pad_sequences(maxlen=50)`

## 🧠 Model Deep Learning

Tiga arsitektur model dibangun dan dibandingkan:

### ✅ 1. Feedforward Neural Network (FNN)

`Embedding → GlobalMaxPooling1D → Dense → Dropout → Output`

- Cocok untuk baseline sederhana klasifikasi teks.

### 🔁 2. Recurrent Neural Network (RNN)

`Embedding → SimpleRNN → Dropout → Output`

- Menangkap urutan kata, lebih kontekstual.

### 📷 3. Convolutional Neural Network (CNN)

`Embedding → Conv1D → GlobalMaxPooling1D → Dropout → Output`

- Menangkap _pattern_ lokal, sangat efektif untuk teks pendek seperti tweet.

---

## 🔧 Training Setup

- **Optimizer**: `Adam(learning_rate=0.001)`
- **Loss Function**: `sparse_categorical_crossentropy`
- **Callbacks**:
  - `EarlyStopping(patience=4, restore_best_weights=True)`
  - `ReduceLROnPlateau(factor=0.5, patience=3)`

---

## 📈 Evaluasi

Model dievaluasi menggunakan beberapa metrik klasifikasi:

- 📉 **Confusion Matrix**
- 📋 **Classification Report** (Precision, Recall, F1 Score)
- 🎯 **Accuracy**

---

## 🏅 F1 Score per Model

| Model | Accuracy | F1 Score   |
| ----- | -------- | ---------- |
| FNN   | ~71%     | 0.7054     |
| RNN   | ~13%     | 0.2330     |
| LTSM  | ~70%     | 0.7018     |
| GRU   | ~13%     | 0.2330     |
| CNN   | **~72%** | **0.7092** |

> 📌 CNN menunjukkan performa terbaik berdasarkan hasil evaluasi, dengan **F1 score tertinggi** dan distribusi prediksi yang lebih stabil di antara tiga kelas sentimen.

---

## 📚 Tech Stack

- 🐍 Python 3
- 🧠 TensorFlow / Keras
- 🧪 Scikit-learn
- 🧮 Pandas, NumPy, Matplotlib
- 📓 Jupyter Notebook
