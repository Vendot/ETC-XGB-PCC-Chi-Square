# 🫀 Heart Disease Analysis - Technical Documentation

## Analisis Penyakit Jantung Menggunakan Hybrid Model ETC-XGBoost

---

## 📋 Daftar Isi

1. [Gambaran Umum Proyek](#gambaran-umum-proyek)
2. [Arsitektur & Flow End-to-End](#arsitektur--flow-end-to-end)
3. [Detail Teknis Per Tahapan](#detail-teknis-per-tahapan)
4. [Konfigurasi & Setup](#konfigurasi--setup)
5. [Cara Menjalankan](#cara-menjalankan)
6. [Troubleshooting](#troubleshooting)

---

## 🎯 Gambaran Umum Proyek

### Tujuan

Mengembangkan model machine learning hybrid untuk memprediksi penyakit jantung dengan akurasi tinggi menggunakan kombinasi **Extra Trees Classifier (ETC)** dan **XGBoost**, didukung oleh teknik seleksi fitur yang optimal.

### Dataset

- **Nama**: Heart Disease Dataset
- **Jumlah Fitur**: 13 fitur input + 1 target
- **Target**: `target` (0 = tidak ada penyakit, 1 = ada penyakit)
- **Ukuran**: ~1000+ observasi

### Fitur Utama

- **Preprocessing**: Deteksi outlier IQR, KNN Imputation
- **Feature Selection**: Pearson Correlation (kontinu) + Chi-Square (diskrit)
- **Model**: Hybrid ETC-XGBoost dengan ensemble learning
- **Evaluasi**: Cross-validation 10-fold, Friedman test, Nemenyi post-hoc

### Teknologi

```
Python 3.13+
├── Data Processing: pandas, numpy
├── Visualization: matplotlib, seaborn
├── ML Framework: scikit-learn, xgboost
└── Statistics: scipy, scikit-posthocs
```

---

## 🏗️ Arsitektur & Flow End-to-End

```
┌─────────────────────────────────────────────────────────────────┐
│                    PIPELINE MACHINE LEARNING                    │
└─────────────────────────────────────────────────────────────────┘

1️⃣  DATA LOADING & INITIAL EXPLORATION
    ├── Load CSV dataset
    ├── Check struktur data (shape, types, memory)
    ├── Identifikasi missing values
    └── Exploratory Data Analysis (EDA)
            ↓
2️⃣  DATA CLEANING
    ├── Deteksi & hapus duplikat
    ├── Validasi konsistensi data
    └── Verifikasi integritas
            ↓
3️⃣  OUTLIER DETECTION & HANDLING
    ├── Hitung IQR untuk fitur kontinu
    ├── Identifikasi outlier (< Q1-1.5*IQR atau > Q3+1.5*IQR)
    ├── Convert outlier → NaN
    └── Visualisasi boxplot sebelum/sesudah
            ↓
4️⃣  MISSING VALUE IMPUTATION
    ├── KNN Imputation (k=5)
    ├── Impute NaN dengan nilai tetangga terdekat
    └── Validasi tidak ada missing values
            ↓
5️⃣  FEATURE SELECTION
    ├── Fitur Kontinu → Pearson Correlation Coefficient (PCC)
    │   └── Ambil fitur dengan |correlation| > threshold (0.2)
    ├── Fitur Diskrit → Chi-Square Test
    │   └── Ambil top-k fitur dengan skor tertinggi (k=4)
    └── Gabungkan hasil seleksi
            ↓
6️⃣  MODEL TRAINING & GRID SEARCH
    ├── Define 4 Skenario Feature Selection:
    │   ├── Skenario 1: Semua fitur (baseline)
    │   ├── Skenario 2: PCC only
    │   ├── Skenario 3: Chi-Square only
    │   └── Skenario 4: PCC + Chi-Square (hybrid)
    ├── Grid Search Parameter:
    │   ├── n_estimators ETC: [50, 100, 150]
    │   ├── n_estimators XGB: [50, 100, 150]
    │   └── Total: 9 kombinasi × 4 skenario = 36 eksperimen
    └── Train dengan 10-Fold Cross-Validation
            ↓
7️⃣  MODEL EVALUATION
    ├── Hitung metrik per fold:
    │   ├── Accuracy
    │   ├── Precision
    │   ├── Recall
    │   ├── F1-Score
    │   └── Specificity
    ├── Agregat: Mean & Std per metrik
    └── Simpan hasil ke DataFrame
            ↓
8️⃣  STATISTICAL TESTING
    ├── Friedman Test (non-parametric ANOVA)
    │   └── H₀: Tidak ada perbedaan performa antar skenario
    ├── Nemenyi Post-Hoc Test
    │   └── Pairwise comparison antar skenario
    └── Critical Distance (CD) diagram
            ↓
9️⃣  VISUALIZATION & REPORTING
    ├── Barplot: Akurasi per skenario
    ├── Heatmap: Metrik lengkap (Acc, Prec, Rec, F1, Spec)
    ├── CD Diagram: Ranking statistikal
    └── Summary: Konfigurasi terbaik
```

---

## 🔬 Detail Teknis Per Tahapan

### **Tahap 1: Data Loading & EDA**

**📂 File**: Cell 5-9, 12-20

#### Kode Utama:

```python
# Konstanta
PATH_DATASET = '/content/drive/MyDrive/dataset.csv'

# Load dataset
df_original = pd.read_csv(PATH_DATASET)

# Informasi dasar
jumlah_baris, jumlah_kolom = df_original.shape
print(f"Dataset: {jumlah_baris} baris × {jumlah_kolom} kolom")

# Statistik deskriptif
statistik_deskriptif = df_original.describe()

# Distribusi target
distribusi_target = df_original['target'].value_counts()
```

#### Teknik:

- **EDA Sistematis**: Info, describe, value_counts untuk pemahaman data
- **Visualisasi**: Histogram, boxplot, correlation heatmap
- **Variable Naming**: `jumlah_baris`, `jumlah_kolom` (deskriptif dalam Bahasa Indonesia)

---

### **Tahap 2: Data Cleaning**

**📂 File**: Cell 24-25

#### Kode Utama:

```python
# Deteksi duplikat
jumlah_baris_awal = len(df)
jumlah_duplikat = df.duplicated().sum()

if jumlah_duplikat > 0:
    df = df.drop_duplicates()
    jumlah_dihapus = jumlah_baris_awal - len(df)
    print(f"✅ {jumlah_dihapus} baris duplikat dihapus")
else:
    print("✅ Tidak ada duplikat")
```

#### Teknik:

- **Duplicate Detection**: `duplicated()` untuk identifikasi
- **Validation**: Raise ValueError jika duplikat > 5%
- **Logging**: Print informasi perubahan

---

### **Tahap 3: Outlier Handling**

**📂 File**: Cell 29, 31

#### Kode Utama:

```python
# Konstanta fitur dengan outlier
FITUR_OUTLIER = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

def hitung_outlier_iqr(df, kolom):
    """
    Menghitung batas outlier menggunakan metode IQR.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset input
    kolom : str
        Nama kolom yang akan dianalisis

    Returns
    -------
    tuple
        (batas_bawah, batas_atas, jumlah_outlier)
    """
    Q1 = df[kolom].quantile(0.25)
    Q3 = df[kolom].quantile(0.75)
    IQR = Q3 - Q1

    batas_bawah = Q1 - 1.5 * IQR
    batas_atas = Q3 + 1.5 * IQR

    outlier_mask = (df[kolom] < batas_bawah) | (df[kolom] > batas_atas)
    jumlah_outlier = outlier_mask.sum()

    return batas_bawah, batas_atas, jumlah_outlier

# Deteksi outlier untuk semua fitur
df_ringkasan_outlier = pd.DataFrame()
for fitur in FITUR_OUTLIER:
    bawah, atas, jumlah = hitung_outlier_iqr(df, fitur)
    df_ringkasan_outlier.loc[fitur, 'Batas_Bawah'] = bawah
    df_ringkasan_outlier.loc[fitur, 'Batas_Atas'] = atas
    df_ringkasan_outlier.loc[fitur, 'Jumlah_Outlier'] = jumlah

# Convert outlier menjadi NaN
for fitur in FITUR_OUTLIER:
    bawah, atas, _ = hitung_outlier_iqr(df, fitur)
    df.loc[(df[fitur] < bawah) | (df[fitur] > atas), fitur] = np.nan
```

#### Teknik:

- **IQR Method**: Interquartile Range untuk deteksi outlier
- **Formula**: `Q1 - 1.5×IQR` dan `Q3 + 1.5×IQR`
- **Handling**: Convert outlier → NaN (bukan hapus baris)
- **Modular Function**: `hitung_outlier_iqr()` untuk reusability

---

### **Tahap 4: KNN Imputation**

**📂 File**: Cell 35, 37

#### Kode Utama:

```python
from sklearn.impute import KNNImputer
import time

# Konstanta
KNN_NEIGHBORS = 5

# Imputer dengan k=5 tetangga terdekat
imputer_knn = KNNImputer(n_neighbors=KNN_NEIGHBORS)

# Timing imputation
waktu_mulai = time.time()
X_imputed = imputer_knn.fit_transform(X)
waktu_selesai = time.time()

durasi = waktu_selesai - waktu_mulai
print(f"⏱️ KNN Imputation selesai dalam {durasi:.2f} detik")

# Convert kembali ke DataFrame
df_imputed = pd.DataFrame(X_imputed, columns=X.columns)
```

#### Teknik:

- **KNN Imputation**: Isi NaN dengan rata-rata k tetangga terdekat
- **K=5**: Optimal untuk balance antara bias-variance
- **Distance Metric**: Euclidean distance (default)
- **Performance**: Timing untuk monitoring

---

### **Tahap 5: Feature Selection**

**📂 File**: Cell 40-47

#### Kode Utama:

```python
# Konstanta daftar fitur
FITUR_KONTINYU = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
FITUR_DISKRIT = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

def pcc_selection(X, y, threshold=0.2):
    """
    Seleksi fitur kontinyu menggunakan Pearson Correlation Coefficient.

    Parameters
    ----------
    X : pd.DataFrame
        Fitur input
    y : pd.Series
        Target variable
    threshold : float, default=0.2
        Ambang batas absolut korelasi

    Returns
    -------
    list
        Daftar nama fitur yang lolos seleksi

    Example
    -------
    >>> fitur_terpilih = pcc_selection(X[FITUR_KONTINYU], y, threshold=0.2)
    >>> print(fitur_terpilih)
    ['age', 'thalach', 'oldpeak']
    """
    korelasi = X.corrwith(y).abs()
    fitur_terpilih = korelasi[korelasi > threshold].index.tolist()
    return fitur_terpilih

def chi2_selection(X, y, k=4):
    """
    Seleksi fitur diskrit menggunakan Chi-Square test.

    Parameters
    ----------
    X : pd.DataFrame
        Fitur input
    y : pd.Series
        Target variable
    k : int, default=4
        Jumlah fitur terbaik yang diambil

    Returns
    -------
    list
        Daftar nama top-k fitur dengan skor tertinggi

    Example
    -------
    >>> fitur_terpilih = chi2_selection(X[FITUR_DISKRIT], y, k=4)
    >>> print(fitur_terpilih)
    ['cp', 'ca', 'thal', 'exang']
    """
    from sklearn.feature_selection import SelectKBest, chi2

    selector = SelectKBest(chi2, k=k)
    selector.fit(X, y)

    fitur_terpilih = X.columns[selector.get_support()].tolist()
    return fitur_terpilih

# Eksekusi seleksi
fitur_pcc = pcc_selection(X[FITUR_KONTINYU], y, threshold=0.2)
fitur_chi2 = chi2_selection(X[FITUR_DISKRIT], y, k=4)
fitur_hybrid = fitur_pcc + fitur_chi2
```

#### Teknik:

- **PCC (Pearson)**: Untuk fitur kontinu, ukur linear correlation dengan target
  - Formula: $r = \frac{\sum(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum(x_i - \bar{x})^2 \sum(y_i - \bar{y})^2}}$
  - Threshold: |r| > 0.2
- **Chi-Square**: Untuk fitur diskrit/kategorikal, ukur independensi dengan target
  - Formula: $\chi^2 = \sum \frac{(O_i - E_i)^2}{E_i}$
  - Ambil top-k=4 fitur dengan skor tertinggi

- **Hybrid**: Gabungkan hasil PCC + Chi-Square untuk coverage maksimal

---

### **Tahap 6: Model & Grid Search**

**📂 File**: Cell 51, 53-58

#### Kode Utama (Model Class):

```python
class ETCXGBHybrid:
    """
    Model hybrid yang menggabungkan Extra Trees Classifier (ETC)
    dan XGBoost dengan voting ensemble.

    Parameters
    ----------
    n_estimators_etc : int, default=100
        Jumlah trees untuk ETC
    n_estimators_xgb : int, default=100
        Jumlah trees untuk XGBoost
    random_state : int, default=42
        Seed untuk reproducibility

    Attributes
    ----------
    model_ : VotingClassifier
        Ensemble model hasil training

    Example
    -------
    >>> model = ETCXGBHybrid(n_estimators_etc=100, n_estimators_xgb=100)
    >>> model.fit(X_train, y_train)
    >>> y_pred = model.predict(X_test)
    >>> accuracy = model.score(X_test, y_test)
    """

    def __init__(self, n_estimators_etc=100, n_estimators_xgb=100, random_state=42):
        self.n_estimators_etc = n_estimators_etc
        self.n_estimators_xgb = n_estimators_xgb
        self.random_state = random_state
        self.model_ = None

    def fit(self, X, y):
        """Training model dengan voting ensemble."""
        etc = ExtraTreesClassifier(
            n_estimators=self.n_estimators_etc,
            random_state=self.random_state
        )
        xgb = XGBClassifier(
            n_estimators=self.n_estimators_xgb,
            random_state=self.random_state,
            eval_metric='logloss'
        )

        self.model_ = VotingClassifier(
            estimators=[('etc', etc), ('xgb', xgb)],
            voting='soft'
        )
        self.model_.fit(X, y)
        return self
```

#### Kode Utama (Grid Search):

```python
# Parameter grid
ESTIMATORS_LIST = [50, 100, 150]

# Helper function
def tampilkan_progress(current, total, skenario, n_etc, n_xgb):
    """Tampilkan progress bar grid search."""
    persen = (current / total) * 100
    print(f"[{persen:5.1f}%] Skenario {skenario} | ETC={n_etc}, XGB={n_xgb}")

# Grid search
total_kombinasi = len(ESTIMATORS_LIST) ** 2 * 4
counter = 0

hasil_list = []
for idx_skenario, skenario_fitur in enumerate(skenario_list, 1):
    for n_etc in ESTIMATORS_LIST:
        for n_xgb in ESTIMATORS_LIST:
            counter += 1
            tampilkan_progress(counter, total_kombinasi, idx_skenario, n_etc, n_xgb)

            # Training & evaluation
            model = ETCXGBHybrid(n_estimators_etc=n_etc, n_estimators_xgb=n_xgb)
            metrik = evaluasi_model_cv(model, X[skenario_fitur], y, cv=10)

            hasil_list.append({
                'Skenario': idx_skenario,
                'n_estimators_ETC': n_etc,
                'n_estimators_XGB': n_xgb,
                **metrik
            })

df_hasil = pd.DataFrame(hasil_list)
```

#### Teknik:

- **Ensemble Learning**: Voting soft dari ETC + XGBoost
- **Grid Search**: Exhaustive search 3×3 parameter = 9 kombinasi
- **4 Skenario**: Baseline, PCC-only, Chi2-only, Hybrid
- **Progress Tracking**: Real-time monitoring dengan `tampilkan_progress()`

---

### **Tahap 7: Model Evaluation**

**📂 File**: Cell 53-58, 63-65

#### Kode Utama:

```python
def hitung_specificity(y_true, y_pred):
    """
    Hitung Specificity (True Negative Rate).

    Specificity = TN / (TN + FP)
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)

def evaluasi_model_cv(model, X, y, cv=10):
    """
    Evaluasi model dengan k-fold cross-validation.

    Parameters
    ----------
    model : estimator
        Model yang akan dievaluasi
    X : array-like
        Fitur input
    y : array-like
        Target variable
    cv : int, default=10
        Jumlah fold untuk cross-validation

    Returns
    -------
    dict
        Dictionary berisi mean ± std untuk setiap metrik:
        - Accuracy
        - Precision
        - Recall
        - F1-Score
        - Specificity

    Example
    -------
    >>> model = ETCXGBHybrid()
    >>> metrik = evaluasi_model_cv(model, X_train, y_train, cv=10)
    >>> print(f"Accuracy: {metrik['Accuracy_Mean']:.4f} ± {metrik['Accuracy_Std']:.4f}")
    """
    from sklearn.model_selection import cross_val_predict

    # Prediksi dengan CV
    y_pred = cross_val_predict(model, X, y, cv=cv)

    # Hitung metrik per fold (manual loop untuk custom metrics)
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    metrik_per_fold = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'specificity': []
    }

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        y_pred_fold = model.predict(X_test)

        metrik_per_fold['accuracy'].append(accuracy_score(y_test, y_pred_fold))
        metrik_per_fold['precision'].append(precision_score(y_test, y_pred_fold))
        metrik_per_fold['recall'].append(recall_score(y_test, y_pred_fold))
        metrik_per_fold['f1'].append(f1_score(y_test, y_pred_fold))
        metrik_per_fold['specificity'].append(hitung_specificity(y_test, y_pred_fold))

    # Agregat mean ± std
    hasil = {}
    for nama_metrik, nilai_list in metrik_per_fold.items():
        hasil[f'{nama_metrik.capitalize()}_Mean'] = np.mean(nilai_list)
        hasil[f'{nama_metrik.capitalize()}_Std'] = np.std(nilai_list)

    return hasil
```

#### Metrik:

- **Accuracy**: $\frac{TP + TN}{TP + TN + FP + FN}$
- **Precision**: $\frac{TP}{TP + FP}$ (ketepatan prediksi positif)
- **Recall**: $\frac{TP}{TP + FN}$ (coverage kasus positif)
- **F1-Score**: $2 \times \frac{Precision \times Recall}{Precision + Recall}$ (harmonic mean)
- **Specificity**: $\frac{TN}{TN + FP}$ (True Negative Rate)

#### Teknik:

- **10-Fold CV**: Data dibagi 10 bagian, iterasi 10x (9 train, 1 test)
- **Stratified Split**: Jaga proporsi kelas balanced
- **Custom Metrics**: Implementasi manual untuk Specificity

---

### **Tahap 8: Statistical Testing**

**📂 File**: Cell 66-75

#### Kode Utama:

```python
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp

# Persiapan data untuk Friedman test
# Shape: (n_kombinasi × n_skenario) = (9 kombinasi × 4 skenario)
akurasi_per_skenario = []
for idx in range(1, 5):
    df_skenario = df_hasil[df_hasil['Skenario'] == idx]
    akurasi_per_skenario.append(df_skenario['Accuracy_Mean'].values)

# Friedman test
stat, p_value = friedmanchisquare(*akurasi_per_skenario)
print(f"Friedman χ² = {stat:.4f}, p-value = {p_value:.4f}")

if p_value < 0.05:
    print("✅ Ada perbedaan signifikan antar skenario")

    # Nemenyi post-hoc test
    data_nemenyi = df_hasil.pivot_table(
        values='Accuracy_Mean',
        index=['n_estimators_ETC', 'n_estimators_XGB'],
        columns='Skenario'
    )

    hasil_nemenyi = sp.posthoc_nemenyi_friedman(data_nemenyi)
    print("\nNemenyi Post-Hoc (p-values):")
    print(hasil_nemenyi)
else:
    print("❌ Tidak ada perbedaan signifikan")
```

#### Teknik:

- **Friedman Test**: Non-parametric ANOVA untuk repeated measures
  - H₀: Tidak ada perbedaan median antar grup
  - Alternatif parametrik: Repeated ANOVA (memerlukan asumsi normalitas)
- **Nemenyi Post-Hoc**: Pairwise comparison setelah Friedman signifikan
  - Mirip Tukey HSD untuk non-parametric
  - Menentukan pasangan skenario mana yang berbeda signifikan
- **Critical Distance (CD)**: Visualisasi ranking dengan interval kepercayaan

---

### **Tahap 9: Visualization & Best Model**

**📂 File**: Cell 70-75

#### Kode Utama:

```python
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Barplot: Akurasi rata-rata per skenario
akurasi_rata_per_skenario = df_hasil.groupby('Skenario')['Accuracy_Mean'].mean()

plt.figure(figsize=(10, 6))
warna_map = {1: 'lightblue', 2: 'lightgreen', 3: 'lightcoral', 4: 'gold'}
colors = [warna_map[s] for s in akurasi_rata_per_skenario.index]

plt.bar(akurasi_rata_per_skenario.index, akurasi_rata_per_skenario.values, color=colors)
plt.xlabel('Skenario', fontsize=12)
plt.ylabel('Akurasi Rata-Rata', fontsize=12)
plt.title('Perbandingan Akurasi per Skenario Feature Selection', fontsize=14, fontweight='bold')
plt.xticks([1, 2, 3, 4], ['Baseline', 'PCC', 'Chi2', 'Hybrid'])
plt.ylim(0.80, 0.90)
plt.grid(axis='y', alpha=0.3)
plt.show()

# 2. Heatmap: Metrik lengkap
metrik_agregat = df_hasil.groupby('Skenario')[[
    'Accuracy_Mean', 'Precision_Mean', 'Recall_Mean',
    'F1_Mean', 'Specificity_Mean'
]].mean()

plt.figure(figsize=(10, 6))
sns.heatmap(metrik_agregat.T, annot=True, fmt='.4f', cmap='RdYlGn',
            cbar_kws={'label': 'Nilai Metrik'})
plt.xlabel('Skenario', fontsize=12)
plt.ylabel('Metrik Evaluasi', fontsize=12)
plt.title('Heatmap Metrik per Skenario', fontsize=14, fontweight='bold')
plt.show()

# 3. Best Model
idx_terbaik = df_hasil['Accuracy_Mean'].idxmax()
config_terbaik = df_hasil.loc[idx_terbaik]

print("🏆 KONFIGURASI MODEL TERBAIK:")
print(f"   Skenario: {int(config_terbaik['Skenario'])}")
print(f"   n_estimators_ETC: {int(config_terbaik['n_estimators_ETC'])}")
print(f"   n_estimators_XGB: {int(config_terbaik['n_estimators_XGB'])}")
print(f"   Accuracy: {config_terbaik['Accuracy_Mean']:.4f} ± {config_terbaik['Accuracy_Std']:.4f}")
print(f"   F1-Score: {config_terbaik['F1_Mean']:.4f}")
```

#### Output:

- **Barplot**: Visualisasi cepat perbandingan skenario
- **Heatmap**: Overview semua metrik sekaligus
- **Best Configuration**: Parameter optimal untuk deployment

---

## ⚙️ Konfigurasi & Setup

### Requirements

```txt
pandas==2.3.2
numpy==1.26.4
matplotlib==3.9.4
seaborn==0.13.2
scikit-learn==1.7.2
xgboost==2.1.4
scipy==1.15.2
scikit-posthocs==0.11.0
```

### Instalasi

```bash
pip install -r requirements.txt
```

### Docker Setup (Opsional)

```bash
# Build image
docker build -t heart-disease-analysis .

# Run container
docker-compose up --build

# Access Jupyter
# Browser: http://localhost:8888
```

---

## 🚀 Cara Menjalankan

### Metode 1: Jupyter Notebook (Lokal)

```bash
# 1. Clone/download repository
cd /path/to/project

# 2. Install dependencies
pip install -r requirements.txt

# 3. Jalankan Jupyter
jupyter notebook

# 4. Buka Heart_Disease_Analysis_Refactored.ipynb

# 5. Run All Cells (Cell → Run All)
```

### Metode 2: Google Colab

```python
# 1. Upload notebook ke Google Drive
# 2. Buka dengan Google Colab
# 3. Mount Google Drive:
from google.colab import drive
drive.mount('/content/drive')

# 4. Update PATH_DATASET sesuai lokasi file
PATH_DATASET = '/content/drive/MyDrive/dataset.csv'

# 5. Run All Cells
```

### Metode 3: Docker (Reproducible)

```bash
docker-compose up --build
# Access: http://localhost:8888
```

---

## 🔧 Troubleshooting

### Problem 1: Import Error

**Error**: `ModuleNotFoundError: No module named 'xgboost'`

**Solusi**:

```bash
pip install xgboost==2.1.4
# Atau untuk semua dependencies:
pip install -r requirements.txt
```

---

### Problem 2: Memory Error (Large Dataset)

**Error**: `MemoryError` saat KNN Imputation

**Solusi**:

```python
# Reduce KNN neighbors
KNN_NEIGHBORS = 3  # Default: 5

# Atau batch processing
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=3)
X_imputed = imputer.fit_transform(X)
```

---

### Problem 3: GridSearch Terlalu Lama

**Masalah**: Grid search 36 eksperimen × 10-fold CV = 360 training iterations

**Solusi**:

```python
# Opsi 1: Reduce parameter grid
ESTIMATORS_LIST = [100]  # Hanya 1 nilai, bukan [50, 100, 150]

# Opsi 2: Reduce CV folds
metrik = evaluasi_model_cv(model, X, y, cv=5)  # Default: 10

# Opsi 3: Parallel processing
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=10, n_jobs=-1)  # Use all CPU cores
```

---

### Problem 4: Reproduksi Hasil Berbeda

**Masalah**: Hasil training tidak konsisten antar run

**Solusi**:

```python
# Set random seed di semua tempat
import random
import numpy as np

RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# Di setiap model
model = ETCXGBHybrid(random_state=RANDOM_STATE)
```

---

### Problem 5: Visualisasi Tidak Muncul

**Masalah**: Plot tidak ditampilkan di Jupyter

**Solusi**:

```python
# Tambahkan magic command di cell pertama
%matplotlib inline

# Atau gunakan explicit show
import matplotlib.pyplot as plt
plt.figure()
plt.plot([1, 2, 3])
plt.show()
```

---

## 📊 Interpretasi Hasil

### Membaca Output

#### 1. Accuracy > 0.85

✅ **Interpretasi**: Model sangat baik untuk prediksi penyakit jantung

#### 2. Precision vs Recall Trade-off

- **High Precision, Low Recall**: Model konservatif (prediksi positif hanya jika yakin)
- **Low Precision, High Recall**: Model agresif (tangkap semua kasus positif, banyak false alarm)
- **Target**: Balance keduanya (F1-Score tinggi)

#### 3. Specificity Tinggi

✅ **Interpretasi**: Model bagus mengidentifikasi orang sehat (True Negative)

#### 4. Friedman p-value < 0.05

✅ **Interpretasi**: Ada perbedaan signifikan antar skenario feature selection

---

## 📚 Referensi & Resources

### Paper & Teori

- Breiman, L. (2001). "Random Forests". _Machine Learning_, 45(1), 5-32.
- Chen, T., & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System". _KDD_.
- Friedman, M. (1937). "The Use of Ranks to Avoid the Assumption of Normality". _JASA_.

### Library Documentation

- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [scikit-posthocs Documentation](https://scikit-posthocs.readthedocs.io/)

### Dataset

- UCI Heart Disease Dataset: [https://archive.ics.uci.edu/dataset/45/heart+disease](https://archive.ics.uci.edu/dataset/45/heart+disease)

---

## 👥 Kontributor & Lisensi

**Author**: Teaching Assistant - Machine Learning Course  
**Last Updated**: 2024  
**License**: MIT License

---

## 📞 Kontak & Support

Jika ada pertanyaan atau issue:

1. Baca bagian [Troubleshooting](#troubleshooting)
2. Check dokumentasi inline di notebook (docstrings)
3. Review error message dengan teliti

**Happy Coding! 🚀**
