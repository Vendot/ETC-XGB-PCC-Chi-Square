# PANDUAN REVISI NOTEBOOK - BAHASA INDONESIA FORMAL AKADEMIK

## STATUS REVISI YANG SUDAH DILAKUKAN

### ✅ SELESAI:
1. **Judul utama** - Sudah diubah ke format akademik formal
2. **Outline detail** - Sudah dihapus (terlalu panjang)
3. **Bagian 1.1-1.4** - Import library, load dataset, validasi ca dan thal sudah formal

### 🔄 PERLU REVISI:

## BAGIAN 2: EXPLORATORY DATA ANALYSIS (EDA)

### Cell yang perlu DIREVISI/DITAMBAH:

#### 1. Hapus Cell "Insight Komprehensif" (Cell 19-20)
- Cell dengan 200+ baris analisis per fitur TERLALU PANJANG
- Ganti dengan analisis per fitur yang lebih ringkas

#### 2. Tambah Markdown: "# 2. Exploratory Data Analysis"
```markdown
# 2. Exploratory Data Analysis (EDA)

Tahap EDA bertujuan untuk memahami karakteristik data tanpa melakukan modifikasi. Analisis meliputi:
- Karakteristik umum dataset
- Distribusi target class
- Distribusi setiap fitur
- Identifikasi nilai ekstrem

**Catatan:** EDA tidak mengubah data. Semua transformasi dilakukan pada tahap Data Cleaning.
```

#### 3. Revisi/Tambah: "## 2.1 Karakteristik Umum Dataset"
```markdown
## 2.1 Karakteristik Umum Dataset

Pada tahap ini dilakukan pemeriksaan karakteristik dasar dataset untuk memahami struktur dan kualitas data.
```

**Kode Cell (tambahkan setelah markdown di atas):**
```python
"""
Menampilkan karakteristik umum dataset setelah validasi awal
"""

print("=" * 80)
print("📊 KARAKTERISTIK DATASET UCI HEART DISEASE")
print("=" * 80)

# 1. Dimensi dataset
print("\n1. DIMENSI DATASET:")
print(f"   • Jumlah sampel: {df_original.shape[0]}")
print(f"   • Jumlah fitur: {df_original.shape[1] - 1}  # minus target")
print(f"   • Total kolom: {df_original.shape[1]}")

# 2. Informasi tipe data
print("\n2. INFORMASI KOLOM:")
print(df_original.info())

# 3. Statistik deskriptif
print("\n3. STATISTIK DESKRIPTIF:")
display(df_original.describe())

# 4. Missing values
print("\n4. MISSING VALUES:")
missing = df_original.isnull().sum()
missing_pct = (missing / len(df_original)) * 100
df_missing = pd.DataFrame({
    'Jumlah': missing[missing > 0],
    'Persentase (%)': missing_pct[missing > 0].round(2)
})
if len(df_missing) > 0:
    display(df_missing)
else:
    print("   ✅ Tidak ada missing values")

print("=" * 80)
```

#### 4. Revisi: "## 2.2 Distribusi Target Class"
```markdown
## 2.2 Distribusi Target Class

Distribusi kelas target menunjukkan proporsi pasien yang terdiagnosis penyakit jantung dibandingkan yang tidak. Keseimbangan distribusi class berimplikasi pada strategi modeling yang digunakan.
```

**Kode tetap sama, tapi tambahkan print interpretasi:**
```python
# ... kode visualisasi yang sudah ada ...

# Interpretasi
balance_ratio = min(target_counts) / max(target_counts)
print("\n💡 INTERPRETASI:")
print("=" * 60)
if balance_ratio > 0.8:
    print("✅ Dataset SEIMBANG")
    print("   • Tidak diperlukan teknik resampling")
    print("   • Stratified K-Fold cukup untuk menjaga proporsi class")
    print("   • Metrik accuracy dapat digunakan sebagai evaluasi utama")
else:
    print("⚠️  Dataset TIDAK SEIMBANG")
    print("   • Perlu Stratified K-Fold untuk menjaga proporsi")
    print("   • Prioritas metrik: Recall > F1-Score > Accuracy")
    print("   • Pertimbangkan class weighting dalam model")

print("\n🏥 KONTEKS MEDIS:")
print("   Dalam diagnosa penyakit jantung, false negative (gagal mendeteksi")
print("   penyakit) lebih berbahaya dibanding false positive. Oleh karena itu,")
print("   metrik RECALL diprioritaskan dalam evaluasi model.")
print("=" * 60)
```

#### 5. Tambah: "## 2.3 Analisis Distribusi Fitur"
```markdown
## 2.3 Analisis Distribusi Fitur

Bagian ini menganalisis distribusi setiap fitur dalam dataset untuk memahami karakteristik dan pola data. Analisis dilakukan per fitur dengan menjelaskan makna klinis dan pola distribusinya.

### Fitur Kontinyu

**a. Age (Usia)**

Fitur age merepresentasikan usia pasien dalam tahun. Dari distribusi terlihat bahwa:
- Rentang usia: 29-77 tahun
- Mayoritas pasien berusia 50-60 tahun (middle-aged)
- Distribusi cenderung normal dengan sedikit skewness ke kanan
- Terdapat beberapa nilai ekstrem pada usia sangat muda (<35) dan sangat tua (>70)

Secara klinis, risiko penyakit jantung meningkat seiring bertambahnya usia, sehingga fitur ini memiliki relevansi medis yang tinggi.

**b. Trestbps (Resting Blood Pressure)**

Fitur trestbps merepresentasikan tekanan darah saat istirahat (mmHg). Analisis menunjukkan:
- Rentang: 94-200 mmHg
- Rata-rata: ~131 mmHg (sedikit di atas normal 120 mmHg)
- Terdapat 9 missing values (dari validasi awal)
- Distribusi relatif normal dengan beberapa nilai ekstrem di ujung atas

Nilai tekanan darah tinggi (>140 mmHg) merupakan indikator hipertensi yang menjadi faktor risiko utama penyakit jantung.

**c. Chol (Serum Cholesterol)**

Fitur chol merepresentasikan kolesterol serum dalam mg/dl. Karakteristik:
- Rentang: 126-564 mg/dl
- Rata-rata: ~246 mg/dl (di atas batas normal 200 mg/dl)
- Terdapat 5 missing values
- Distribusi right-skewed dengan beberapa nilai sangat tinggi

Kolesterol tinggi (>200 mg/dl) meningkatkan risiko penyumbatan pembuluh darah dan penyakit jantung koroner.

**d. Thalach (Maximum Heart Rate Achieved)**

Fitur thalach merepresentasikan detak jantung maksimum yang dicapai saat tes stress. Pola distribusi:
- Rentang: 71-202 bpm
- Rata-rata: ~150 bpm
- Distribusi mendekati normal dengan sedikit left-skew
- Terdapat 1 missing value

Secara klinis, detak jantung maksimum yang rendah pada tes stress dapat mengindikasikan kapasitas jantung yang menurun.

**e. Oldpeak (ST Depression)**

Fitur oldpeak merepresentasikan depresi segmen ST yang diinduksi oleh exercise. Karakteristik:
- Rentang: -2.6 hingga 6.2
- Mayoritas nilai mendekati 0-2
- Distribusi sangat right-skewed
- Terdapat 5 missing values

ST depression yang signifikan (>2) merupakan indikator kuat iskemia jantung.

### Fitur Diskrit

**f. Sex (Jenis Kelamin)**

Encoding: 0=Female, 1=Male
- Distribusi: ~68% Male, ~32% Female
- Dataset memiliki bias gender karena penyakit jantung lebih prevalens pada pria
- Jenis kelamin adalah faktor risiko yang sudah diketahui secara medis

**g. CP (Chest Pain Type)**

Encoding: 0=typical angina, 1=atypical angina, 2=non-anginal pain, 3=asymptomatic
- Distribusi relatif merata dengan tipe 0 (typical angina) paling banyak
- Tipe chest pain adalah indikator penting untuk diagnosis awal
- Asymptomatic (tipe 3) menunjukkan kasus yang lebih sulit dideteksi

**h. FBS (Fasting Blood Sugar)**

Encoding: 0=<120 mg/dl, 1=>120 mg/dl
- Mayoritas pasien (~85%) memiliki FBS <120 mg/dl
- FBS >120 mg/dl mengindikasikan diabetes atau prediabetes
- Diabetes merupakan komorbiditas umum pada penyakit jantung

**i. RestECG (Resting Electrocardiographic)**

Encoding: 0=normal, 1=ST-T abnormality, 2=left ventricular hypertrophy
- Mayoritas pasien memiliki hasil ECG normal
- Abnormalitas ECG dapat mengindikasikan kerusakan jantung

**j. Exang (Exercise Induced Angina)**

Encoding: 0=No, 1=Yes
- Distribusi: ~67% No, ~33% Yes
- Angina yang dipicu oleh exercise adalah tanda kuat iskemia jantung

**k. Slope (Slope of Peak Exercise ST Segment)**

Encoding: 0=upsloping, 1=flat, 2=downsloping
- Mayoritas pasien memiliki slope tipe 1 (flat)
- Slope downsloping mengindikasikan iskemia yang lebih parah

**l. CA (Number of Major Vessels Colored by Fluoroscopy)**

Nilai valid: 0-3
- Mayoritas pasien memiliki ca=0 (tidak ada pembuluh tersumbat)
- Semakin tinggi ca, semakin banyak pembuluh yang tersumbat
- Terdapat 4 missing values (dari konversi ca=4)

**m. Thal (Thalassemia)**

Nilai valid: 1=normal, 2=fixed defect, 3=reversible defect
- Distribusi mayoritas pada tipe 2 (fixed defect)
- Thalassemia dapat menyebabkan anemia yang memperburuk kondisi jantung
- Terdapat 2 missing values (dari konversi thal=0)
```

**CATATAN:** Cell dengan kode visualisasi distribusi (histogram) sudah ada, TIDAK PERLU diubah.

#### 6. Revisi: "## 2.4 Identifikasi Nilai Ekstrem"
```markdown
## 2.4 Identifikasi Nilai Ekstrem (Outlier Awareness)

Pada tahap ini dilakukan identifikasi terhadap nilai-nilai ekstrem menggunakan metode IQR (Interquartile Range) sebagai alat bantu. Penting untuk dicatat bahwa **nilai ekstrem tidak otomatis dihapus** karena dalam konteks medis, nilai ekstrem dapat merepresentasikan kondisi pasien yang sebenarnya.

**Metode IQR:**
- Q1 = Kuartil ke-25
- Q3 = Kuartil ke-75
- IQR = Q3 - Q1
- Batas bawah = Q1 - 1.5 × IQR
- Batas atas = Q3 + 1.5 × IQR

Nilai di luar batas ini dikategorikan sebagai nilai ekstrem.

**Pendekatan dalam Penelitian:**
1. Nilai ekstrem diidentifikasi tetapi tidak langsung dihapus
2. Konteks klinis dijadikan pertimbangan utama
3. Treatment nilai ekstrem dilakukan pada tahap Data Cleaning dengan mengonversi ke NaN untuk diimputasi
```

**Cell kode yang sudah ada (identifikasi outlier dengan IQR) tetap digunakan.**

#### 7. HAPUS Cell "## 2.5 Analisis Korelasi" 
Pindahkan ke EDA (atau buat di bagian feature selection saja)

---

## BAGIAN 3: DATA CLEANING

Cell markdown yang sudah ada cukup baik, hanya perlu sedikit penyesuaian bahasa:

### Revisi Cell "# 3. Data Cleaning":
```markdown
# 3. Data Cleaning

Data cleaning merupakan tahap penting untuk memastikan kualitas data sebelum masuk ke tahap modeling. Pada bagian ini dilakukan:
1. Deteksi dan penghapusan data duplikat
2. Penanganan missing value melalui imputasi
```

### Revisi "## 3.1 Deteksi dan Penghapusan Data Duplikat":
Cell sudah bagus, hanya tambahkan interpretasi:
```markdown
## 3.1 Deteksi dan Penghapusan Data Duplikat

Data duplikat dapat menyebabkan bias dalam proses training model karena sampel yang sama diperhitungkan lebih dari satu kali. Deteksi dilakukan dengan membandingkan semua kolom secara bersamaan.
```

### Hapus Bagian "# 4. Penanganan Outlier"

**JANGAN PISAHKAN!** Outlier handling adalah bagian dari data cleaning. Revisi strukturnya:

```markdown
## 3.2 Penanganan Nilai Ekstrem (Outlier Treatment)

Nilai ekstrem yang teridentifikasi pada tahap EDA akan ditangani pada bagian ini. Pendekatan yang digunakan adalah mengonversi nilai ekstrem menjadi NaN (bukan menghapus) untuk kemudian diimputasi. Pendekatan ini dipilih karena:

1. **Menghindari loss of information**: Menghapus data akan mengurangi jumlah sampel
2. **Konteks medis**: Nilai ekstrem bisa jadi merepresentasikan kondisi klinis yang nyata
3. **Imputasi lebih baik**: Nilai akan diestimasi berdasarkan pola data sekitarnya

**Fitur yang ditangani:** trestbps, chol, thalach, oldpeak (fitur kontinyu yang memiliki nilai ekstrem signifikan)

**Metode:** IQR dengan threshold 1.5 (standar Tukey, 1977)
```

### Tambahkan "## 3.3 Imputasi Missing Value"
```markdown
## 3.3 Imputasi Missing Value

Setelah validasi domain dan treatment outlier, terdapat beberapa missing value yang perlu diimputasi. Missing value berasal dari:
1. Validasi domain (ca=4 → NaN, thal=0 → NaN)  
2. Missing value original dari dataset (trestbps, chol, thalach, oldpeak)
3. Konversi nilai ekstrem menjadi NaN

**Metode Imputasi: KNN (K-Nearest Neighbors) dengan k=5**

Alasan pemilihan KNN:
- Mempertahankan pola lokal data (lebih akurat daripada mean/median)
- Cocok untuk dataset multivariat dengan korelasi antar fitur
- Tidak mengasumsikan distribusi tertentu
- Proven effective untuk medical datasets

Setelah imputasi, dilakukan verifikasi untuk memastikan tidak ada missing value tersisa.
```

---

## BAGIAN 4: FEATURE ENGINEERING DAN FEATURE SELECTION

### Revisi struktur menjadi:

```markdown
# 4. Feature Engineering dan Feature Selection

Tahap ini melakukan preparasi fitur dan seleksi fitur untuk meningkatkan performa model. Tahapan meliputi:
1. Pemisahan fitur kontinyu dan diskrit
2. Standardisasi fitur kontinyu
3. Implementasi berbagai strategi feature selection

## 4.1 Pemisahan dan Standardisasi Fitur

Fitur dipisahkan menjadi dua kategori:

**Fitur Kontinyu (5):** age, trestbps, chol, thalach, oldpeak
- Memerlukan standardisasi (StandardScaler)
- Menggunakan Pearson Correlation Coefficient (PCC) untuk seleksi

**Fitur Diskrit (8):** sex, cp, fbs, restecg, exang, slope, ca, thal
- Tidak perlu standardisasi
- Menggunakan Chi-Square test untuk seleksi

Standardisasi dilakukan menggunakan StandardScaler untuk menghindari bias akibat perbedaan skala antar fitur.

## 4.2 Strategi Feature Selection

Penelitian ini mengevaluasi 4 skenario feature selection:

1. **Baseline**: Menggunakan semua fitur (13 fitur) tanpa seleksi
2. **PCC (Pearson Correlation Coefficient)**: Seleksi fitur kontinyu berdasarkan korelasi dengan target
   - Threshold: 0.1, 0.15, 0.2
3. **Chi-Square**: Seleksi fitur diskrit berdasarkan dependensi dengan target
   - K terbaik: 5, 7, 9 fitur
4. **Combined (PCC + Chi2)**: Kombinasi kedua metode

Setiap skenario akan dievaluasi untuk menentukan strategi feature selection yang paling efektif.
```

---

## BAGIAN 5: MODELING DAN EVALUASI

```markdown
# 5. Modeling dan Evaluasi

Bagian ini mengimplementasikan model hybrid ETCXGBHybrid (Extra Trees Classifier + XGBoost) dan mengevaluasi performanya menggunakan 10-Fold Stratified Cross-Validation.

## 5.1 Arsitektur Model ETCXGBHybrid

Model ETCXGBHybrid menggabungkan kekuatan dua algoritma ensemble:
- **Extra Trees Classifier**: Extreme randomization untuk mengurangi overfitting
- **XGBoost**: Gradient boosting untuk akurasi tinggi

## 5.2 Strategi Cross-Validation

Evaluasi menggunakan **10-Fold Stratified Cross-Validation** dengan alasan:
- Stratified: Menjaga proporsi class pada setiap fold
- 10-Fold: Standar dalam machine learning research
- Memberikan estimasi performa yang reliable

## 5.3 Metrik Evaluasi

Model dievaluasi menggunakan 5 metrik:
1. **Accuracy**: Overall correctness
2. **Precision**: Positive predictive value
3. **Recall (Sensitivity)**: True positive rate - **PRIORITAS UTAMA**
4. **Specificity**: True negative rate
5. **F1-Score**: Harmonic mean of precision and recall

**Dalam konteks medis**, Recall diprioritaskan karena false negative (gagal mendeteksi penyakit) lebih berbahaya dibanding false positive.

## 5.4 Hyperparameter Tuning

Kombinasi hyperparameter yang diuji:
- Estimators: {100, 200, 300}
- PCC Threshold: {0.1, 0.15, 0.2}
- Chi2 K: {5, 7, 9}

Total konfigurasi: 3 × 3 × 3 × 4 skenario = 108 konfigurasi
```

---

## BAGIAN 6: ANALISIS STATISTIK

```markdown
# 6. Analisis Statistik

Untuk membandingkan performa antar skenario feature selection secara statistik, digunakan uji Friedman Test.

## 6.1 Uji Friedman Test

**Tujuan**: Menguji apakah terdapat perbedaan signifikan antar skenario feature selection

**Karakteristik**:
- Non-parametric test (tidak mengasumsikan distribusi normal)
- Equivalent dengan repeated measures ANOVA
- Menggunakan ranking dari performa di setiap fold

**Hipotesis**:
- H₀: Tidak ada perbedaan signifikan antar skenario
- H₁: Minimal ada satu skenario yang berbeda signifikan
- α = 0.05

## 6.2 Uji Post-Hoc Nemenyi

Jika Friedman Test menunjukkan perbedaan signifikan (p < 0.05), dilanjutkan dengan uji Nemenyi untuk identifikasi pasangan skenario mana yang berbeda.

**Output**:
- Matriks p-value antar pasangan skenario
- Visualisasi heatmap untuk interpretasi mudah
- Identifikasi pasangan dengan perbedaan signifikan

## 6.3 Interpretasi Hasil

Hasil uji statistik digunakan untuk:
1. Validasi scientific: Apakah feature selection memberikan improvement signifikan?
2. Pemilihan skenario terbaik berdasarkan bukti statistik
3. Rekomendasi untuk implementasi praktis
```

---

## CATATAN IMPLEMENTASI

1. **Jangan menambahkan boxplot fitur numerik kontinyu** - sudah ada di bagian identifikasi outlier
2. **Jangan mengubah urutan logika** - validasi domain → EDA → cleaning → feature selection → modeling
3. **Semua markdown HARUS bahasa Indonesia formal akademik**
4. **Fokus pada INSIGHT bukan teori statistik**
5. **Setiap kode harus punya dokumentasi lengkap**

## LANGKAH REVISI SISTEMATIS

1. ✅ Bagian 1: Sudah OK (Import, Load, Validasi ca/thal)
2. 🔄 Bagian 2: Revisi EDA - PRIORITAS TINGGI
   - Hapus cell insight komprehensif
   - Ganti dengan analisis per fitur di markdown
   - Pastikan tidak ada boxplot tambahan
3. 🔄 Bagian 3: Data Cleaning - penyesuaian kecil
   - Gabungkan outlier handling ke sini
   - Tambahkan imputasi sebagai 3.3
4. 🔄 Bagian 4: Feature Selection - revisi struktur
5. 🔄 Bagian 5-6: Modeling dan Statistik - penyesuaian bahasa

---

**FILE INI ADALAH PANDUAN KOMPREHENSIF UNTUK REVISI MANDIRI KARENA NOTEBOOK TERLALU BESAR (90 CELLS) UNTUK DIEDIT SEKALIGUS MELALUI API.**
