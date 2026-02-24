# ✅ IMPLEMENTASI NESTED CV 5x5 - SUDAH BENAR

## 📋 Struktur Nested CV yang Diimplementasikan:

### **Outer Loop (5-fold StratifiedKFold)**
- Purpose: Evaluasi generalisasi model yang unbiased
- Data: Seluruh dataset (X, y)  
- Untuk setiap fold:
  - Split menjadi: `X_train_outer` (80%) dan `X_test_outer` (20%)

### **Inner Loop (5-fold GridSearchCV)**
- Purpose: Hyperparameter tuning
- Data: `X_train_outer` dari outer fold
- GridSearchCV mencari best hyperparameters menggunakan 5-fold CV

### **Proses Per Outer Fold:**
```
Outer Fold 1:
  → Train outer (80% data) → GridSearchCV (5-fold inner CV)
  → Best model → Evaluate on Test outer (20% data) → Outer score 1
  
Outer Fold 2:
  → Train outer (80% data) → GridSearchCV (5-fold inner CV)
  → Best model → Evaluate on Test outer (20% data) → Outer score 2
  
Outer Fold 3:
  → Train outer (80% data) → GridSearchCV (5-fold inner CV)
  → Best model → Evaluate on Test outer (20% data) → Outer score 3
  
Outer Fold 4:
  → Train outer (80% data) → GridSearchCV (5-fold inner CV)
  → Best model → Evaluate on Test outer (20% data) → Outer score 4
  
Outer Fold 5:
  → Train outer (80% data) → GridSearchCV (5-fold inner CV)
  → Best model → Evaluate on Test outer (20% data) → Outer score 5

Final Outer CV Score = Mean(5 outer scores) ± Std(5 outer scores)
```

### **Final GridSearchCV pada Full Data:**
- Purpose: Mendapatkan best configuration untuk production
- Data: Seluruh dataset (X, y)
- Menghasilkan: Inner CV scores untuk semua kombinasi parameter

## 📊 Output yang Dihasilkan:

### 1. **df_cv_biasa** (atau df_nested_cv)
Berisi semua kombinasi parameter dengan:
- **Inner CV scores**: `Accuracy_Mean`, `Precision_Mean`, `Recall_Mean`, `F1_Mean`
- **Outer CV scores**: `Outer_CV_Accuracy_Mean`, `Outer_CV_Precision_Mean`, dll.
- **Gap**: Inner CV - Outer CV (menunjukkan hyperparameter overfitting)

### 2. **df_best_per_scenario**
Best configuration per scenario dengan:
- `Inner_CV_Accuracy`: Best dari GridSearchCV
- `Outer_CV_Accuracy`: Unbiased estimate
- `Outer_CV_Std`: Stability estimate

## 🎯 Interpretasi Hasil:

### **Inner CV Accuracy**
- Dari GridSearchCV pada data training
- Cenderung **optimistic** (lebih tinggi)
- Digunakan untuk **memilih best hyperparameters**

### **Outer CV Accuracy**
- Dari evaluasi pada outer test folds
- **Unbiased estimate** of true performance
- Ini yang **harus dilaporkan** di paper/publikasi

### **Gap (Inner - Outer)**
- Gap kecil (<0.02): ✅ Excellent - model stabil
- Gap sedang (0.02-0.05): ⚠️ Good - cukup stabil  
- Gap besar (>0.05): ❌ Warning - hyperparameter overfitting

## 🔧 Perubahan dari Sebelumnya:

### ❌ SEBELUM (SALAH):
```python
# Split data 80/20
X_train, X_test, y_train, y_test = train_test_split(...)

# GridSearchCV pada train data
grid_search.fit(X_train, y_train)

# Evaluate pada test data
y_pred = grid_search.predict(X_test)
```
**Masalah**: Test set terpapar saat hyperparameter tuning (data leakage!)

### ✅ SEKARANG (BENAR):
```python
# Nested CV
for train_outer_idx, test_outer_idx in outer_cv.split(X, y):
    X_train_outer, X_test_outer = X.iloc[train_outer_idx], X.iloc[test_outer_idx]
    
    # GridSearchCV pada train outer
    grid_search.fit(X_train_outer, y_train_outer)
    
    # Evaluate best model pada test outer (UNBIASED!)
    y_pred = grid_search.best_estimator_.predict(X_test_outer)
    outer_scores.append(accuracy)

# Final: Mean outer scores = Unbiased estimate
```

## 📝 Kode Cell yang Diubah:

1. **Cell 49** (Markdown): Penjelasan Nested CV
2. **Cell 50** (DIHAPUS): Train-test split
3. **Cell 53** (Code): Implementasi Nested CV 5x5  
4. **Cell 62** (Code): Analisis Inner vs Outer CV
5. **Cell 63** (Code): Best model per scenario analysis
6. **Cell 64** (Code): Line chart Inner vs Outer CV
7. **Cell 66** (Code): Final model training pada full data

## ✅ KESIMPULAN:

**Implementasi Nested CV 5x5 sudah BENAR!**

- ✅ Outer loop: 5-fold untuk evaluasi generalisasi
- ✅ Inner loop: 5-fold GridSearchCV untuk tuning
- ✅ Unbiased performance estimate dari outer CV
- ✅ No data leakage
- ✅ Production-ready best model dari final GridSearchCV
