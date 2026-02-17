# Update Parameter Grids untuk Memisahkan ETC dan XGB Estimators

## Perubahan yang Dilakukan

### 1. Class ETCXGBHybrid sudah diupdate ✅
Class sekarang menerima parameter terpisah:
- `etc_estimators`: untuk ExtraTreesClassifier  
- `xgb_estimators`: untuk XGBClassifier

### 2. Update param_grids

Ganti kode param_grids dengan kode berikut:

```python
# Konfigurasi parameter grid untuk eksperimen

# Parameter lists untuk ETC dan XGB terpisah
ETC_ESTIMATORS_LIST = [200, 300, 400]  # Untuk ExtraTreesClassifier
XGB_ESTIMATORS_LIST = [50, 100, 150]   # Untuk XGBClassifier
PCC_THRESHOLD_LIST = [0.15, 0.2]
CHI2_K_LIST = [1, 2, 3, 4, 5, 6]
RANDOM_STATE = 42

param_grids = {
    'baseline': {
        'selector__scenario': ['baseline'],
        'model__etc_estimators': ETC_ESTIMATORS_LIST,
        'model__xgb_estimators': XGB_ESTIMATORS_LIST
    },
    'pcc': {
        'selector__scenario': ['pcc'],
        'selector__pcc_thresh': PCC_THRESHOLD_LIST,
        'model__etc_estimators': ETC_ESTIMATORS_LIST,
        'model__xgb_estimators': XGB_ESTIMATORS_LIST
    },
    'chi2': {
        'selector__scenario': ['chi2'],
        'selector__chi2_k': CHI2_K_LIST,
        'model__etc_estimators': ETC_ESTIMATORS_LIST,
        'model__xgb_estimators': XGB_ESTIMATORS_LIST
    },
    'combined': {
        'selector__scenario': ['combined'],
        'selector__pcc_thresh': PCC_THRESHOLD_LIST,
        'selector__chi2_k': CHI2_K_LIST,
        'model__etc_estimators': ETC_ESTIMATORS_LIST,
        'model__xgb_estimators': XGB_ESTIMATORS_LIST
    }
}

print("=" * 80)
print("⚙️  KONFIGURASI PARAMETER GRID (ETC & XGB TERPISAH)")
print("=" * 80)

print("\n📊 Parameter Model:")
print(f"  • ETC Estimators: {ETC_ESTIMATORS_LIST} ({len(ETC_ESTIMATORS_LIST)} nilai)")
print(f"  • XGB Estimators: {XGB_ESTIMATORS_LIST} ({len(XGB_ESTIMATORS_LIST)} nilai)")
print(f"  • Random State: {RANDOM_STATE}")

print("\n📊 Parameter Feature Selection:")
print(f"  • PCC Threshold: {PCC_THRESHOLD_LIST} ({len(PCC_THRESHOLD_LIST)} nilai)")
print(f"  • Chi2 K: {CHI2_K_LIST} ({len(CHI2_K_LIST)} nilai)")

print("\n📋 Parameter Grids per Scenario:")
for scenario, params in param_grids.items():
    # Hitung jumlah kombinasi
    n_combinations = 1
    for key, values in params.items():
        n_combinations *= len(values)
    print(f"  {scenario:10s}: {n_combinations:4d} kombinasi")

total_combinations = sum([len(p.get('selector__pcc_thresh', [1])) * 
                         len(p.get('selector__chi2_k', [1])) * 
                         len(p['model__etc_estimators']) * 
                         len(p['model__xgb_estimators']) 
                         for p in param_grids.values()])

print("\n" + "=" * 80)
print(f"📈 TOTAL KOMBINASI: {total_combinations} kombinasi")
print("=" * 80)
print("\n✅ PARAMETER GRID UPDATED:")
print("   - ETC dan XGB sekarang punya parameter n_estimators terpisah")
print("   - model__etc_estimators: untuk ExtraTreesClassifier")
print("   - model__xgb_estimators: untuk XGBClassifier")
print("=" * 80)
```

## Keuntungan Pemisahan Parameter:

1. **Tuning Independen**: Bisa tuning jumlah trees ETC dan boosting rounds XGB secara terpisah
2. **Kombinasi Optimal**: Eksplorasi kombinasi yang lebih baik (misal: ETC=300 + XGB=50)
3. **Resource Management**: XGB biasanya butuh lebih sedikit estimators dari ETC
4. **Best Practice**: ETC benefit dari banyak trees, XGB benefit dari early stopping

## Contoh Kombinasi yang Bisa Diexplorasi:

- **High ETC, Low XGB**: ETC=400 + XGB=50 (fokus ensemble diversity)
- **Medium Both**: ETC=300 + XGB=100 (balanced)  
- **Low ETC, High XGB**: ETC=200 + XGB=150 (fokus boosting)

## Total Kombinasi Setelah Update:

- **Baseline**: 3 × 3 = 9 kombinasi (sebelumnya 3)
- **PCC**: 2 × 3 × 3 = 18 kombinasi (sebelumnya 6)
- **Chi2**: 6 × 3 × 3 = 54 kombinasi (sebelumnya 18)
- **Combined**: 2 × 6 × 3 × 3 = 108 kombinasi (sebelumnya 36)
- **TOTAL**: 189 kombinasi (sebelumnya 63)

⚠️ **Note**: Total kombinasi meningkat 3x lipat. Pertimbangkan untuk:
- Reduce list values jika waktu eksekusi terlalu lama
- Gunakan n_jobs=-1 di GridSearchCV untuk parallel processing
- Atau jalankan secara bertahap per scenario
