# ================================================================================
# 🎯 PENDEKATAN 3: RE-RUN GridSearchCV PADA FULL DATA
# ================================================================================
# TUJUAN: Mendapatkan SATU set parameter terbaik final untuk setiap scenario
# METODOLOGI: Gold standard untuk penelitian - memaksimalkan data untuk final model
# ================================================================================

print("=" * 80)
print("🎯 TAHAP FINAL: GridSearchCV pada FULL DATA")
print("=" * 80)
print("⚠️  PENTING:")
print("   • Nested CV (5x5) memberikan estimasi performa OBJEKTIF")
print("   • GridSearchCV ini untuk mendapat PARAMETER FINAL model deployment")
print("   • Performa yang dilaporkan tetap dari Nested CV, BUKAN dari sini!")
print("=" * 80)
print()

import time
waktu_mulai_final = time.time()

final_best_params = []

for scenario_name, param_grid in param_grids.items():
    print(f"📍 {scenario_name.upper()}: Mencari best parameters pada FULL dataset...")
    
    # GridSearchCV pada SELURUH data
    final_grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=inner_cv,  # Gunakan inner_cv yang sama (5-fold)
        scoring=scoring,
        refit='f1',  # Refit dengan F1 score
        return_train_score=False,
        n_jobs=-1,
        verbose=0
    )
    
    # Fit pada FULL data (X, y)
    final_grid_search.fit(X, y)
    
    # Simpan best parameters
    final_params = final_grid_search.best_params_
    
    # Format parameter untuk disimpan
    param_str_parts = []
    for key, value in final_params.items():
        # Remove 'feature_selector__' and 'model__' prefix
        clean_key = key.replace('feature_selector__', '').replace('model__', '')
        param_str_parts.append(f"{clean_key}={value}")
    
    final_param_string = ', '.join(param_str_parts) + f', scenario={scenario_name}'
    
    final_best_params.append({
        'Scenario': scenario_name,
        'Final_Best_Params_String': final_param_string,
        'Final_Best_Params_Dict': final_params,
        'Final_Best_CV_Score': final_grid_search.best_score_,  # Mean CV score
        'Final_Best_Model': final_grid_search.best_estimator_
    })
    
    print(f"   ✓ Best Params: {final_param_string}")
    print(f"   ✓ Cross-Val F1 Score: {final_grid_search.best_score_:.4f}")
    print()

# Create DataFrame for final results
df_final_params = pd.DataFrame(final_best_params)

waktu_final = time.time() - waktu_mulai_final

print("=" * 80)
print("✅ FINAL GridSearchCV SELESAI")
print("=" * 80)
print(f"⏱️  Waktu eksekusi: {waktu_final:.1f} detik ({waktu_final/60:.1f} menit)")
print()

# ================================================================================
# 📊 MENAMBAHKAN FINAL PARAMETERS KE df_cv_biasa
# ================================================================================

# Extract clean parameters from Final_Best_Params_String
def extract_final_params(params_string):
    """Extract final best parameters from string format"""
    params_dict = {}
    for item in params_string.split(', '):
        if '=' in item and 'scenario' not in item:
            key, value = item.split('=', 1)
            # Try to convert to appropriate type
            try:
                if '.' in value:
                    value = float(value)
                else:
                    value = int(value)
            except:
                pass  # Keep as string
            params_dict[key] = value
    
    return {
        'Final_PCC_Threshold': params_dict.get('pcc_thresh'),
        'Final_Chi2_K': params_dict.get('chi2_k'),
        'Final_ETC_n_estimators': params_dict.get('etc_n_estimators'),
        'Final_XGB_n_estimators': params_dict.get('xgb_n_estimators')
    }

# Add final parameters to df_cv_biasa
final_param_extracted = df_final_params['Final_Best_Params_String'].apply(extract_final_params).apply(pd.Series)
df_cv_biasa_with_final = pd.concat([df_cv_biasa, final_param_extracted], axis=1)

print("=" * 80)
print("📊 DATAFRAME DENGAN FINAL PARAMETERS")
print("=" * 80)
print("Kolom yang ditambahkan:")
print("   • Final_PCC_Threshold: Threshold PCC terbaik dari full data")
print("   • Final_Chi2_K: K optimal Chi2 dari full data")
print("   • Final_ETC_n_estimators: n_estimators ETC terbaik dari full data")
print("   • Final_XGB_n_estimators: n_estimators XGB terbaik dari full data")
print()
display(df_cv_biasa_with_final)

# ================================================================================
# 💡 INTERPRETASI HASIL
# ================================================================================

print()
print("=" * 80)
print("💡 CARA MEMBACA HASIL INI:")
print("=" * 80)
print()
print("1️⃣  PERFORMA MODEL (dari Nested CV):")
print("   • Test_Accuracy_Mean, Test_F1_Mean, dll.")
print("   • INI adalah estimasi performa OBJEKTIF (unbiased)")
print("   • Gunakan angka INI untuk melaporkan performa model")
print()
print("2️⃣  PARAMETER TERBAIK (dari Final GridSearchCV):")
print("   • Final_PCC_Threshold, Final_Chi2_K, dll.")
print("   • INI adalah parameter untuk DEPLOY model ke production")
print("   • Train final model dengan parameter INI pada FULL data")
print()
print("3️⃣  WORKFLOW DEPLOYMENT:")
print("   a. Report performa: Test_Accuracy_Mean (dari Nested CV)")
print("   b. Train final model: Gunakan Final_* parameters pada FULL data")
print("   c. Deploy: Model final siap untuk production")
print()
print("=" * 80)
print("🎓 BEST PRACTICE DARI LITERATUR:")
print("=" * 80)
print("• Cawley & Talbot (2010): 'On over-fitting in model selection'")
print("• Varma & Simon (2006): 'Bias in error estimation'")
print("• Scikit-learn Documentation: 'Nested Cross-Validation'")
print("=" * 80)
print()

# ================================================================================
# 📝 CONTOH PENGGUNAAN FINAL MODEL
# ================================================================================

print("=" * 80)
print("📝 CONTOH: Deploy Final Model untuk Scenario BASELINE")
print("=" * 80)

baseline_final_model = df_final_params[df_final_params['Scenario'] == 'baseline']['Final_Best_Model'].values[0]
baseline_test_acc = df_cv_biasa_with_final[df_cv_biasa_with_final['Scenario'] == 'baseline']['Test_Accuracy_Mean'].values[0]

print(f"✓ Final Model telah di-train pada FULL data")
print(f"✓ Expected Test Accuracy: {baseline_test_acc:.4f} (dari Nested CV)")
print(f"✓ Model siap untuk prediction:")
print()
print("   # Contoh prediction:")
print("   # predictions = baseline_final_model.predict(X_new_data)")
print()
print("=" * 80)
