#!/usr/bin/env python3
"""
Fix cell 54 in Heart_Disease_Analysis_Refactored.ipynb
Add individual ETC and XGB parameter columns
"""

# Read notebook
with open('Heart_Disease_Analysis_Refactored.ipynb', 'r') as f:
    lines = f.readlines()

# Target line to replace (line 3687, index 3686)
old_line = "    \"            'Fold_Accuracies': fold_accuracies  # Simpan per-fold accuracies\\n\",\n"

# New lines to insert
new_lines = [
    "    \"            'Fold_Accuracies': fold_accuracies,\\n\",\n",
    "    \"            # Store individual ETC parameters\\n\",\n",
    "    \"            'ETC_n_estimators': etc_params.get('etc_n_estimators'),\\n\",\n",
    "    \"            'ETC_max_features': etc_params.get('etc_max_features'),\\n\",\n",
    "    \"            'ETC_min_samples_split': etc_params.get('etc_min_samples_split'),\\n\",\n",
    "    \"            'ETC_min_samples_leaf': etc_params.get('etc_min_samples_leaf'),\\n\",\n",
    "    \"            'ETC_max_depth': etc_params.get('etc_max_depth'),\\n\",\n",
    "    \"            'ETC_criterion': etc_params.get('etc_criterion'),\\n\",\n",
    "    \"            'ETC_class_weight': etc_params.get('etc_class_weight'),\\n\",\n",
    "    \"            'ETC_bootstrap': etc_params.get('etc_bootstrap'),\\n\",\n",
    "    \"            'ETC_random_state': etc_params.get('etc_random_state'),\\n\",\n",
    "    \"            # Store individual XGB parameters\\n\",\n",
    "    \"            'XGB_n_estimators': xgb_params.get('xgb_n_estimators'),\\n\",\n",
    "    \"            'XGB_learning_rate': xgb_params.get('xgb_learning_rate'),\\n\",\n",
    "    \"            'XGB_max_depth': xgb_params.get('xgb_max_depth'),\\n\",\n",
    "    \"            'XGB_colsample_bytree': xgb_params.get('xgb_colsample_bytree'),\\n\",\n",
    "    \"            'XGB_subsample': xgb_params.get('xgb_subsample'),\\n\",\n",
    "    \"            'XGB_random_state': xgb_params.get('xgb_random_state')\\n\",\n"
]

# Check and replace
if lines[3686] == old_line:
    lines[3686:3687] = new_lines
    print("✅ Successfully replaced line 3687")
else:
    print(f"❌ Line 3687 doesn't match:")
    print(f"Expected: {repr(old_line)}")
    print(f"Got: {repr(lines[3686])}")
    exit(1)

# Write back
with open('Heart_Disease_Analysis_Refactored.ipynb', 'w') as f:
    f.writelines(lines)

print("✅ File saved successfully")
