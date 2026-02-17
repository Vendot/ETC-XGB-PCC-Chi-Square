import json
import sys

# Read notebook
with open('Heart_Disease_Analysis_Refactored.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find and fix cell 54
for cell in nb['cells']:
    if cell.get('id') == '#VSC-81b15eee':
        source = ''.join(cell['source'])
        
        # Fix 1: Parameter extraction
        old = "        # Extract parameters\\n        estimators = params['model__estimators']\\n        pcc_thresh"
        new = "        # Extract parameters - ETC dan XGB terpisah\\n        etc_estimators = params.get('model__etc_estimators', None)\\n        xgb_estimators = params.get('model__xgb_estimators', None)\\n        pcc_thresh"
        source = source.replace(old, new)
        
        # Fix 2: Hasil dict
        source = source.replace(
            "'Estimators': estimators,",
            "'ETC_Estimators': etc_estimators,\\n            'XGB_Estimators': xgb_estimators,"
        )
        
        cell['source'] = source.splitlines(True)
        print("✅ Updated cell #VSC-81b15eee")
        break

# Save
with open('Heart_Disease_Analysis_Refactored.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("✅ Notebook saved")
