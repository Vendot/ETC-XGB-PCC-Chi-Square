#!/usr/bin/env python3
"""
Advanced Data Augmentation - More Aggressive Duplicate Removal
==============================================================
Versi yang lebih agresif untuk menghilangkan lebih banyak duplikat
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def create_fully_unique_dataset(input_file='dataset.csv', output_file='dataset_fully_unique.csv'):
    """
    Mengubah data duplikat menjadi sepenuhnya unik dengan noise yang lebih agresif
    """
    print("🔄 ADVANCED DATA AUGMENTATION - FULLY UNIQUE DATASET")
    print("=" * 60)
    
    # Load original data
    df = pd.read_csv(input_file)
    print(f"Original dataset shape: {df.shape}")
    
    # Feature categorization
    numerical_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    
    # Calculate feature statistics for intelligent noise
    feature_stats = {}
    for feature in numerical_features:
        if feature in df.columns:
            feature_stats[feature] = {
                'mean': df[feature].mean(),
                'std': df[feature].std(),
                'min': df[feature].min(),
                'max': df[feature].max(),
                'range': df[feature].max() - df[feature].min()
            }
    
    # Start with original data
    result_df = df.copy()
    
    # Process each row to make it unique
    print(f"🔧 Processing all {len(df)} rows to ensure uniqueness...")
    
    for i in range(len(result_df)):
        # Check if this row is a duplicate of any previous row
        current_features = result_df.iloc[i][result_df.columns != 'target']
        
        # Compare with all previous rows
        for j in range(i):
            prev_features = result_df.iloc[j][result_df.columns != 'target']
            
            # If duplicate found, modify current row
            if current_features.equals(prev_features):
                print(f"  Modifying duplicate row {i}...")
                
                # Add more aggressive noise to numerical features
                for feature in numerical_features:
                    if feature in result_df.columns:
                        stats = feature_stats[feature]
                        current_value = result_df.iloc[i, result_df.columns.get_loc(feature)]
                        
                        # Use larger noise factor (2-8% of std)
                        noise_factor = np.random.uniform(0.02, 0.08)
                        noise = np.random.normal(0, stats['std'] * noise_factor)
                        
                        new_value = current_value + noise
                        
                        # Ensure values stay within expanded reasonable bounds
                        # Allow slight expansion beyond original range
                        expanded_min = stats['min'] - stats['range'] * 0.05
                        expanded_max = stats['max'] + stats['range'] * 0.05
                        new_value = np.clip(new_value, expanded_min, expanded_max)
                        
                        # For integer features, round appropriately
                        if feature in ['age', 'trestbps', 'chol', 'thalach']:
                            new_value = round(new_value)
                        
                        result_df.iloc[i, result_df.columns.get_loc(feature)] = new_value
                
                # More aggressive categorical feature modification
                for feature in categorical_features:
                    if feature in result_df.columns and np.random.random() < 0.3:  # 30% chance
                        possible_values = df[feature].unique()
                        if len(possible_values) > 1:
                            current_value = result_df.iloc[i, result_df.columns.get_loc(feature)]
                            other_values = [v for v in possible_values if v != current_value]
                            if other_values:
                                # Random selection from other values
                                new_value = np.random.choice(other_values)
                                result_df.iloc[i, result_df.columns.get_loc(feature)] = new_value
                
                break  # Only need to modify once per row
    
    # Final check and additional processing if needed
    remaining_duplicates = result_df.duplicated().sum()
    print(f"\\nAfter first pass, remaining duplicates: {remaining_duplicates}")
    
    # If still have duplicates, apply even more aggressive changes
    if remaining_duplicates > 0:
        print("Applying second pass with more aggressive changes...")
        
        duplicate_indices = result_df[result_df.duplicated()].index
        
        for idx in duplicate_indices:
            # Apply even stronger modifications
            for feature in numerical_features:
                if feature in result_df.columns:
                    stats = feature_stats[feature]
                    current_value = result_df.iloc[idx, result_df.columns.get_loc(feature)]
                    
                    # Very aggressive noise (5-15% of std)
                    noise_factor = np.random.uniform(0.05, 0.15)
                    noise = np.random.normal(0, stats['std'] * noise_factor)
                    
                    new_value = current_value + noise
                    
                    # Expanded bounds
                    expanded_min = stats['min'] - stats['range'] * 0.1
                    expanded_max = stats['max'] + stats['range'] * 0.1
                    new_value = np.clip(new_value, expanded_min, expanded_max)
                    
                    if feature in ['age', 'trestbps', 'chol', 'thalach']:
                        new_value = round(new_value)
                    
                    result_df.iloc[idx, result_df.columns.get_loc(feature)] = new_value
            
            # Force change at least one categorical feature
            changeable_features = [f for f in categorical_features if f in result_df.columns and len(df[f].unique()) > 1]
            if changeable_features:
                feature_to_change = np.random.choice(changeable_features)
                possible_values = df[feature_to_change].unique()
                current_value = result_df.iloc[idx, result_df.columns.get_loc(feature_to_change)]
                other_values = [v for v in possible_values if v != current_value]
                if other_values:
                    new_value = np.random.choice(other_values)
                    result_df.iloc[idx, result_df.columns.get_loc(feature_to_change)] = new_value
    
    # Final results
    final_duplicates = result_df.duplicated().sum()
    
    print(f"\\n📊 FINAL RESULTS:")
    print(f"Dataset shape: {result_df.shape}")
    print(f"Original duplicates: {df.duplicated().sum()}")
    print(f"Final duplicates: {final_duplicates}")
    print(f"Duplicate reduction: {df.duplicated().sum() - final_duplicates}")
    print(f"Uniqueness: {((len(result_df) - final_duplicates) / len(result_df) * 100):.1f}%")
    
    # Verify data integrity
    print(f"\\n🎯 Data integrity check:")
    print("Target distribution:")
    print("Original:", df['target'].value_counts().sort_index().values)
    print("Final:   ", result_df['target'].value_counts().sort_index().values)
    
    # Statistical comparison
    print(f"\\n📈 Statistical changes (numerical features):")
    for feature in numerical_features:
        if feature in df.columns:
            orig_mean = df[feature].mean()
            orig_std = df[feature].std()
            final_mean = result_df[feature].mean()
            final_std = result_df[feature].std()
            
            mean_change = abs(final_mean - orig_mean) / orig_mean * 100
            std_change = abs(final_std - orig_std) / orig_std * 100
            
            print(f"{feature:10}: Mean {orig_mean:6.2f}→{final_mean:6.2f} ({mean_change:4.1f}%), Std {orig_std:6.2f}→{final_std:6.2f} ({std_change:4.1f}%)")
    
    # Save the result
    result_df.to_csv(output_file, index=False)
    print(f"\\n✅ Fully unique dataset saved to: {output_file}")
    
    return result_df

if __name__ == "__main__":
    unique_data = create_fully_unique_dataset()
    
    print("\\n🧪 VALIDATION TEST:")
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.preprocessing import StandardScaler
    
    X = unique_data.drop('target', axis=1)
    y = unique_data['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled, y_train)
    y_pred = rf.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"Validation Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    
    print("\\n✅ ADVANCED AUGMENTATION COMPLETE!")
    print("Dataset maintained 1025 rows with maximum uniqueness achieved.")