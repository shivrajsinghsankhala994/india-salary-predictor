import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

# ============================================
#   Day 6 - Encoding + Scaling
# ============================================

df = pd.read_csv('day5/featured_employees.csv')

print("=" * 45)
print(" Day 6 - Encoding Started")
print("=" * 45)
print(f" Rows    : {df.shape[0]}")
print(f" Columns : {list(df.columns)}")

# ============================================
# STEP 1 - LabelEncoder (Education + Role)
# ============================================
print("\n" + "=" * 45)
print(" STEP 1 : Label Encoding")
print("=" * 45)

le_edu  = LabelEncoder()
le_role = LabelEncoder()
le_city = LabelEncoder()

df['education_encoded'] = le_edu.fit_transform(df['education'])
df['role_encoded']      = le_role.fit_transform(df['role'])
df['city_encoded']      = le_city.fit_transform(df['city'])

print("\n Education Encoding:")
for orig, enc in zip(le_edu.classes_,
                     le_edu.transform(le_edu.classes_)):
    print(f"   {orig:<20} -> {enc}")

print("\n Role Encoding:")
for orig, enc in zip(le_role.classes_,
                     le_role.transform(le_role.classes_)):
    print(f"   {orig:<20} -> {enc}")

# ============================================
# STEP 2 - OneHotEncoding (City)
# ============================================
print("\n" + "=" * 45)
print(" STEP 2 : OneHot Encoding (City)")
print("=" * 45)

city_dummies = pd.get_dummies(df['city'], prefix='city')
df = pd.concat([df, city_dummies], axis=1)

print(f" New city columns : {list(city_dummies.columns)}")

# ============================================
# STEP 3 - Feature Scaling
# ============================================
print("\n" + "=" * 45)
print(" STEP 3 : Feature Scaling")
print("=" * 45)

scaler = StandardScaler()

df[['age_scaled', 'experience_scaled']] = scaler.fit_transform(
    df[['age', 'experience']]
)

print(f" Before -> Experience range : {df['experience'].min()} - {df['experience'].max()}")
print(f" After  -> Experience range : {df['experience_scaled'].min():.2f} - {df['experience_scaled'].max():.2f}")

# ============================================
# STEP 4 - ML Ready Dataset
# ============================================
print("\n" + "=" * 45)
print(" STEP 4 : Final ML-Ready Dataset")
print("=" * 45)

ml_cols = [
    'age_scaled',
    'experience_scaled',
    'edu_numeric',
    'role_encoded',
    'city_encoded',
    'city_tier',
    'salary_lpa'
]

df_ml = df[ml_cols].copy()

print(f" Shape   : {df_ml.shape}")
print(f" Columns : {list(df_ml.columns)}")
print(f"\n First 3 rows:")
print(df_ml.head(3).round(2).to_string(index=False))

# ============================================
# STEP 5 - Save
# ============================================
os.makedirs('day6', exist_ok=True)
df_ml.to_csv('day6/ml_ready_data.csv', index=False)

print("\n" + "=" * 45)
print(" Day 6 Complete!")
print(f" Rows    : {df_ml.shape[0]}")
print(f" Columns : {df_ml.shape[1]}")
print(" Saved   : day6/ml_ready_data.csv")
print("=" * 45)