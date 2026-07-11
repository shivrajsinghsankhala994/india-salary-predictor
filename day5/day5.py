import pandas as pd
import numpy as np
import os

# ============================================
#   Day 5 - Feature Engineering
# ============================================

# Load clean data
df = pd.read_csv('day3/clean_employees.csv')

print("=" * 45)
print(" Day 5 - Feature Engineering Started")
print("=" * 45)
print(f" Rows loaded : {df.shape[0]}")
print(f" Columns     : {list(df.columns)}")

# ============================================
# STEP 1 - City ko Tier mein Convert karo
# ============================================
print("\n" + "=" * 45)
print(" STEP 1 : City -> City Tier")
print("=" * 45)

tier1 = ['Bangalore', 'Mumbai', 'Delhi', 'Hyderabad']
tier2 = ['Pune', 'Chennai', 'Jaipur', 'Ahmedabad']

def get_city_tier(city):
    if city in tier1:
        return 1
    elif city in tier2:
        return 2
    else:
        return 3

df['city_tier'] = df['city'].apply(get_city_tier)

print(df[['city', 'city_tier']].drop_duplicates().sort_values('city_tier').to_string(index=False))

# ============================================
# STEP 2 - Experience Bucket banao
# ============================================
print("\n" + "=" * 45)
print(" STEP 2 : Experience -> Experience Level")
print("=" * 45)

def get_exp_level(exp):
    if exp <= 1:
        return 'Fresher'
    elif exp <= 4:
        return 'Junior'
    elif exp <= 8:
        return 'Mid'
    elif exp <= 15:
        return 'Senior'
    else:
        return 'Lead'

df['exp_level'] = df['experience'].apply(get_exp_level)

print(df.groupby('exp_level')['salary_lpa'].mean().round(2).sort_values(ascending=False).to_string())

# ============================================
# STEP 3 - Age Group banao
# ============================================
print("\n" + "=" * 45)
print(" STEP 3 : Age -> Age Group")
print("=" * 45)

def get_age_group(age):
    if age <= 25:
        return 'Early Career'
    elif age <= 35:
        return 'Mid Career'
    elif age <= 45:
        return 'Senior Career'
    else:
        return 'Expert'

df['age_group'] = df['age'].apply(get_age_group)

print(df.groupby('age_group')['salary_lpa'].mean().round(2).sort_values(ascending=False).to_string())

# ============================================
# STEP 4 - Education ko Number mein Convert
# ============================================
print("\n" + "=" * 45)
print(" STEP 4 : Education -> Numeric")
print("=" * 45)

edu_map = {
    '12th/Diploma'  : 1,
    "Bachelor's"    : 2,
    'B.Tech'        : 3,
    'MBA/M.Tech'    : 4,
    'PhD'           : 5
}

df['edu_numeric'] = df['education'].map(edu_map)

print(df[['education', 'edu_numeric']].drop_duplicates().sort_values('edu_numeric').to_string(index=False))

# ============================================
# STEP 5 - Role ko Number mein Convert
# ============================================
print("\n" + "=" * 45)
print(" STEP 5 : Role -> Numeric")
print("=" * 45)

role_map = {
    'HR'                 : 1,
    'QA Tester'          : 2,
    'Software Developer' : 3,
    'DevOps'             : 4,
    'Data Analyst'       : 5,
    'Data Scientist'     : 6,
    'ML Engineer'        : 7
}

df['role_numeric'] = df['role'].map(role_map)

print(df[['role', 'role_numeric']].drop_duplicates().sort_values('role_numeric').to_string(index=False))

# ============================================
# STEP 6 - New Features Check karo
# ============================================
print("\n" + "=" * 45)
print(" STEP 6 : Final Dataset Columns")
print("=" * 45)
print(f" Old columns : {df.shape[1] - 4}")
print(f" New columns : {df.shape[1]}")
print(f" All columns : {list(df.columns)}")

# ============================================
# STEP 7 - Save karo
# ============================================
os.makedirs('day5', exist_ok=True)
df.to_csv('day5/featured_employees.csv', index=False)

print("\n" + "=" * 45)
print(" Day 5 Complete!")
print(f" Rows    : {df.shape[0]}")
print(f" Columns : {df.shape[1]}")
print(" Saved   : day5/featured_employees.csv")
print("=" * 45)