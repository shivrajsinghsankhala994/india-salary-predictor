import pandas as pd
import numpy as np
import os

# ============================================
#   Day 15 - Bigger Indian Dataset (5000 rows)
# ============================================

np.random.seed(42)
n = 5000

print("=" * 45)
print(" Day 15 - Creating 5000-row Indian Dataset")
print("=" * 45)

# ============================================
# STEP 1 - Real Indian Categories
# ============================================
cities = [
    'Bangalore', 'Mumbai', 'Delhi', 'Gurgaon', 'Hyderabad',
    'Pune', 'Chennai', 'Kolkata', 'Jaipur', 'Ahmedabad',
    'Lucknow', 'Chandigarh', 'Indore', 'Bhopal', 'Nagpur'
]

roles = [
    'Software Developer', 'Full Stack Developer', 'Data Analyst',
    'Data Scientist', 'ML Engineer', 'AI Engineer', 'DevOps Engineer',
    'Cloud Engineer', 'Product Manager', 'HR Manager',
    'QA Tester', 'Business Analyst'
]

educations = [
    '12th/Diploma', "Bachelor's", 'BCA', 'B.Tech',
    'MCA', 'MBA', 'M.Tech', 'PhD'
]

industries = [
    'IT MNC', 'IT Indian Company', 'Fintech', 'E-commerce',
    'Startup', 'Healthcare', 'Manufacturing', 'Government'
]

# ============================================
# STEP 2 - Realistic Salary Multipliers
# ============================================
city_mult = {
    'Bangalore': 1.35, 'Mumbai': 1.3, 'Delhi': 1.25, 'Gurgaon': 1.28,
    'Hyderabad': 1.2, 'Pune': 1.15, 'Chennai': 1.12, 'Kolkata': 1.05,
    'Jaipur': 1.0, 'Ahmedabad': 1.0, 'Lucknow': 0.95, 'Chandigarh': 0.97,
    'Indore': 0.93, 'Bhopal': 0.92, 'Nagpur': 0.92
}

role_mult = {
    'Software Developer': 1.0, 'Full Stack Developer': 1.08,
    'Data Analyst': 1.05, 'Data Scientist': 1.25, 'ML Engineer': 1.3,
    'AI Engineer': 1.35, 'DevOps Engineer': 1.15, 'Cloud Engineer': 1.18,
    'Product Manager': 1.4, 'HR Manager': 0.85, 'QA Tester': 0.9,
    'Business Analyst': 1.05
}

edu_mult = {
    '12th/Diploma': 0.72, "Bachelor's": 0.85, 'BCA': 0.88,
    'B.Tech': 1.0, 'MCA': 1.1, 'MBA': 1.18, 'M.Tech': 1.22, 'PhD': 1.35
}

ind_mult = {
    'IT MNC': 1.35, 'IT Indian Company': 1.05, 'Fintech': 1.2,
    'E-commerce': 1.15, 'Startup': 1.1, 'Healthcare': 1.0,
    'Manufacturing': 0.88, 'Government': 0.8
}

# ============================================
# STEP 3 - Data Generate karo
# ============================================
print("\n" + "=" * 45)
print(" STEP 3 : Generating Realistic Data")
print("=" * 45)

data = {
    'name'       : [f'Employee_{i}' for i in range(1, n + 1)],
    'age'        : np.random.randint(22, 58, n),
    'experience' : np.random.randint(0, 28, n),
    'city'       : np.random.choice(cities, n),
    'role'       : np.random.choice(roles, n),
    'education'  : np.random.choice(educations, n),
    'industry'   : np.random.choice(industries, n),
}

df = pd.DataFrame(data)

# Salary realistic formula se banao
base = 3.0
salaries = []

for _, row in df.iterrows():
    exp_bonus = min(row['experience'] * 0.32, 7.0)
    sal = ((base + exp_bonus)
           * edu_mult[row['education']]
           * role_mult[row['role']]
           * city_mult[row['city']]
           * ind_mult[row['industry']])
    noise = np.random.normal(1.0, 0.1)
    salaries.append(round(max(sal * noise, 2.5), 2))

df['salary_lpa'] = salaries

print(f" {n} rows generated successfully!")

# ============================================
# STEP 4 - Real-World Issues Add karo
# ============================================
print("\n" + "=" * 45)
print(" STEP 4 : Adding Real-World Data Issues")
print("=" * 45)

# Missing values
miss_idx = np.random.choice(df.index, size=80, replace=False)
df.loc[miss_idx[:40], 'education'] = np.nan
df.loc[miss_idx[40:], 'industry']  = np.nan

# Duplicate rows
dup_rows = df.sample(20)
df = pd.concat([df, dup_rows], ignore_index=True)

# Outliers (kuch CEO level salaries)
outlier_idx = np.random.choice(df.index, size=10, replace=False)
df.loc[outlier_idx, 'salary_lpa'] = np.random.uniform(60, 90, 10)

print(f" Missing values added : 80")
print(f" Duplicate rows added : 20")
print(f" Outliers added       : 10")
print(f" Final dataset size   : {df.shape[0]} rows")

# ============================================
# STEP 5 - Save karo
# ============================================
os.makedirs('day15', exist_ok=True)
df.to_csv('day15/indian_salary_5000.csv', index=False)

print("\n" + "=" * 45)
print(" STEP 5 : Dataset Saved")
print("=" * 45)
print(f" File : day15/indian_salary_5000.csv")
print(f" Shape : {df.shape}")

# ============================================
# STEP 6 - Quick Insights
# ============================================
print("\n" + "=" * 45)
print(" STEP 6 : Quick Insights")
print("=" * 45)

print(f"\n Average Salary : {df['salary_lpa'].mean():.2f} LPA")
print(f" Median Salary  : {df['salary_lpa'].median():.2f} LPA")

print(f"\n Top 5 Highest Paying Cities:")
city_avg = df.groupby('city')['salary_lpa'].mean().sort_values(ascending=False)
for city, sal in city_avg.head(5).items():
    print(f"   {city:<15} : {sal:.2f} LPA")

print(f"\n Top 5 Highest Paying Roles:")
role_avg = df.groupby('role')['salary_lpa'].mean().sort_values(ascending=False)
for role, sal in role_avg.head(5).items():
    print(f"   {role:<22} : {sal:.2f} LPA")

print("\n" + "=" * 45)
print(" Day 15 Complete!")
print(" Next : Day 16 - Retrain model on this new data")
print("=" * 45)