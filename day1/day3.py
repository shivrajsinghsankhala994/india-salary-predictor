import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Load dataset
df = pd.read_csv('day2/employees.csv')

print("=" * 45)
print(" STEP 1 : Original Dataset Info")
print("=" * 45)
print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")

# -----------------------------------------------
# STEP 2 : Check Missing Values
# -----------------------------------------------
print("\n" + "=" * 45)
print(" STEP 2 : Missing Values Check")
print("=" * 45)

for col in df.columns:
    count = df[col].isnull().sum()
    status = f"{count} missing" if count > 0 else "OK"
    print(f"  {col:<15} -> {status}")

# -----------------------------------------------
# STEP 3 : Fix Missing Values
# -----------------------------------------------
print("\n" + "=" * 45)
print(" STEP 3 : Fixing Missing Values")
print("=" * 45)

# Education -> fill with most common value
edu_mode = df['education'].mode()[0]
df['education'] = df['education'].fillna(edu_mode)
print(f"  Education  -> filled with '{edu_mode}'")

# Salary -> fill with median (better than mean for salary)
sal_median = df['salary_lpa'].median()
df['salary_lpa'] = df['salary_lpa'].fillna(sal_median)
print(f"  Salary     -> filled with {sal_median} LPA")

# Verify
total_missing = df.isnull().sum().sum()
print(f"\n  Missing values remaining : {total_missing}")

# -----------------------------------------------
# STEP 4 : Remove Duplicates
# -----------------------------------------------
print("\n" + "=" * 45)
print(" STEP 4 : Removing Duplicates")
print("=" * 45)

before = len(df)
df = df.drop_duplicates()
after = len(df)

print(f"  Before : {before} rows")
print(f"  After  : {after} rows")
print(f"  Removed: {before - after} duplicate rows")

# -----------------------------------------------
# STEP 5 : Remove Outliers (IQR Method)
# -----------------------------------------------
print("\n" + "=" * 45)
print(" STEP 5 : Removing Salary Outliers")
print("=" * 45)

Q1  = df['salary_lpa'].quantile(0.25)
Q3  = df['salary_lpa'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"  Q1          : {Q1:.2f} LPA")
print(f"  Q3          : {Q3:.2f} LPA")
print(f"  IQR         : {IQR:.2f}")
print(f"  Lower Bound : {lower_bound:.2f} LPA")
print(f"  Upper Bound : {upper_bound:.2f} LPA")

before = len(df)
df = df[
    (df['salary_lpa'] >= lower_bound) &
    (df['salary_lpa'] <= upper_bound)
]
after = len(df)

print(f"\n  Outliers Removed : {before - after}")
print(f"  Clean Rows       : {after}")

# -----------------------------------------------
# STEP 6 : Save Clean Data
# -----------------------------------------------
os.makedirs('day3', exist_ok=True)
df.to_csv('day3/clean_employees.csv', index=False)

print("\n" + "=" * 45)
print(" STEP 6 : Final Clean Dataset")
print("=" * 45)
print(f"  Rows          : {df.shape[0]}")
print(f"  Columns       : {df.shape[1]}")
print(f"  Missing Values: {df.isnull().sum().sum()}")
print(f"  Saved to      : day3/clean_employees.csv")

# -----------------------------------------------
# STEP 7 : Plot Before vs After
# -----------------------------------------------
df_raw = pd.read_csv('day2/employees.csv')

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Day 3 - Before vs After Cleaning', fontsize=14)

axes[0].hist(df_raw['salary_lpa'].dropna(), bins=20,
             color='#e74c3c', edgecolor='white')
axes[0].set_title('Before Cleaning')
axes[0].set_xlabel('Salary (LPA)')
axes[0].set_ylabel('Count')

axes[1].hist(df['salary_lpa'], bins=20,
             color='#2ecc71', edgecolor='white')
axes[1].set_title('After Cleaning')
axes[1].set_xlabel('Salary (LPA)')
axes[1].set_ylabel('Count')

plt.tight_layout()
plt.savefig('day3/day3_chart.png', dpi=150)
plt.show()

print("\n  Chart saved -> day3/day3_chart.png")
print("\n  Day 3 Done!")