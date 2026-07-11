import pandas as pd
import os

# ============================================
#   Day 7 - Week 1 Complete Review
# ============================================

print("=" * 50)
print("   WEEK 1 COMPLETE REVIEW")
print("   Indian Salary Predictor Project")
print("=" * 50)

# ============================================
# STEP 1 - Check all files exist
# ============================================
print("\n" + "=" * 50)
print(" STEP 1 : Project Files Check")
print("=" * 50)

files = {
    'Day 1 - explore.py'          : 'day1/explore.py',
    'Day 2 - employees.csv'       : 'day2/employees.csv',
    'Day 3 - clean_employees.csv' : 'day3/clean_employees.csv',
    'Day 4 - chart_all.png'       : 'day4/chart_all.png',
    'Day 5 - featured_data.csv'   : 'day5/featured_employees.csv',
    'Day 6 - ml_ready_data.csv'   : 'day6/ml_ready_data.csv',
}

all_ok = True
for name, path in files.items():
    exists = os.path.exists(path)
    status = 'OK' if exists else 'MISSING'
    print(f"  {status}  {name}")
    if not exists:
        all_ok = False

if all_ok:
    print("\n  All files present!")
else:
    print("\n  Some files missing — check above")

# ============================================
# STEP 2 - Dataset Journey
# ============================================
print("\n" + "=" * 50)
print(" STEP 2 : Dataset Journey")
print("=" * 50)

try:
    df2 = pd.read_csv('day2/employees.csv')
    df3 = pd.read_csv('day3/clean_employees.csv')
    df5 = pd.read_csv('day5/featured_employees.csv')
    df6 = pd.read_csv('day6/ml_ready_data.csv')

    print(f"  Day 2 - Raw data       : {df2.shape[0]} rows, {df2.shape[1]} cols")
    print(f"  Day 3 - Clean data     : {df3.shape[0]} rows, {df3.shape[1]} cols")
    print(f"  Day 5 - Featured data  : {df5.shape[0]} rows, {df5.shape[1]} cols")
    print(f"  Day 6 - ML ready data  : {df6.shape[0]} rows, {df6.shape[1]} cols")
    print(f"\n  Rows removed in cleaning : {df2.shape[0] - df3.shape[0]}")
    print(f"  New columns added        : {df5.shape[1] - df3.shape[1]}")
except Exception as e:
    print(f"  Error: {e}")

# ============================================
# STEP 3 - Salary Insights
# ============================================
print("\n" + "=" * 50)
print(" STEP 3 : Key Salary Insights")
print("=" * 50)

df = pd.read_csv('day3/clean_employees.csv')

print(f"\n  Average Salary   : {df['salary_lpa'].mean():.2f} LPA")
print(f"  Highest Salary   : {df['salary_lpa'].max():.2f} LPA")
print(f"  Lowest Salary    : {df['salary_lpa'].min():.2f} LPA")

print(f"\n  Top Paying City  : {df.groupby('city')['salary_lpa'].mean().idxmax()}")
print(f"  Top Paying Role  : {df.groupby('role')['salary_lpa'].mean().idxmax()}")
print(f"  Top Education    : {df.groupby('education')['salary_lpa'].mean().idxmax()}")

# ============================================
# STEP 4 - Week 1 Learning Summary
# ============================================
print("\n" + "=" * 50)
print(" STEP 4 : Week 1 Learning Summary")
print("=" * 50)

summary = [
    ("Day 1", "Python basics, DataFrame, bar chart"),
    ("Day 2", "1000 rows CSV, load, missing values"),
    ("Day 3", "Data cleaning, outliers, IQR method"),
    ("Day 4", "4 charts — hist, bar, scatter, boxplot"),
    ("Day 5", "Feature engineering, city tier, exp level"),
    ("Day 6", "LabelEncoding, OneHot, StandardScaler"),
]

for day, learning in summary:
    print(f"  {day}  ->  {learning}")

print("\n" + "=" * 50)
print(" Week 1 Complete!")
print(" Next : Day 8 - First ML Model!")
print("=" * 50)