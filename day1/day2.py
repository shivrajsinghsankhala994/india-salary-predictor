import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================
#   Day 2 - 1000 Employees Dataset
# ============================================

# --- Create 1000 employees data ---
np.random.seed(42)
n = 1000

cities     = ['Bangalore', 'Mumbai', 'Delhi', 'Hyderabad', 'Pune', 'Jaipur']
roles      = ['Software Developer', 'Data Analyst', 'Data Scientist', 'ML Engineer', 'HR', 'DevOps']
educations = ['12th/Diploma', "Bachelor's", 'B.Tech', 'MBA/M.Tech', 'PhD']

data = {
    'name'       : [f'Employee_{i}' for i in range(1, n+1)],
    'age'        : np.random.randint(22, 55, n),
    'experience' : np.random.randint(0, 25, n),
    'city'       : np.random.choice(cities, n),
    'role'       : np.random.choice(roles, n),
    'education'  : np.random.choice(educations, n),
    'salary_lpa' : np.round(np.random.uniform(3.0, 25.0, n), 2)
}

df = pd.DataFrame(data)

# --- Add some missing values (like real data) ---
df.loc[10:20, 'education']  = None
df.loc[50:55, 'salary_lpa'] = None

# --- Save to CSV file ---
df.to_csv('day2/employees.csv', index=False)

print("┌─────────────────────────────────────┐")
print("│       CSV FILE SAVED SUCCESSFULLY   │")
print("└─────────────────────────────────────┘")

# ============================================
# STEP 1: Load Dataset and Show Basic Info
# ============================================
df = pd.read_csv('day2/employees.csv')

print()
print("┌─────────────────────────────────────┐")
print("│         DATASET INFORMATION         │")
print("└─────────────────────────────────────┘")
print(f"  Total Rows       : {df.shape[0]}")
print(f"  Total Columns    : {df.shape[1]}")
print(f"  Column Names     : {list(df.columns)}")

# ============================================
# STEP 2: Show First 5 Rows
# ============================================
print()
print("┌─────────────────────────────────────┐")
print("│            FIRST 5 ROWS             │")
print("└─────────────────────────────────────┘")
print(df.head().to_string(index=False))

# ============================================
# STEP 3: Check Missing Values
# ============================================
print()
print("┌─────────────────────────────────────┐")
print("│           MISSING VALUES            │")
print("└─────────────────────────────────────┘")
for col in df.columns:
    missing = df[col].isnull().sum()
    if missing > 0:
        print(f"  ⚠️  {col:<15} : {missing} missing values found")
    else:
        print(f"  ✅ {col:<15} : No missing values")

# ============================================
# STEP 4: Salary Statistics
# ============================================
print()
print("┌─────────────────────────────────────┐")
print("│          SALARY STATISTICS          │")
print("└─────────────────────────────────────┘")
print(f"  Highest Salary   : {df['salary_lpa'].max():.2f} LPA")
print(f"  Lowest  Salary   : {df['salary_lpa'].min():.2f} LPA")
print(f"  Average Salary   : {df['salary_lpa'].mean():.2f} LPA")
print(f"  Total Employees  : {df.shape[0]}")

# ============================================
# STEP 5: City-wise Average Salary
# ============================================
print()
print("┌─────────────────────────────────────┐")
print("│       CITY-WISE AVERAGE SALARY      │")
print("└─────────────────────────────────────┘")
city_avg = df.groupby('city')['salary_lpa'].mean().sort_values(ascending=False)
for city, salary in city_avg.items():
    print(f"  🏙️  {city:<15} : {salary:.2f} LPA")

# ============================================
# STEP 6: Role-wise Average Salary
# ============================================
print()
print("┌─────────────────────────────────────┐")
print("│       ROLE-WISE AVERAGE SALARY      │")
print("└─────────────────────────────────────┘")
role_avg = df.groupby('role')['salary_lpa'].mean().sort_values(ascending=False)
for role, salary in role_avg.items():
    print(f"  💼 {role:<22} : {salary:.2f} LPA")

# ============================================
# STEP 7: Create 4 Charts
# ============================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Day 2 - 1000 Employees Salary Analysis', fontsize=16, fontweight='bold')

# Chart 1: Salary Distribution
axes[0,0].hist(df['salary_lpa'].dropna(), bins=20, color='#2ecc71', edgecolor='white')
axes[0,0].set_title('Salary Distribution')
axes[0,0].set_xlabel('Salary (LPA)')
axes[0,0].set_ylabel('Number of Employees')
axes[0,0].axvline(df['salary_lpa'].mean(), color='red', linestyle='--', label='Average')
axes[0,0].legend()

# Chart 2: City-wise Average Salary
city_avg.plot(kind='bar', ax=axes[0,1], color='#3498db', edgecolor='white')
axes[0,1].set_title('City-wise Average Salary')
axes[0,1].set_xlabel('City')
axes[0,1].set_ylabel('Avg Salary (LPA)')
axes[0,1].tick_params(axis='x', rotation=30)

# Chart 3: Experience vs Salary
axes[1,0].scatter(df['experience'], df['salary_lpa'], alpha=0.4, color='#e74c3c', s=15)
axes[1,0].set_title('Experience vs Salary')
axes[1,0].set_xlabel('Experience (Years)')
axes[1,0].set_ylabel('Salary (LPA)')

# Chart 4: Role-wise Average Salary
role_avg_asc = df.groupby('role')['salary_lpa'].mean().sort_values(ascending=True)
role_avg_asc.plot(kind='barh', ax=axes[1,1], color='#9b59b6', edgecolor='white')
axes[1,1].set_title('Role-wise Average Salary')
axes[1,1].set_xlabel('Avg Salary (LPA)')

plt.tight_layout()
plt.savefig('day2/day2_charts.png', dpi=150, bbox_inches='tight')
plt.show()

print()
print("┌─────────────────────────────────────┐")
print("│          DAY 2 COMPLETED!           │")
print("│  CSV File  : day2/employees.csv     │")
print("│  Chart     : day2/day2_charts.png   │")
print("└─────────────────────────────────────┘")