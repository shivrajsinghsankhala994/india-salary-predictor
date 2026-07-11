import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# ============================================
#   Day 4 - Data Visualization
# ============================================

# Load clean data
df = pd.read_csv('day3/clean_employees.csv')

print("=" * 45)
print(" Day 4 - Data Visualization Started")
print("=" * 45)
print(f" Rows loaded : {df.shape[0]}")
print(f" Columns     : {list(df.columns)}")

os.makedirs('day4', exist_ok=True)

# ============================================
# CHART 1 - Salary Distribution
# ============================================
plt.figure(figsize=(10, 5))

plt.hist(df['salary_lpa'], bins=25,
         color='#2ecc71', edgecolor='white', linewidth=0.6)

plt.axvline(df['salary_lpa'].mean(),
            color='red', linestyle='--', linewidth=1.5,
            label=f"Avg: {df['salary_lpa'].mean():.1f} LPA")

plt.axvline(df['salary_lpa'].median(),
            color='blue', linestyle='--', linewidth=1.5,
            label=f"Median: {df['salary_lpa'].median():.1f} LPA")

plt.title('Salary Distribution of 1000 Employees',
          fontsize=14, fontweight='bold')
plt.xlabel('Salary (LPA)')
plt.ylabel('Number of Employees')
plt.legend()
plt.tight_layout()
plt.savefig('day4/chart1_salary_distribution.png', dpi=150)
plt.show()
print("\n Chart 1 saved -> salary_distribution.png")

# ============================================
# CHART 2 - City-wise Average Salary
# ============================================
city_avg = df.groupby('city')['salary_lpa'].mean().sort_values(ascending=False)

plt.figure(figsize=(10, 5))

bars = plt.bar(city_avg.index, city_avg.values,
               color=['#1D9E75','#2ecc71','#3498db',
                      '#378ADD','#9b59b6','#e74c3c'],
               edgecolor='white', linewidth=0.6)

# Har bar ke upar value dikhao
for bar, val in zip(bars, city_avg.values):
    plt.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.1,
             f'{val:.1f}', ha='center',
             fontsize=9, fontweight='bold')

plt.title('City-wise Average Salary (LPA)',
          fontsize=14, fontweight='bold')
plt.xlabel('City')
plt.ylabel('Average Salary (LPA)')
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig('day4/chart2_city_salary.png', dpi=150)
plt.show()
print(" Chart 2 saved -> city_salary.png")

# ============================================
# CHART 3 - Experience vs Salary (Scatter)
# ============================================
plt.figure(figsize=(10, 5))

plt.scatter(df['experience'], df['salary_lpa'],
            alpha=0.4, color='#e74c3c',
            s=20, edgecolors='none')

# Trend line add karo
import numpy as np
z = np.polyfit(df['experience'], df['salary_lpa'], 1)
p = np.poly1d(z)
x_line = sorted(df['experience'].unique())
plt.plot(x_line, p(x_line),
         color='#2c3e50', linewidth=2,
         linestyle='--', label='Trend line')

plt.title('Experience vs Salary',
          fontsize=14, fontweight='bold')
plt.xlabel('Experience (Years)')
plt.ylabel('Salary (LPA)')
plt.legend()
plt.tight_layout()
plt.savefig('day4/chart3_experience_salary.png', dpi=150)
plt.show()
print(" Chart 3 saved -> experience_salary.png")

# ============================================
# CHART 4 - Role-wise Salary Boxplot
# ============================================
roles      = df['role'].unique()
role_data  = [df[df['role'] == r]['salary_lpa'].values for r in roles]

plt.figure(figsize=(12, 5))

bp = plt.boxplot(role_data, labels=roles,
                 patch_artist=True,
                 medianprops=dict(color='white', linewidth=2))

colors = ['#1D9E75','#2ecc71','#3498db',
          '#9b59b6','#e74c3c','#f39c12']

for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)

plt.title('Role-wise Salary Range (Boxplot)',
          fontsize=14, fontweight='bold')
plt.xlabel('Job Role')
plt.ylabel('Salary (LPA)')
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig('day4/chart4_role_boxplot.png', dpi=150)
plt.show()
print(" Chart 4 saved -> role_boxplot.png")

# ============================================
# FINAL - Sab Charts Ek Saath
# ============================================
fig = plt.figure(figsize=(16, 12))
gs  = gridspec.GridSpec(2, 2, figure=fig)
fig.suptitle('Day 4 - Complete Salary Analysis',
             fontsize=16, fontweight='bold')

# Chart 1
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(df['salary_lpa'], bins=25,
         color='#2ecc71', edgecolor='white')
ax1.axvline(df['salary_lpa'].mean(),
            color='red', linestyle='--',
            label=f"Avg: {df['salary_lpa'].mean():.1f}")
ax1.set_title('Salary Distribution')
ax1.set_xlabel('Salary (LPA)')
ax1.legend(fontsize=8)

# Chart 2
ax2 = fig.add_subplot(gs[0, 1])
ax2.bar(city_avg.index, city_avg.values,
        color='#3498db', edgecolor='white')
ax2.set_title('City-wise Avg Salary')
ax2.set_xlabel('City')
ax2.tick_params(axis='x', rotation=30)

# Chart 3
ax3 = fig.add_subplot(gs[1, 0])
ax3.scatter(df['experience'], df['salary_lpa'],
            alpha=0.3, color='#e74c3c', s=10)
ax3.plot(x_line, p(x_line),
         color='#2c3e50', linewidth=2, linestyle='--')
ax3.set_title('Experience vs Salary')
ax3.set_xlabel('Experience (Years)')
ax3.set_ylabel('Salary (LPA)')

# Chart 4
ax4 = fig.add_subplot(gs[1, 1])
bp2 = ax4.boxplot(role_data, labels=roles,
                  patch_artist=True,
                  medianprops=dict(color='white', linewidth=1.5))
for patch, color in zip(bp2['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)
ax4.set_title('Role-wise Salary Range')
ax4.tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.savefig('day4/chart_all.png', dpi=150, bbox_inches='tight')
plt.show()

print()
print("=" * 45)
print(" Day 4 Complete!")
print(" All charts saved in day4/ folder")
print("=" * 45)