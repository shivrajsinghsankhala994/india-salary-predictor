import pandas as pd
import matplotlib.pyplot as plt

# Indian employees data
data = {
    'name':       ['Rahul', 'Priya', 'Amit', 'Sneha', 'Vikram'],
    'city':       ['Bangalore', 'Mumbai', 'Jaipur', 'Delhi', 'Pune'],
    'experience': [3, 7, 1, 5, 10],
    'salary_lpa': [6.5, 12.0, 3.8, 9.2, 15.5]
}

df = pd.DataFrame(data)

print("=== Employee List ===")
print(df)

print("\n=== Salary Information ===")
print("Highest Salary :", df['salary_lpa'].max(), "LPA")
print("Lowest Salary  :", df['salary_lpa'].min(), "LPA")
print("Average Salary :", df['salary_lpa'].mean(), "LPA")

print("\n=== City-wise Average Salary ===")
print(df.groupby('city')['salary_lpa'].mean())

# Create chart
plt.figure(figsize=(8, 4))
plt.bar(df['city'], df['salary_lpa'], color='green')
plt.title('City-wise Salary')
plt.xlabel('City')
plt.ylabel('Salary (LPA)')
plt.tight_layout()
plt.savefig('day1_chart.png')
plt.show()

print("Chart created successfully!")