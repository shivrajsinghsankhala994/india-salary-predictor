import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import joblib
import shap
import os

# ============================================
#   Day 13 - Model Explainability (SHAP)
# ============================================

df = pd.read_csv('day6/ml_ready_data.csv')

print("=" * 45)
print(" Day 13 - Model Explainability")
print("=" * 45)

# ============================================
# STEP 1 - Load Model + Data
# ============================================
X = df.drop('salary_lpa', axis=1)
y = df['salary_lpa']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = joblib.load('day11/tuned_best_model.pkl')
print(" Tuned model loaded from Day 11")
print(f" Features : {list(X.columns)}")

# ============================================
# STEP 2 - SHAP Explainer banao
# ============================================
print("\n" + "=" * 45)
print(" STEP 2 : Building SHAP Explainer")
print("=" * 45)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

print(" SHAP explainer built successfully!")
print(f" SHAP values shape : {shap_values.shape}")

# ============================================
# STEP 3 - Overall Feature Importance
# ============================================
print("\n" + "=" * 45)
print(" STEP 3 : Overall Feature Importance")
print("=" * 45)

mean_abs_shap = np.abs(shap_values).mean(axis=0)

importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Avg_Impact_LPA': mean_abs_shap
}).sort_values('Avg_Impact_LPA', ascending=False)

print(f"\n {'Feature':<22} {'Avg Impact (LPA)':<18}")
print(f" {'-'*40}")
for _, row in importance_df.iterrows():
    print(f" {row['Feature']:<22} {row['Avg_Impact_LPA']:<18.3f}")

# ============================================
# STEP 4 - Explain ek Specific Employee
# ============================================
print("\n" + "=" * 45)
print(" STEP 4 : Explain One Employee's Prediction")
print("=" * 45)

sample_idx = 0
sample = X_test.iloc[sample_idx]
actual_salary = y_test.iloc[sample_idx]
predicted_salary = model.predict(X_test.iloc[[sample_idx]])[0]

# Fix: base_value kabhi-kabhi array hoti hai, isliye single number nikal rahe hain
base_value = explainer.expected_value
if hasattr(base_value, '__len__'):
    base_value = base_value[0]

print(f"\n Employee Details:")
for col in X.columns:
    print(f"   {col:<20} : {sample[col]:.2f}")

print(f"\n Base Salary (average) : {base_value:.2f} LPA")
print(f" Predicted Salary       : {predicted_salary:.2f} LPA")
print(f" Actual Salary          : {actual_salary:.2f} LPA")

print(f"\n How each feature changed the prediction:")
sample_shap = shap_values[sample_idx]
feature_impact = pd.DataFrame({
    'Feature': X.columns,
    'Impact': sample_shap
}).sort_values('Impact', key=abs, ascending=False)

for _, row in feature_impact.iterrows():
    direction = "increased" if row['Impact'] > 0 else "decreased"
    print(f"   {row['Feature']:<20} {direction} salary by {abs(row['Impact']):.2f} LPA")

# ============================================
# STEP 5 - SHAP Summary Chart
# ============================================
print("\n" + "=" * 45)
print(" STEP 5 : Creating SHAP Charts")
print("=" * 45)

os.makedirs('day13', exist_ok=True)

# Chart 1: Bar chart - overall importance
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Avg_Impact_LPA'],
         color='#1D9E75', edgecolor='white')
plt.title('SHAP Feature Importance (Average Impact on Salary)',
          fontsize=14, fontweight='bold')
plt.xlabel('Average Impact on Prediction (LPA)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('day13/shap_importance.png', dpi=150)
plt.show()
print(" Chart 1 saved -> shap_importance.png")

# Chart 2: SHAP Summary Plot (official SHAP visualization)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
