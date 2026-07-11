import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import os

# ============================================
#   Day 8 - First ML Model (Linear Regression)
# ============================================

# Load ML-ready data
df = pd.read_csv('day6/ml_ready_data.csv')

print("=" * 45)
print(" Day 8 - Building First ML Model")
print("=" * 45)
print(f" Dataset shape : {df.shape}")
print(f" Columns       : {list(df.columns)}")

# ============================================
# STEP 1 - Features (X) aur Target (y) alag karo
# ============================================
print("\n" + "=" * 45)
print(" STEP 1 : Splitting Features and Target")
print("=" * 45)

X = df.drop('salary_lpa', axis=1)
y = df['salary_lpa']

print(f" Features (X) : {list(X.columns)}")
print(f" Target (y)   : salary_lpa")
print(f" X shape      : {X.shape}")
print(f" y shape      : {y.shape}")

# ============================================
# STEP 2 - Train Test Split
# ============================================
print("\n" + "=" * 45)
print(" STEP 2 : Train-Test Split (80-20)")
print("=" * 45)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f" Training data : {X_train.shape[0]} rows")
print(f" Testing data  : {X_test.shape[0]} rows")

# ============================================
# STEP 3 - Model Train karo
# ============================================
print("\n" + "=" * 45)
print(" STEP 3 : Training Linear Regression Model")
print("=" * 45)

model = LinearRegression()
model.fit(X_train, y_train)

print(" Model trained successfully!")

# ============================================
# STEP 4 - Predictions karo
# ============================================
print("\n" + "=" * 45)
print(" STEP 4 : Making Predictions")
print("=" * 45)

y_pred = model.predict(X_test)

print("\n Sample Predictions vs Actual:")
print(f" {'Actual':<10} {'Predicted':<10} {'Difference':<10}")
for i in range(5):
    actual = y_test.iloc[i]
    predicted = y_pred[i]
    diff = abs(actual - predicted)
    print(f" {actual:<10.2f} {predicted:<10.2f} {diff:<10.2f}")

# ============================================
# STEP 5 - Model Evaluation
# ============================================
print("\n" + "=" * 45)
print(" STEP 5 : Model Evaluation")
print("=" * 45)

r2  = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f" R2 Score  : {r2:.4f}  ({r2*100:.1f}% accuracy)")
print(f" MAE       : {mae:.2f} LPA  (average error)")
print(f" RMSE      : {rmse:.2f} LPA")

if r2 > 0.7:
    print("\n Good model! Above 70% accuracy.")
elif r2 > 0.5:
    print("\n Decent model. Can improve with better models.")
else:
    print("\n Model needs improvement.")

# ============================================
# STEP 6 - Feature Importance dekho
# ============================================
print("\n" + "=" * 45)
print(" STEP 6 : Which Feature Matters Most?")
print("=" * 45)

coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Impact': model.coef_
}).sort_values('Impact', ascending=False)

print(coefficients.to_string(index=False))

# ============================================
# STEP 7 - Predicted vs Actual Chart
# ============================================
plt.figure(figsize=(10, 6))

plt.scatter(y_test, y_pred, alpha=0.5, color='#1D9E75', s=30)

# Perfect prediction line
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val],
         color='red', linestyle='--', linewidth=2,
         label='Perfect Prediction')

plt.title(f'Linear Regression: Actual vs Predicted (R2={r2:.2f})',
          fontsize=14, fontweight='bold')
plt.xlabel('Actual Salary (LPA)')
plt.ylabel('Predicted Salary (LPA)')
plt.legend()
plt.tight_layout()

os.makedirs('day8', exist_ok=True)
plt.savefig('day8/predicted_vs_actual.png', dpi=150)
plt.show()

print("\n Chart saved -> day8/predicted_vs_actual.png")

# ============================================
# STEP 8 - Model Save karo
# ============================================
import joblib

joblib.dump(model, 'day8/linear_model.pkl')

print("\n" + "=" * 45)
print(" Day 8 Complete!")
print(f" Model Accuracy : {r2*100:.1f}%")
print(" Model saved    : day8/linear_model.pkl")
print("=" * 45)