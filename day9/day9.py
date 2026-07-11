import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import joblib
import os

# ============================================
#   Day 9 - Random Forest Model
# ============================================

df = pd.read_csv('day6/ml_ready_data.csv')

print("=" * 45)
print(" Day 9 - Random Forest Model")
print("=" * 45)
print(f" Dataset shape : {df.shape}")

# ============================================
# STEP 1 - Split Data (same as Day 8)
# ============================================
X = df.drop('salary_lpa', axis=1)
y = df['salary_lpa']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n Training data : {X_train.shape[0]} rows")
print(f" Testing data  : {X_test.shape[0]} rows")

# ============================================
# STEP 2 - Random Forest Model Train karo
# ============================================
print("\n" + "=" * 45)
print(" STEP 2 : Training Random Forest")
print("=" * 45)

rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

rf_model.fit(X_train, y_train)
print(" Random Forest trained successfully!")
print(f" Number of trees : 100")

# ============================================
# STEP 3 - Predictions
# ============================================
rf_pred = rf_model.predict(X_test)

print("\n Sample Predictions vs Actual:")
print(f" {'Actual':<10} {'Predicted':<10} {'Difference':<10}")
for i in range(5):
    actual = y_test.iloc[i]
    predicted = rf_pred[i]
    diff = abs(actual - predicted)
    print(f" {actual:<10.2f} {predicted:<10.2f} {diff:<10.2f}")

# ============================================
# STEP 4 - Random Forest Evaluation
# ============================================
print("\n" + "=" * 45)
print(" STEP 4 : Random Forest Evaluation")
print("=" * 45)

rf_r2   = r2_score(y_test, rf_pred)
rf_mae  = mean_absolute_error(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))

print(f" R2 Score  : {rf_r2:.4f}  ({rf_r2*100:.1f}% accuracy)")
print(f" MAE       : {rf_mae:.2f} LPA")
print(f" RMSE      : {rf_rmse:.2f} LPA")

# ============================================
# STEP 5 - Compare with Linear Regression
# ============================================
print("\n" + "=" * 45)
print(" STEP 5 : Linear Regression vs Random Forest")
print("=" * 45)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_r2 = r2_score(y_test, lr_pred)
lr_mae = mean_absolute_error(y_test, lr_pred)

print(f"\n {'Model':<22} {'R2 Score':<12} {'MAE':<10}")
print(f" {'-'*44}")
print(f" {'Linear Regression':<22} {lr_r2:<12.4f} {lr_mae:<10.2f}")
print(f" {'Random Forest':<22} {rf_r2:<12.4f} {rf_mae:<10.2f}")

if rf_r2 > lr_r2:
    improvement = (rf_r2 - lr_r2) * 100
    print(f"\n Random Forest is better by {improvement:.1f}%!")
else:
    print(f"\n Linear Regression performed better this time.")

# ============================================
# STEP 6 - Feature Importance
# ============================================
print("\n" + "=" * 45)
print(" STEP 6 : Feature Importance (Random Forest)")
print("=" * 45)

importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print(importance_df.to_string(index=False))

# ============================================
# STEP 7 - Charts
# ============================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Chart 1: Predicted vs Actual
axes[0].scatter(y_test, rf_pred, alpha=0.5, color='#1D9E75', s=30)
min_val = min(y_test.min(), rf_pred.min())
max_val = max(y_test.max(), rf_pred.max())
axes[0].plot([min_val, max_val], [min_val, max_val],
             color='red', linestyle='--', linewidth=2)
axes[0].set_title(f'Random Forest: Actual vs Predicted (R2={rf_r2:.2f})')
axes[0].set_xlabel('Actual Salary (LPA)')
axes[0].set_ylabel('Predicted Salary (LPA)')

# Chart 2: Feature Importance
axes[1].barh(importance_df['Feature'], importance_df['Importance'],
             color='#3498db', edgecolor='white')
axes[1].set_title('Feature Importance')
axes[1].set_xlabel('Importance Score')
axes[1].invert_yaxis()

plt.tight_layout()
os.makedirs('day9', exist_ok=True)
plt.savefig('day9/random_forest_results.png', dpi=150)
plt.show()

print("\n Chart saved -> day9/random_forest_results.png")

# ============================================
# STEP 8 - Best Model Save karo
# ============================================
if rf_r2 > lr_r2:
    joblib.dump(rf_model, 'day9/best_model.pkl')
    best_name = 'Random Forest'
    best_r2 = rf_r2
else:
    joblib.dump(lr_model, 'day9/best_model.pkl')
    best_name = 'Linear Regression'
    best_r2 = lr_r2

print("\n" + "=" * 45)
print(" Day 9 Complete!")
print(f" Best Model     : {best_name}")
print(f" Best Accuracy  : {best_r2*100:.1f}%")
print(" Saved          : day9/best_model.pkl")
print("=" * 45)