import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import joblib
import os

# ============================================
#   Day 12 - Deep Model Evaluation
# ============================================

df = pd.read_csv('day6/ml_ready_data.csv')

print("=" * 45)
print(" Day 12 - Deep Model Evaluation")
print("=" * 45)

# ============================================
# STEP 1 - Load Tuned Model (Day 11)
# ============================================
X = df.drop('salary_lpa', axis=1)
y = df['salary_lpa']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = joblib.load('day11/tuned_best_model.pkl')
print(" Tuned model loaded from Day 11")

y_pred = model.predict(X_test)

# ============================================
# STEP 2 - All Evaluation Metrics
# ============================================
print("\n" + "=" * 45)
print(" STEP 2 : Complete Metrics Report")
print("=" * 45)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print(f" R2 Score   : {r2:.4f}  (higher is better, max 1.0)")
print(f" MAE        : {mae:.2f} LPA  (average error)")
print(f" MSE        : {mse:.2f}")
print(f" RMSE       : {rmse:.2f} LPA  (penalizes big errors)")
print(f" MAPE       : {mape:.2f}%  (average % error)")

# ============================================
# STEP 3 - Residuals Calculate karo
# ============================================
print("\n" + "=" * 45)
print(" STEP 3 : Residual Analysis")
print("=" * 45)

residuals = y_test.values - y_pred

print(f" Mean residual     : {residuals.mean():.3f}  (should be near 0)")
print(f" Std of residuals  : {residuals.std():.3f}")
print(f" Max overestimate  : {residuals.min():.2f} LPA")
print(f" Max underestimate : {residuals.max():.2f} LPA")

# ============================================
# STEP 4 - Worst Predictions dhundo
# ============================================
print("\n" + "=" * 45)
print(" STEP 4 : Top 5 Worst Predictions")
print("=" * 45)

results_df = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_pred,
    'Error': np.abs(residuals)
}).sort_values('Error', ascending=False)

print(f"\n {'Actual':<10} {'Predicted':<12} {'Error':<10}")
for _, row in results_df.head(5).iterrows():
    print(f" {row['Actual']:<10.2f} {row['Predicted']:<12.2f} {row['Error']:<10.2f}")

# ============================================
# STEP 5 - Prediction Accuracy Buckets
# ============================================
print("\n" + "=" * 45)
print(" STEP 5 : Prediction Accuracy Buckets")
print("=" * 45)

error_pct = np.abs(residuals / y_test.values) * 100

within_5  = (error_pct <= 5).sum()
