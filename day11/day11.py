import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import joblib
import os
import time

# ============================================
#   Day 11 - Hyperparameter Tuning
# ============================================

df = pd.read_csv('day6/ml_ready_data.csv')

print("=" * 45)
print(" Day 11 - Hyperparameter Tuning")
print("=" * 45)
print(f" Dataset shape : {df.shape}")

# ============================================
# STEP 1 - Split Data
# ============================================
X = df.drop('salary_lpa', axis=1)
y = df['salary_lpa']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n Training data : {X_train.shape[0]} rows")
print(f" Testing data  : {X_test.shape[0]} rows")

# ============================================
# STEP 2 - Baseline Model (Before Tuning)
# ============================================
print("\n" + "=" * 45)
print(" STEP 2 : Baseline Model (Default Settings)")
print("=" * 45)

baseline_model = GradientBoostingRegressor(random_state=42)
baseline_model.fit(X_train, y_train)
baseline_pred = baseline_model.predict(X_test)
baseline_r2 = r2_score(y_test, baseline_pred)

print(f" Default settings R2 : {baseline_r2:.4f} ({baseline_r2*100:.1f}%)")

# ============================================
# STEP 3 - Define Parameter Grid
# ============================================
print("\n" + "=" * 45)
print(" STEP 3 : Testing Different Settings")
print("=" * 45)

param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.05, 0.1, 0.15],
    'max_depth': [3, 4, 5],
}

total_combos = 3 * 3 * 3
print(f" Testing {total_combos} different combinations...")
print(" This will take 1-2 minutes, please wait...")

# ============================================
# STEP 4 - GridSearchCV Run karo
# ============================================
start_time = time.time()

grid_search = GridSearchCV(
    estimator=GradientBoostingRegressor(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

end_time = time.time()
time_taken = end_time - start_time

print(f"\n Search complete! Time taken: {time_taken:.1f} seconds")

# ============================================
# STEP 5 - Best Parameters dekho
# ============================================
print("\n" + "=" * 45)
print(" STEP 5 : Best Parameters Found")
print("=" * 45)

best_params = grid_search.best_params_
print(f" Best n_estimators  : {best_params['n_estimators']}")
print(f" Best learning_rate : {best_params['learning_rate']}")
print(f" Best max_depth     : {best_params['max_depth']}")
print(f" Best CV Score      : {grid_search.best_score_:.4f}")

# ============================================
# STEP 6 - Best Model se Predict karo
# ============================================
print("\n" + "=" * 45)
print(" STEP 6 : Tuned Model Performance")
print("=" * 45)

best_model = grid_search.best_estimator_
tuned_pred = best_model.predict(X_test)

tuned_r2 = r2_score(y_test, tuned_pred)
tuned_mae = mean_absolute_error(y_test, tuned_pred)
tuned_rmse = np.sqrt(mean_squared_error(y_test, tuned_pred))

print(f" R2 Score  : {tuned_r2:.4f} ({tuned_r2*100:.1f}%)")
print(f" MAE       : {tuned_mae:.2f} LPA")
print(f" RMSE      : {tuned_rmse:.2f} LPA")

# ============================================
# STEP 7 - Before vs After Comparison
# ============================================
print("\n" + "=" * 45)
print(" STEP 7 : Before vs After Tuning")
print("=" * 45)

improvement = (tuned_r2 - baseline_r2) * 100

print(f"\n {'Metric':<20} {'Before':<12} {'After':<12}")
print(f" {'-'*44}")
print(f" {'R2 Score':<20} {baseline_r2:<12.4f} {tuned_r2:<12.4f}")
print(f" {'Accuracy %':<20} {baseline_r2*100:<12.1f} {tuned_r2*100:<12.1f}")

if improvement > 0:
    print(f"\n Improvement: +{improvement:.2f}% accuracy gained!")
else:
    print(f"\n Default settings were already good.")

# ============================================
# STEP 8 - Chart Banao
# ============================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Chart 1: Before vs After
models_comp = ['Before Tuning', 'After Tuning']
scores_comp = [baseline_r2, tuned_r2]
colors = ['#e74c3c', '#1D9E75']

bars = axes[0].bar(models_comp, scores_comp, color=colors, edgecolor='white')
axes[0].set_title('Before vs After Tuning', fontweight='bold')
axes[0].set_ylabel('R2 Score')
axes[0].set_ylim(0, 1)
for bar, val in zip(bars, scores_comp):
    axes[0].text(bar.get_x() + bar.get_width()/2, val + 0.02,
                 f'{val:.3f}', ha='center', fontweight='bold')

# Chart 2: Tuned Model Predictions
axes[1].scatter(y_test, tuned_pred, alpha=0.5, color='#1D9E75', s=30)
min_val = min(y_test.min(), tuned_pred.min())
max_val = max(y_test.max(), tuned_pred.max())
axes[1].plot([min_val, max_val], [min_val, max_val],
             color='red', linestyle='--', linewidth=2)
axes[1].set_title(f'Tuned Model (R2={tuned_r2:.2f})', fontweight='bold')
axes[1].set_xlabel('Actual Salary (LPA)')
axes[1].set_ylabel('Predicted Salary (LPA)')

plt.tight_layout()
os.makedirs('day11', exist_ok=True)
plt.savefig('day11/tuning_results.png', dpi=150)
plt.show()

print("\n Chart saved -> day11/tuning_results.png")

# ============================================
# STEP 9 - Tuned Model Save karo
# ============================================
joblib.dump(best_model, 'day11/tuned_best_model.pkl')

print("\n" + "=" * 45)
print(" Day 11 Complete!")
print(f" Final Accuracy : {tuned_r2*100:.1f}%")
print(f" Best Settings  : {best_params}")
print(" Saved          : day11/tuned_best_model.pkl")
print("=" * 45)