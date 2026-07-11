import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import joblib
import os

# ============================================
#   Day 10 - Gradient Boosting + Comparison
# ============================================

df = pd.read_csv('day6/ml_ready_data.csv')

print("=" * 45)
print(" Day 10 - Gradient Boosting Model")
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
# STEP 2 - Train All 3 Models
# ============================================
print("\n" + "=" * 45)
print(" STEP 2 : Training 3 Models")
print("=" * 45)

# Model 1: Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
print(" 1. Linear Regression trained")

# Model 2: Random Forest
rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
print(" 2. Random Forest trained")

# Model 3: Gradient Boosting
gb_model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=4,
    random_state=42
)
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)
print(" 3. Gradient Boosting trained")

# ============================================
# STEP 3 - Evaluate All Models
# ============================================
print("\n" + "=" * 45)
print(" STEP 3 : Model Evaluation")
print("=" * 45)

def evaluate(name, y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {'Model': name, 'R2': r2, 'MAE': mae, 'RMSE': rmse}

results = []
results.append(evaluate('Linear Regression', y_test, lr_pred))
results.append(evaluate('Random Forest', y_test, rf_pred))
results.append(evaluate('Gradient Boosting', y_test, gb_pred))

results_df = pd.DataFrame(results).sort_values('R2', ascending=False)

print(f"\n {'Model':<22} {'R2 Score':<12} {'MAE':<10} {'RMSE':<10}")
print(f" {'-'*54}")
for _, row in results_df.iterrows():
    print(f" {row['Model']:<22} {row['R2']:<12.4f} {row['MAE']:<10.2f} {row['RMSE']:<10.2f}")

best_model_name = results_df.iloc[0]['Model']
best_r2 = results_df.iloc[0]['R2']

print(f"\n WINNER: {best_model_name} with {best_r2*100:.1f}% accuracy!")

# ============================================
# STEP 4 - Cross Validation Check
# ============================================
print("\n" + "=" * 45)
print(" STEP 4 : Cross Validation (5-Fold)")
print("=" * 45)

from sklearn.model_selection import cross_val_score

models_dict = {
    'Linear Regression': lr_model,
    'Random Forest': rf_model,
    'Gradient Boosting': gb_model
}

for name, model in models_dict.items():
    scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    print(f" {name:<22} : {scores.mean():.4f} (+/- {scores.std():.4f})")

# ============================================
# STEP 5 - Comparison Chart
# ============================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Chart 1: R2 Score Comparison
colors = ['#e74c3c', '#3498db', '#1D9E75']
axes[0].bar(results_df['Model'], results_df['R2'], color=colors, edgecolor='white')
axes[0].set_title('Model Comparison - R2 Score', fontweight='bold')
axes[0].set_ylabel('R2 Score')
axes[0].set_ylim(0, 1)
for i, val in enumerate(results_df['R2']):
    axes[0].text(i, val + 0.02, f'{val:.3f}', ha='center', fontweight='bold')

# Chart 2: Best Model Predictions
best_pred = gb_pred if best_model_name == 'Gradient Boosting' else (
    rf_pred if best_model_name == 'Random Forest' else lr_pred
)
axes[1].scatter(y_test, best_pred, alpha=0.5, color='#1D9E75', s=30)
min_val = min(y_test.min(), best_pred.min())
max_val = max(y_test.max(), best_pred.max())
axes[1].plot([min_val, max_val], [min_val, max_val],
             color='red', linestyle='--', linewidth=2)
axes[1].set_title(f'Best Model: {best_model_name}', fontweight='bold')
axes[1].set_xlabel('Actual Salary (LPA)')
axes[1].set_ylabel('Predicted Salary (LPA)')

plt.tight_layout()
os.makedirs('day10', exist_ok=True)
plt.savefig('day10/model_comparison.png', dpi=150)
plt.show()

print("\n Chart saved -> day10/model_comparison.png")

# ============================================
# STEP 6 - Save Best Model
# ============================================
best_model = {
    'Linear Regression': lr_model,
    'Random Forest': rf_model,
    'Gradient Boosting': gb_model
}[best_model_name]

joblib.dump(best_model, 'day10/final_best_model.pkl')

print("\n" + "=" * 45)
print(" Day 10 Complete!")
print(f" Best Model    : {best_model_name}")
print(f" Best Accuracy : {best_r2*100:.1f}%")
print(" Saved         : day10/final_best_model.pkl")
print("=" * 45)