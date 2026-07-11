import pandas as pd
import os
import joblib

# ============================================
#   Day 17 - Phase 2 Complete Review
# ============================================

print("=" * 50)
print("   PHASE 2 COMPLETE REVIEW")
print("   ML Model Development (Day 8 - Day 16)")
print("=" * 50)

# ============================================
# STEP 1 - Check All ML Files Exist
# ============================================
print("\n" + "=" * 50)
print(" STEP 1 : Checking All Phase 2 Files")
print("=" * 50)

files = {
    'Day 8  - Linear Regression model'   : 'day8/linear_model.pkl',
    'Day 9  - Random Forest model'        : 'day9/best_model.pkl',
    'Day 10 - Model comparison chart'     : 'day10/model_comparison.png',
    'Day 11 - Tuned model'                : 'day11/tuned_best_model.pkl',
    'Day 12 - Diagnostic charts'          : 'day12/diagnostic_charts.png',
    'Day 13 - SHAP explainability'        : 'day13/shap_importance.png',
    'Day 14 - Final pipeline model'       : 'day14/final_model.pkl',
    'Day 15 - 5000-row dataset'           : 'day15/indian_salary_5000.csv',
    'Day 16 - Final v2 model'             : 'day16/final_model_v2.pkl',
}

all_ok = True
for name, path in files.items():
    exists = os.path.exists(path)
    status = 'OK' if exists else 'MISSING'
    print(f"  {status}  {name}")
    if not exists:
        all_ok = False

if all_ok:
    print("\n  All Phase 2 files present!")
else:
    print("\n  Some files missing — check above")

# ============================================
# STEP 2 - Model Journey Summary
# ============================================
print("\n" + "=" * 50)
print(" STEP 2 : Model Evolution Journey")
print("=" * 50)

journey = [
    ("Day 8",  "Linear Regression",   "Baseline",        "~72%"),
    ("Day 9",  "Random Forest",       "Better accuracy", "~84%"),
    ("Day 10", "Gradient Boosting",   "Best of 3",        "~86%"),
    ("Day 11", "Tuned Gradient Boost","Hyperparameter tuned", "~87%"),
    ("Day 14", "Pipeline Model",      "Production-ready", "~87%"),
    ("Day 16", "Final Model v2",      "5000-row dataset", "~89%"),
]

print(f"\n {'Day':<8} {'Model':<22} {'Note':<25} {'Accuracy':<10}")
print(f" {'-'*68}")
for day, model, note, acc in journey:
    print(f" {day:<8} {model:<22} {note:<25} {acc:<10}")

# ============================================
# STEP 3 - Final Model Details
# ============================================
print("\n" + "=" * 50)
print(" STEP 3 : Final Production Model")
print("=" * 50)

try:
    final_model = joblib.load('day16/final_model_v2.pkl')
    print(f"\n  Model Type     : Gradient Boosting Regressor")
    print(f"  Trees          : {final_model.n_estimators}")
    print(f"  Learning Rate  : {final_model.learning_rate}")
    print(f"  Max Depth      : {final_model.max_depth}")
    print(f"  Trained on     : 5000 rows of Indian salary data")
except Exception as e:
    print(f"  Could not load model: {e}")

# ============================================
# STEP 4 - Phase 2 Learning Summary
# ============================================
print("\n" + "=" * 50)
print(" STEP 4 : Phase 2 - What You Learned")
print("=" * 50)

summary = [
    ("Day 8",  "Train-test split, Linear Regression, R2 score"),
    ("Day 9",  "Random Forest, model comparison"),
    ("Day 10", "Gradient Boosting, cross-validation"),
    ("Day 11", "GridSearchCV, hyperparameter tuning"),
    ("Day 12", "Residual analysis, error diagnostics"),
    ("Day 13", "SHAP values, model explainability"),
    ("Day 14", "Complete prediction pipeline, encoders saving"),
    ("Day 15", "Realistic Indian dataset creation"),
    ("Day 16", "Model retraining on bigger data"),
]

for day, learning in summary:
    print(f"  {day}  ->  {learning}")

print("\n" + "=" * 50)
print(" Phase 2 Complete!")
print(" Next : Day 18 - Building the Web App!")
print("=" * 50)