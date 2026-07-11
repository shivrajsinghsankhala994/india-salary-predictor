import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import joblib
import os

# ============================================
#   Day 16 - Retrain on 5000-row Indian Data
# ============================================

print("=" * 45)
print(" Day 16 - Retraining on Bigger Dataset")
print("=" * 45)

df = pd.read_csv('day15/indian_salary_5000.csv')
print(f"\n Raw data loaded : {df.shape}")

# ============================================
# STEP 1 - Data Cleaning
# ============================================
print("\n" + "=" * 45)
print(" STEP 1 : Data Cleaning")
print("=" * 45)

before = df.shape[0]

# Missing values fix karo
edu_mode = df['education'].mode()[0]
df['education'] = df['education'].fillna(edu_mode)

ind_mode = df['industry'].mode()[0]
df['industry'] = df['industry'].fillna(ind_mode)

print(f" Missing values filled")

# Duplicates hatao
df = df.drop_duplicates()
print(f" Duplicates removed : {before - df.shape[0]} rows")

# Outliers hatao (IQR method)
Q1 = df['salary_lpa'].quantile(0.25)
Q3 = df['salary_lpa'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

before_outlier = df.shape[0]
df = df[(df['salary_lpa'] >= lower) & (df['salary_lpa'] <= upper)]
print(f" Outliers removed   : {before_outlier - df.shape[0]} rows")
print(f" Clean data shape   : {df.shape}")

# ============================================
# STEP 2 - Feature Engineering
# ============================================
print("\n" + "=" * 45)
print(" STEP 2 : Feature Engineering")
print("=" * 45)

tier1 = ['Bangalore', 'Mumbai', 'Delhi', 'Gurgaon', 'Hyderabad']
tier2 = ['Pune', 'Chennai', 'Kolkata', 'Jaipur', 'Ahmedabad']

def get_city_tier(city):
    if city in tier1:
        return 1
    elif city in tier2:
        return 2
    else:
        return 3

df['city_tier'] = df['city'].apply(get_city_tier)

edu_map = {
    '12th/Diploma': 1, "Bachelor's": 2, 'BCA': 2.5,
    'B.Tech': 3, 'MCA': 3.5, 'MBA': 4, 'M.Tech': 4.2, 'PhD': 5
}
df['edu_numeric'] = df['education'].map(edu_map)

print(" city_tier and edu_numeric created")

# ============================================
# STEP 3 - Encoding + Scaling
# ============================================
print("\n" + "=" * 45)
print(" STEP 3 : Encoding and Scaling")
print("=" * 45)

le_role = LabelEncoder()
le_city = LabelEncoder()
le_industry = LabelEncoder()

df['role_encoded']     = le_role.fit_transform(df['role'])
df['city_encoded']     = le_city.fit_transform(df['city'])
df['industry_encoded'] = le_industry.fit_transform(df['industry'])

scaler = StandardScaler()
df[['age_scaled', 'experience_scaled']] = scaler.fit_transform(
    df[['age', 'experience']]
)

os.makedirs('day16', exist_ok=True)
joblib.dump(le_role, 'day16/encoder_role.pkl')
joblib.dump(le_city, 'day16/encoder_city.pkl')
joblib.dump(le_industry, 'day16/encoder_industry.pkl')
joblib.dump(scaler, 'day16/scaler.pkl')

print(" All encoders saved")

# ============================================
# STEP 4 - Train Final Model
# ============================================
print("\n" + "=" * 45)
print(" STEP 4 : Training Final Model")
print("=" * 45)

feature_cols = [
    'age_scaled', 'experience_scaled', 'edu_numeric',
    'role_encoded', 'city_encoded', 'city_tier', 'industry_encoded'
]

X = df[feature_cols]
y = df['salary_lpa']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f" Training rows : {X_train.shape[0]}")
print(f" Testing rows  : {X_test.shape[0]}")

model = GradientBoostingRegressor(
    n_estimators=150,
    learning_rate=0.1,
    max_depth=4,
    random_state=42
)

model.fit(X_train, y_train)
print(" Model trained successfully!")

# ============================================
# STEP 5 - Evaluate New Model
# ============================================
print("\n" + "=" * 45)
print(" STEP 5 : New Model Performance")
print("=" * 45)

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f" R2 Score  : {r2:.4f} ({r2*100:.1f}%)")
print(f" MAE       : {mae:.2f} LPA")
print(f" RMSE      : {rmse:.2f} LPA")

# ============================================
# STEP 6 - Compare with Day 14 Model
# ============================================
print("\n" + "=" * 45)
print(" STEP 6 : Comparison with Day 14 Model")
print("=" * 45)

print(f"\n {'Version':<25} {'Dataset Size':<15} {'R2 Score':<12}")
print(f" {'-'*52}")
print(f" {'Day 14 (old model)':<25} {'980 rows':<15} {'~0.87':<12}")
print(f" {'Day 16 (new model)':<25} {str(df.shape[0])+' rows':<15} {r2:<12.4f}")

if r2 > 0.87:
    print(f"\n New model is better with more data!")
else:
    print(f"\n Similar performance — more data didn't change much")

# ============================================
# STEP 7 - Feature Importance Chart
# ============================================
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'],
         color='#1D9E75', edgecolor='white')
plt.title('Feature Importance - New Model (5000 rows)',
          fontsize=14, fontweight='bold')
plt.xlabel('Importance Score')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('day16/feature_importance_v2.png', dpi=150)
plt.show()

print("\n Chart saved -> day16/feature_importance_v2.png")

# ============================================
# STEP 8 - Save Final Model
# ============================================
joblib.dump(model, 'day16/final_model_v2.pkl')

print("\n" + "=" * 45)
print(" Day 16 Complete!")
print(f" Final Accuracy : {r2*100:.1f}%")
print(f" Trained on     : {df.shape[0]} rows")
print(" Model saved    : day16/final_model_v2.pkl")
print("=" * 45)