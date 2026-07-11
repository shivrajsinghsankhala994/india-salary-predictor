import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import joblib
import os

# ============================================
#   Day 14 - Complete Pipeline + Final Model
# ============================================

print("=" * 45)
print(" Day 14 - Building Complete Pipeline")
print("=" * 45)

# ============================================
# STEP 1 - Raw Clean Data Load karo (Day 3 wala)
# ============================================
df = pd.read_csv('day3/clean_employees.csv')

print(f"\n Raw clean data loaded : {df.shape}")
print(f" Columns : {list(df.columns)}")

# ============================================
# STEP 2 - Feature Engineering (Day 5 wala)
# ============================================
print("\n" + "=" * 45)
print(" STEP 2 : Feature Engineering")
print("=" * 45)

tier1 = ['Bangalore', 'Mumbai', 'Delhi', 'Hyderabad']
tier2 = ['Pune', 'Chennai', 'Jaipur', 'Ahmedabad']

def get_city_tier(city):
    if city in tier1:
        return 1
    elif city in tier2:
        return 2
    else:
        return 3

df['city_tier'] = df['city'].apply(get_city_tier)

edu_map = {
    '12th/Diploma'  : 1,
    "Bachelor's"    : 2,
    'B.Tech'        : 3,
    'MBA/M.Tech'    : 4,
    'PhD'           : 5
}
df['edu_numeric'] = df['education'].map(edu_map)

print(" Feature engineering done — city_tier, edu_numeric added")

# ============================================
# STEP 3 - Encoders Fit karo aur SAVE karo
# ============================================
print("\n" + "=" * 45)
print(" STEP 3 : Fitting and Saving Encoders")
print("=" * 45)

le_role = LabelEncoder()
le_city = LabelEncoder()

df['role_encoded'] = le_role.fit_transform(df['role'])
df['city_encoded'] = le_city.fit_transform(df['city'])

scaler = StandardScaler()
df[['age_scaled', 'experience_scaled']] = scaler.fit_transform(
    df[['age', 'experience']]
)

os.makedirs('day14', exist_ok=True)

# Sab encoders aur scaler save karo — future mein use honge
joblib.dump(le_role, 'day14/encoder_role.pkl')
joblib.dump(le_city, 'day14/encoder_city.pkl')
joblib.dump(scaler, 'day14/scaler.pkl')

print(" Saved : encoder_role.pkl")
print(" Saved : encoder_city.pkl")
print(" Saved : scaler.pkl")

# ============================================
# STEP 4 - Final Model Train karo (best settings)
# ============================================
print("\n" + "=" * 45)
print(" STEP 4 : Training Final Model")
print("=" * 45)

feature_cols = [
    'age_scaled', 'experience_scaled',
    'edu_numeric', 'role_encoded',
    'city_encoded', 'city_tier'
]

X = df[feature_cols]
y = df['salary_lpa']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Day 11 ki best settings use kar rahe hain
final_model = GradientBoostingRegressor(
    n_estimators=150,
    learning_rate=0.1,
    max_depth=4,
    random_state=42
)

final_model.fit(X_train, y_train)
print(" Final model trained!")

y_pred = final_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f" R2 Score : {r2:.4f} ({r2*100:.1f}%)")
print(f" MAE      : {mae:.2f} LPA")

joblib.dump(final_model, 'day14/final_model.pkl')
print(" Saved : final_model.pkl")

# ============================================
# STEP 5 - Predict Function banao (Important!)
# ============================================
print("\n" + "=" * 45)
print(" STEP 5 : Testing the Prediction Function")
print("=" * 45)

def predict_salary(age, experience, city, role, education):
    """
    Naya employee data leke salary predict karta hai.
    Ye function future mein web app mein use hoga.
    """
    # Encoders load karo
    le_role  = joblib.load('day14/encoder_role.pkl')
    le_city  = joblib.load('day14/encoder_city.pkl')
    scaler   = joblib.load('day14/scaler.pkl')
    model    = joblib.load('day14/final_model.pkl')

    # City tier nikalo
    city_tier = get_city_tier(city)

    # Education number nikalo
    edu_numeric = edu_map.get(education, 2)

    # Role aur city encode karo
    try:
        role_encoded = le_role.transform([role])[0]
    except:
        role_encoded = 0

    try:
        city_encoded = le_city.transform([city])[0]
    except:
        city_encoded = 0

    # Age aur experience scale karo
    scaled = scaler.transform([[age, experience]])
    age_scaled = scaled[0][0]
    exp_scaled = scaled[0][1]

    # Final input banao
    input_data = pd.DataFrame([{
        'age_scaled': age_scaled,
        'experience_scaled': exp_scaled,
        'edu_numeric': edu_numeric,
        'role_encoded': role_encoded,
        'city_encoded': city_encoded,
        'city_tier': city_tier
    }])

    # Predict karo
    predicted_salary = model.predict(input_data)[0]
    return round(predicted_salary, 2)

# ============================================
# STEP 6 - Function Test karo (Sample Predictions)
# ============================================
test_cases = [
    {'age': 25, 'experience': 2, 'city': 'Bangalore',
     'role': 'Software Developer', 'education': 'B.Tech'},
    {'age': 30, 'experience': 7, 'city': 'Mumbai',
     'role': 'Data Scientist', 'education': 'MBA/M.Tech'},
    {'age': 35, 'experience': 12, 'city': 'Jaipur',
     'role': 'ML Engineer', 'education': 'PhD'},
]

print("\n Sample Predictions:")
print(f"\n {'Profile':<55} {'Predicted Salary':<15}")
print(f" {'-'*70}")

for case in test_cases:
    salary = predict_salary(**case)
    profile = f"{case['age']}y, {case['experience']}exp, {case['city']}, {case['role']}"
    print(f" {profile:<55} {salary} LPA")

# ============================================
# STEP 7 - Final Summary
# ============================================
print("\n" + "=" * 45)
print(" Day 14 Complete!")
print(" Pipeline files saved in day14/ folder:")
print("   - final_model.pkl")
print("   - encoder_role.pkl")
print("   - encoder_city.pkl")
print("   - scaler.pkl")
print(f" Model Accuracy : {r2*100:.1f}%")
print(" Ready for Day 18 - Web App!")
print("=" * 45)