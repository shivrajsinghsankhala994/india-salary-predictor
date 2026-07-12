import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import io
import os

# ============================================
#   India Salary Predictor - Final Version
# ============================================

st.set_page_config(
    page_title="India Salary Predictor",
    page_icon="💰",
    layout="wide"
)

# ============================================
# Custom CSS
# ============================================
st.markdown("""
<style>
    .suggestion-card {
        padding: 15px;
        border-radius: 10px;
        margin: 5px;
        height: 150px;
    }
    .share-btn {
        background: #1D9E75;
        color: white;
        padding: 8px 16px;
        border-radius: 8px;
        text-decoration: none;
        font-weight: 600;
    }
    .metric-card {
        background: #1E1E1E;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #333;
    }
    div[data-testid="stMetricValue"] {
        font-size: 28px !important;
        color: #1D9E75 !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# Load Model
# ============================================
@st.cache_resource
def load_model_files():
    BASE = os.path.dirname(__file__)
    model       = joblib.load(os.path.join(BASE, 'final_model_v2.pkl'))
    le_role     = joblib.load(os.path.join(BASE, 'encoder_role.pkl'))
    le_city     = joblib.load(os.path.join(BASE, 'encoder_city.pkl'))
    le_industry = joblib.load(os.path.join(BASE, 'encoder_industry.pkl'))
    scaler      = joblib.load(os.path.join(BASE, 'scaler.pkl'))
    return model, le_role, le_city, le_industry, scaler

model, le_role, le_city, le_industry, scaler = load_model_files()

# ============================================
# Helper Data
# ============================================
edu_map = {
    '12th/Diploma': 1, "Bachelor's": 2, 'BCA': 2.5,
    'B.Tech': 3, 'MCA': 3.5, 'MBA': 4, 'M.Tech': 4.2, 'PhD': 5
}

city_avg_salary = {
    'Bangalore': 14.8, 'Mumbai': 14.2, 'Delhi': 13.5, 'Gurgaon': 13.8,
    'Hyderabad': 12.9, 'Pune': 12.1, 'Chennai': 11.8, 'Kolkata': 10.5,
    'Jaipur': 9.8, 'Ahmedabad': 9.8, 'Lucknow': 8.9, 'Chandigarh': 9.2,
    'Indore': 8.6, 'Bhopal': 8.4, 'Nagpur': 8.4
}

role_avg_salary = {
    'Software Developer': 9.8, 'Full Stack Developer': 11.2,
    'Data Analyst': 10.5, 'Data Scientist': 14.8,
    'ML Engineer': 16.2, 'AI Engineer': 17.5,
    'DevOps Engineer': 13.4, 'Cloud Engineer': 14.1,
    'Product Manager': 18.5, 'HR Manager': 8.2,
    'QA Tester': 7.8, 'Business Analyst': 10.8
}

def get_city_tier(city):
    tier1 = ['Bangalore', 'Mumbai', 'Delhi', 'Gurgaon', 'Hyderabad']
    tier2 = ['Pune', 'Chennai', 'Kolkata', 'Jaipur', 'Ahmedabad']
    return 1 if city in tier1 else 2 if city in tier2 else 3

def predict_salary_ml(age, experience, city, role, education, industry):
    city_tier      = get_city_tier(city)
    edu_numeric    = edu_map.get(education, 2)
    try: role_encoded = le_role.transform([role])[0]
    except: role_encoded = 0
    try: city_encoded = le_city.transform([city])[0]
    except: city_encoded = 0
    try: industry_encoded = le_industry.transform([industry])[0]
    except: industry_encoded = 0
    scaled = scaler.transform([[age, experience]])
    input_data = pd.DataFrame([{
        'age_scaled'        : scaled[0][0],
        'experience_scaled' : scaled[0][1],
        'edu_numeric'       : edu_numeric,
        'role_encoded'      : role_encoded,
        'city_encoded'      : city_encoded,
        'city_tier'         : city_tier,
        'industry_encoded'  : industry_encoded
    }])
    return round(model.predict(input_data)[0], 2)

# ============================================
# Header
# ============================================
col_title, col_share = st.columns([4, 1])

with col_title:
    st.title("💰 India Salary Predictor")
    st.write("ML-powered · 89% Accuracy · 5000 Indian employee records")

with col_share:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <a href="https://india-salary-predictor.onrender.com" target="_blank"
    style="background:#1D9E75;color:white;padding:10px 18px;
    border-radius:8px;text-decoration:none;font-weight:600;font-size:14px">
    🔗 Share App
    </a>
    """, unsafe_allow_html=True)

st.markdown("---")

# ============================================
# Sidebar
# ============================================
st.sidebar.header("📋 Your Details")
st.sidebar.markdown("---")

age = st.sidebar.slider("🎂 Age", 22, 58, 28)
experience = st.sidebar.slider("💼 Years of Experience", 0, 28, 3)
education = st.sidebar.selectbox(
    "🎓 Education",
    ['12th/Diploma', "Bachelor's", 'BCA', 'B.Tech',
     'MCA', 'MBA', 'M.Tech', 'PhD'], index=3
)
city = st.sidebar.selectbox(
    "🏙️ City",
    ['Bangalore', 'Mumbai', 'Delhi', 'Gurgaon', 'Hyderabad',
     'Pune', 'Chennai', 'Kolkata', 'Jaipur', 'Ahmedabad',
     'Lucknow', 'Chandigarh', 'Indore', 'Bhopal', 'Nagpur']
)
role = st.sidebar.selectbox(
    "👨‍💻 Job Role",
    ['Software Developer', 'Full Stack Developer', 'Data Analyst',
     'Data Scientist', 'ML Engineer', 'AI Engineer', 'DevOps Engineer',
     'Cloud Engineer', 'Product Manager', 'HR Manager',
     'QA Tester', 'Business Analyst']
)
industry = st.sidebar.selectbox(
    "🏢 Industry",
    ['IT MNC', 'IT Indian Company', 'Fintech', 'E-commerce',
     'Startup', 'Healthcare', 'Manufacturing', 'Government']
)

st.sidebar.markdown("---")
st.sidebar.button("🔮 Predict My Salary", type="primary")

st.sidebar.markdown("---")
st.sidebar.markdown("**🤖 Model Info**")
st.sidebar.markdown("Algorithm: Gradient Boosting")
st.sidebar.markdown("Accuracy: **89%**")
st.sidebar.markdown("Dataset: **5000 records**")

# ============================================
# Predictions
# ============================================
with st.spinner("Calculating your salary..."):
    salary  = predict_salary_ml(age, experience, city, role, education, industry)

low     = round(salary * 0.85, 2)
high    = round(salary * 1.18, 2)
inhand  = round((salary * 100000 * 0.72) / 12)
city_avg = city_avg_salary.get(city, 10)
role_avg = role_avg_salary.get(role, 10)

# ============================================
# Top Metrics
# ============================================
col1, col2, col3, col4 = st.columns(4)
col1.metric("💰 Expected CTC",   f"₹{salary} LPA",
            delta=f"{round(((salary-city_avg)/city_avg)*100, 1)}% vs city avg")
col2.metric("📉 Min Range",      f"₹{low} LPA")
col3.metric("📈 Max Range",      f"₹{high} LPA")
col4.metric("🏦 In-hand/month",  f"₹{inhand:,}")

st.markdown("---")

# ============================================
# Charts Row 1
# ============================================
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("🎯 Salary Gauge")
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=salary,
        title={'text': "Your Predicted Salary (LPA)", 'font': {'size': 14}},
        delta={'reference': city_avg, 'suffix': ' vs city avg'},
        gauge={
            'axis': {'range': [0, 30]},
            'bar': {'color': "#1D9E75"},
            'steps': [
                {'range': [0, 6],   'color': '#FFE5E5'},
                {'range': [6, 12],  'color': '#FFF3CD'},
                {'range': [12, 20], 'color': '#D4EDDA'},
                {'range': [20, 30], 'color': '#CCE5FF'},
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': city_avg
            }
        }
    ))
    fig_gauge.update_layout(height=300, margin=dict(t=50, b=0))
    st.plotly_chart(fig_gauge, use_container_width=True)
    st.caption(f"🔴 Red line = {city} city average ({city_avg} LPA)")

with chart_col2:
    st.subheader("🏙️ City-wise Avg Salary")
    city_df = pd.DataFrame({
        'City': list(city_avg_salary.keys()),
        'Avg Salary': list(city_avg_salary.values())
    }).sort_values('Avg Salary', ascending=True)
    colors = ['#1D9E75' if c == city else '#B0D9CC' for c in city_df['City']]
    fig_city = go.Figure(go.Bar(
        x=city_df['Avg Salary'], y=city_df['City'],
        orientation='h', marker_color=colors,
        text=city_df['Avg Salary'], textposition='outside'
    ))
    fig_city.update_layout(height=400, margin=dict(t=10, b=10),
                           xaxis_title="Avg Salary (LPA)")
    st.plotly_chart(fig_city, use_container_width=True)

st.markdown("---")

# ============================================
# Charts Row 2
# ============================================
chart_col3, chart_col4 = st.columns(2)

with chart_col3:
    st.subheader("📈 Your Salary Growth Over Time")
    exp_range = list(range(0, 26))
    salary_growth = [predict_salary_ml(age, e, city, role, education, industry)
                     for e in exp_range]
    fig_growth = go.Figure()
    fig_growth.add_trace(go.Scatter(
        x=exp_range, y=salary_growth,
        mode='lines+markers',
        line=dict(color='#1D9E75', width=3),
        marker=dict(size=5),
        fill='tozeroy',
        fillcolor='rgba(29,158,117,0.1)'
    ))
    fig_growth.add_vline(x=experience, line_dash="dash",
                         line_color="red",
                         annotation_text=f"You ({experience} yrs)")
    fig_growth.update_layout(
        height=350,
        xaxis_title="Experience (Years)",
        yaxis_title="Salary (LPA)",
        margin=dict(t=10, b=10)
    )
    st.plotly_chart(fig_growth, use_container_width=True)

with chart_col4:
    st.subheader("👨‍💻 Role-wise Avg Salary")
    role_df = pd.DataFrame({
        'Role': list(role_avg_salary.keys()),
        'Avg Salary': list(role_avg_salary.values())
    }).sort_values('Avg Salary', ascending=True)
    colors_role = ['#1D9E75' if r == role else '#B0D9CC' for r in role_df['Role']]
    fig_role = go.Figure(go.Bar(
        x=role_df['Avg Salary'], y=role_df['Role'],
        orientation='h', marker_color=colors_role,
        text=role_df['Avg Salary'], textposition='outside'
    ))
    fig_role.update_layout(height=400, margin=dict(t=10, b=10),
                           xaxis_title="Avg Salary (LPA)")
    st.plotly_chart(fig_role, use_container_width=True)

st.markdown("---")

# ============================================
# Smart Suggestions
# ============================================
st.header("💡 Smart Suggestions to Boost Your Salary")
st.write("Based on your profile, here are personalized recommendations:")

suggestions = []

if city != 'Bangalore':
    sal_b = predict_salary_ml(age, experience, 'Bangalore', role, education, industry)
    boost = round(((sal_b - salary) / salary) * 100, 1)
    suggestions.append({
        'icon': '🏙️',
        'title': 'Move to Bangalore',
        'impact': f'+{boost}%',
        'detail': f'Get ₹{sal_b} LPA instead of ₹{salary} LPA',
        'color': '#1D9E75'
    })

if role != 'Product Manager':
    sal_pm = predict_salary_ml(age, experience, city, 'Product Manager', education, industry)
    boost = round(((sal_pm - salary) / salary) * 100, 1)
    if boost > 0:
        suggestions.append({
            'icon': '💼',
            'title': 'Switch to Product Manager',
            'impact': f'+{boost}%',
            'detail': f'Earn ₹{sal_pm} LPA — highest paying role',
            'color': '#3498db'
        })

sal_5yr = predict_salary_ml(age, min(experience + 5, 28), city, role, education, industry)
boost_exp = round(((sal_5yr - salary) / salary) * 100, 1)
suggestions.append({
    'icon': '⏰',
    'title': '5 More Years Experience',
    'impact': f'+{boost_exp}%',
    'detail': f'With {experience+5} yrs exp → ₹{sal_5yr} LPA',
    'color': '#9b59b6'
})

if education not in ['PhD', 'M.Tech']:
    sal_mt = predict_salary_ml(age, experience, city, role, 'M.Tech', industry)
    boost_edu = round(((sal_mt - salary) / salary) * 100, 1)
    if boost_edu > 0:
        suggestions.append({
            'icon': '🎓',
            'title': 'Upgrade to M.Tech',
            'impact': f'+{boost_edu}%',
            'detail': f'Higher education → ₹{sal_mt} LPA',
            'color': '#e74c3c'
        })

if industry != 'IT MNC':
    sal_mnc = predict_salary_ml(age, experience, city, role, education, 'IT MNC')
    boost_ind = round(((sal_mnc - salary) / salary) * 100, 1)
    if boost_ind > 0:
        suggestions.append({
            'icon': '🏢',
            'title': 'Switch to IT MNC',
            'impact': f'+{boost_ind}%',
            'detail': f'MNC pays ₹{sal_mnc} LPA vs ₹{salary} LPA',
            'color': '#f39c12'
        })

cols = st.columns(len(suggestions))
for i, sug in enumerate(suggestions):
    with cols[i]:
        st.markdown(f"""
        <div style="background:{sug['color']}15;
             border-left:4px solid {sug['color']};
             border-radius:10px;padding:16px;
             min-height:160px">
            <div style="font-size:28px">{sug['icon']}</div>
            <div style="font-weight:700;font-size:14px;
                 margin:6px 0">{sug['title']}</div>
            <div style="font-size:26px;font-weight:800;
                 color:{sug['color']}">{sug['impact']}</div>
            <div style="font-size:12px;color:gray;
                 margin-top:6px">{sug['detail']}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# ============================================
# Market Comparison
# ============================================
st.header("📊 Market Comparison")

comp_col1, comp_col2 = st.columns(2)

with comp_col1:
    comparison_data = {
        'Metric'       : ['Your Salary', 'City Average',
                          'Role Average', 'India Average'],
        'Salary (LPA)' : [salary, city_avg, role_avg, 10.5],
        'vs You'       : ['-',
                          f"{round(((city_avg-salary)/salary)*100,1)}%",
                          f"{round(((role_avg-salary)/salary)*100,1)}%",
                          f"{round(((10.5-salary)/salary)*100,1)}%"]
    }
    comp_df = pd.DataFrame(comparison_data)
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

with comp_col2:
    fig_comp = go.Figure(go.Bar(
        x=['Your Salary', 'City Average', 'Role Average', 'India Average'],
        y=[salary, city_avg, role_avg, 10.5],
        marker_color=['#1D9E75', '#3498db', '#9b59b6', '#e74c3c'],
        text=[f'₹{v}L' for v in [salary, city_avg, role_avg, 10.5]],
        textposition='outside'
    ))
    fig_comp.update_layout(height=280, margin=dict(t=20, b=10),
                           yaxis_title="Salary (LPA)",
                           showlegend=False)
    st.plotly_chart(fig_comp, use_container_width=True)

st.markdown("---")

# ============================================
# Download Report
# ============================================
st.header("📥 Download Your Salary Report")

report_data = {
    'Field': ['Age', 'Experience', 'Education', 'City', 'Role',
              'Industry', 'Expected CTC', 'Min Range', 'Max Range',
              'In-hand Monthly', 'City Average', 'Role Average',
              'India Average'],
    'Value': [f"{age} years", f"{experience} years", education,
              city, role, industry,
              f"₹{salary} LPA", f"₹{low} LPA", f"₹{high} LPA",
              f"₹{inhand:,}/month",
              f"₹{city_avg} LPA", f"₹{role_avg} LPA",
              "₹10.5 LPA"]
}

report_df = pd.DataFrame(report_data)

csv_buffer = io.StringIO()
report_df.to_csv(csv_buffer, index=False)

col_dl1, col_dl2 = st.columns([1, 3])
with col_dl1:
    st.download_button(
        label="⬇️ Download CSV Report",
        data=csv_buffer.getvalue(),
        file_name=f"salary_report_{city}_{role}.csv",
        mime="text/csv"
    )

st.dataframe(report_df, use_container_width=True, hide_index=True)

st.markdown("---")

# ============================================
# Footer
# ============================================
st.markdown("""
<div style="text-align:center;padding:20px;color:gray">
    <p>🤖 Built with Python · Scikit-learn · Streamlit · Plotly</p>
    <p>📊 Gradient Boosting Model · 89% Accuracy · 5000 Indian Records</p>
    
</div>
""", unsafe_allow_html=True)