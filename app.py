import streamlit as pd_st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os

# ============================================================================
# 🎨 1. APPLICATION & UI SETUP (EXECUTIVE SaaS DESIGN)
# ============================================================================
pd_st.set_page_config(
    page_title="AI-Based Student Performance Advisor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Global Design (SaaS Platform Look)
CUSTOM_CSS = """
<style>
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        color: #1E3A8A !important;
    }
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #0F172A;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1rem;
        color: #64748B;
        margin-bottom: 2rem;
    }
    .card {
        background-color: #FFFFFF;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05), 0 2px 4px -1px rgba(0,0,0,0.03);
        margin-bottom: 1.5rem;
        border: 1px solid #F1F5F9;
    }
    .badge-no-risk {
        background-color: #DCFCE7;
        color: #15803D;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-risk {
        background-color: #FEE2E2;
        color: #B91C1C;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-neutral {
        background-color: #F1F5F9;
        color: #475569;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
</style>
"""
pd_st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ============================================================================
# 🧪 2. ROBUST DATA GENERATOR (FALLBACK FOR PRODUCTION STABILITY)
# ============================================================================
def generate_mock_datasets():
    """Generates synthetic, deterministic structural dataset profiles matching the exact requirements."""
    np.random.seed(42)
    student_ids = [f"STU{str(i).zfill(3)}" for i in range(1, 201)]
    names = [
        "Liam Smith", "Olivia Nova", "Noah Vance", "Emma Roy", "Oliver King", "Ava Reed", "Elijah Ward",
        "Charlotte Fox", "William Ross", "Sophia Bell", "James Cook", "Amelia Diaz", "Benjamin Cruz",
        "Isabella Reyes", "Lucas Ortiz", "Mia Ramos", "Mason Ruiz", "Evelyn Gomez", "Ethan Flores", "Harper Perez"
    ] * 10
    departments = ["Computer Science", "Data Science", "Electrical Eng", "Mechanical Eng"] * 50
    semesters = ["Semester 3", "Semester 5", "Semester 7"] * 67 + ["Semester 3", "Semester 5"]
    subjects = ["Programming", "Mathematics", "Physics", "Chemistry", "Database Systems"]

    # Dataset 1: Subject Analytics
    sub_records = []
    for idx, s_id in enumerate(student_ids):
        for sub in subjects:
            att = float(np.random.randint(65, 100))
            qz = float(np.random.randint(10, 20))
            asg = float(np.random.randint(12, 20))
            mid = float(np.random.randint(15, 30))
            fin = float(np.random.randint(30, 50))
            # artificially degrade specific students to make analytics meaningful
            if idx % 7 == 0:
                att = float(np.random.randint(55, 75))
                fin = float(np.random.randint(15, 35))
            sub_records.append([s_id, names[idx], departments[idx], semesters[idx], sub, att, qz, asg, mid, fin])
    
    df1 = pd.DataFrame(sub_records, columns=[
        "Student_ID", "Student_Name", "Department", "Semester", "Subject", 
        "Attendance", "Quiz_Score", "Assignment_Score", "Midterm_Marks", "Final_Marks"
    ])

    # Dataset 2: ML Risk Analysis
    ml_records = []
    for idx, s_id in enumerate(student_ids):
        att_avg = df1[df1["Student_ID"] == s_id]["Attendance"].mean()
        asg_avg = df1[df1["Student_ID"] == s_id]["Assignment_Score"].mean()
        qz_avg = df1[df1["Student_ID"] == s_id]["Quiz_Score"].mean()
        mid_avg = df1[df1["Student_ID"] == s_id]["Midterm_Marks"].mean()
        fin_avg = df1[df1["Student_ID"] == s_id]["Final_Marks"].mean()
        
        internal_marks = float(mid_avg + qz_avg + asg_avg)
        prev_sem = float(np.random.randint(60, 95))
        recent_test = float(fin_avg * 2) # normalized approximate mapping
        missed_asg = int(np.random.randint(0, 4) if att_avg > 75 else np.random.randint(2, 6))
        
        trend = "Improving" if recent_test > prev_sem else ("Declining" if recent_test < (prev_sem - 10) else "Stable")
        hours = int(np.random.randint(10, 25) if att_avg > 75 else np.random.randint(4, 12))
        sub_rate = float(np.clip((100 - (missed_asg * 10)), 50, 100))
        participation = float(np.random.randint(60, 100) if att_avg > 80 else np.random.randint(40, 70))
        
        # Risk logical determination rules
        risk = 0
        if att_avg < 75 or internal_marks < 35 or missed_asg >= 3 or trend == "Declining":
            risk = 1
            
        ml_records.append([
            s_id, names[idx], att_avg, asg_avg, qz_avg, internal_marks, prev_sem,
            recent_test, missed_asg, trend, hours, sub_rate, participation, risk
        ])
        
    df2 = pd.DataFrame(ml_records, columns=[
        "Student_ID", "Student_Name", "Attendance", "Assignment_Avg", "Quiz_Avg", "Internal_Marks",
        "Previous_Sem_Avg", "Recent_Test_Avg", "Missed_Assessments", "Performance_Trend",
        "Study_Hours_Per_Week", "Submission_Rate", "Class_Participation", "Academic_Risk"
    ])
    
    return df1, df2

# ============================================================================
# 💾 3. DATA LOADING AND CACHING LAYER
# ============================================================================
@pd_st.cache_data
def load_and_prepare_data():
    file1_path = "data/Subject_Analytics_Dataset_200_Students.xlsx"
    file2_path = "data/ML_Risk_Dataset_200_Students.xlsx"
    
    if os.path.exists(file1_path) and os.path.exists(file2_path):
        try:
            df_subj = pd.read_excel(file1_path)
            df_risk = pd.read_excel(file2_path)
            return df_subj, df_risk
        except Exception as e:
            pd_st.warning(f"Error reading Excel files: {e}. Generating live runtime sandbox metrics dataset instead.")
            return generate_mock_datasets()
    else:
        return generate_mock_datasets()

df_subject_raw, df_risk_raw = load_and_prepare_data()

# ============================================================================
# 🤖 4. MACHINE LEARNING RISK ENGINE
# ============================================================================
class MLRiskEngine:
    def __init__(self, data):
        self.data = data
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.features = [
            "Attendance", "Assignment_Avg", "Quiz_Avg", "Internal_Marks", 
            "Previous_Sem_Avg", "Recent_Test_Avg", "Missed_Assessments", 
            "Study_Hours_Per_Week", "Submission_Rate", "Class_Participation"
        ]
        self._train_model()

    def _train_model(self):
        X = self.data[self.features].copy()
        y = self.data["Academic_Risk"].copy()
        self.model.fit(X, y)

    def predict_risk(self, student_features_df):
        pred = self.model.predict(student_features_df[self.features])[0]
        prob = self.model.predict_proba(student_features_df[self.features])[0]
        return pred, prob[1]

    def get_feature_importance(self):
        importances = self.model.feature_importances__
        return pd.DataFrame({
            "Feature": [f.replace("_", " ") for f in self.features],
            "Importance": importances
        }).sort_values(by="Importance", ascending=True)

risk_engine = MLRiskEngine(df_risk_raw)

# ============================================================================
# 🧭 5. SIDEBAR FILTER & ACADEMIC GLOBAL SELECTOR
# ============================================================================
pd_st.sidebar.image("https://flaticon.com", width=80)
pd_st.sidebar.markdown("### **Navigation Workspace**")
selected_page = pd_st.sidebar.radio(
    "Go To Workspace:",
    [
        "📊 Executive Overview",
        "📚 Subject Analytics",
        "📈 Performance Trends",
        "🚨 Academic Risk Analysis",
        "🎯 Strengths & Weaknesses",
        "💡 Recommendations",
        "💬 AI Academic Advisor",
        "🧠 Explainable AI (XAI)"
    ]
)

pd_st.sidebar.markdown("---")
pd_st.sidebar.markdown("### **Student Context Filter**")

# Standard Uniform Dropdown Selectors
student_list = df_risk_raw[["Student_ID", "Student_Name"]].drop_duplicates()
student_options = {f"{row['Student_ID']} - {row['Student_Name']}": row['Student_ID'] for _, row in student_list.iterrows()}
selected_student_label = pd_st.sidebar.selectbox("Active Student Profile Selection:", list(student_options.keys()))
active_student_id = student_options[selected_student_label]

# Isolate contextual records for views
student_meta = df_subject_raw[df_subject_raw["Student_ID"] == active_student_id].iloc[0]
student_subjects = df_subject_raw[df_subject_raw["Student_ID"] == active_student_id]
student_risk_profile = df_risk_raw[df_risk_raw["Student_ID"] == active_student_id].iloc[0]

# Pre-calculations for logical rules
student_subjects = student_subjects.copy()
student_subjects["Total_Score"] = student_subjects["Final_Marks"] + student_subjects["Midterm_Marks"] + student_subjects["Assignment_Score"] + student_subjects["Quiz_Score"]
