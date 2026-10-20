import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# --- Page Configuration (Sidebar collapsed by default) ---
st.set_page_config(
    page_title="AI Student Performance Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- UI/UX Sleek Custom CSS ---
st.markdown("""
    <style>
        .stApp {
            background-color: #f8fafc !important;
            color: #0f172a !important;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }
        
        .stat-card {
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            padding: 18px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.02);
            margin-bottom: 16px;
        }
        .stat-card-blue {
            background: linear-gradient(135deg, #0052FF 0%, #1d4ed8 100%);
            color: #ffffff;
            border: none;
            border-radius: 14px;
            padding: 18px;
            box-shadow: 0 6px 16px rgba(0, 82, 255, 0.25);
            margin-bottom: 16px;
        }
        .stat-card label, .stat-card-blue label {
            font-size: 11px;
            text-transform: uppercase;
            font-weight: 700;
            display: block;
        }
        .stat-card label { color: #64748b; }
        .stat-card-blue label { color: #bfdbfe; }
        
        .stat-card h3, .stat-card-blue h3 {
            margin: 6px 0 0 0;
            font-size: 22px;
            font-weight: 800;
        }
        .stat-card h3 { color: #0f172a; }
        .stat-card-blue h3 { color: #ffffff; }

        .chart-card {
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.02);
            margin-bottom: 20px;
        }

        .profile-badge {
            background-color: #f1f5f9;
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            padding: 10px 14px;
            margin-bottom: 8px;
        }
        .profile-badge label {
            font-size: 10px;
            color: #64748b;
            text-transform: uppercase;
            font-weight: 700;
            display: block;
        }
        .profile-badge span {
            font-size: 14px;
            color: #0f172a;
            font-weight: 700;
        }
    </style>
""", unsafe_allow_html=True)

# --- 9-Step scikit-learn ML Workflow ---
@st.cache_resource
def run_ml_pipeline():
    excel_path = "data/Subject_Analytics_Dataset_200_Students.xlsx"
    
    if not os.path.exists(excel_path):
        st.error(f"⚠️ Dataset not found at '{excel_path}'. Please ensure the file is placed inside the 'data' directory.")
        st.stop()

    try:
        df = pd.read_excel(excel_path)
    except Exception as e:
        st.error(f"⚠️ Error reading dataset: {e}")
        st.stop()

    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.strip()

    if "performance_trend" not in df.columns:
        df["performance_trend"] = np.where(df["Final_Marks"] >= df["Midterm_Marks"], "Improving/Stable", "Declining")

    if "missed_assessments" not in df.columns:
        df["missed_assessments"] = np.where(df["Attendance"] < 75, 2, 0)

    trend_numeric = df["performance_trend"].map({"Improving/Stable": 1, "Declining": -1}).fillna(0)
    feature_cols = ["Attendance", "Quiz_Score", "Assignment_Score", "Midterm_Marks", "Final_Marks", "missed_assessments"]
    
    X = df[feature_cols].copy()
    X["trend_numeric"] = trend_numeric

    composite_score = (df["Attendance"] * 0.25) + (df["Final_Marks"] * 0.35) + (df["Midterm_Marks"] * 0.25) + (df["Quiz_Score"] * 0.15)
    df["risk_target"] = np.where(composite_score < 58, 1, 0)
    y = df["risk_target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lr_baseline = LogisticRegression(random_state=42, max_iter=300)
    lr_baseline.fit(X_train, y_train)
    lr_pred = lr_baseline.predict(X_test)
    lr_report = classification_report(y_test, lr_pred, output_dict=True)

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    
    rf_report = classification_report(y_test, rf_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, rf_pred)

    df["ML_Attention_Predicted"] = rf_model.predict(X)
    df["ML_Risk_Probability"] = rf_model.predict_proba(X)[:, 1]

    def get_grade(marks):
        if marks >= 80: return "A"
        elif marks >= 70: return "B"
        elif marks >= 60: return "C"
        elif marks >= 50: return "D"
        else: return "F"

    df["Grade"] = df["Final_Marks"].apply(get_grade)

    return df, rf_model, rf_report, lr_report, conf_matrix, list(X.columns)

df, rf_model, rf_report, lr_report, conf_matrix, feature_cols = run_ml_pipeline()

# --- Top Header Navigation Bar ---
nav_col, sel_col = st.columns([1.5, 1.5])

with nav_col:
    selected_section = st.radio(
        "Navigation",
        ["📋 Student Performance Desk", "📊 Analytics & ML Insights", "🏠 Institutional Overview"],
        index=0,
        horizontal=True
    )

with sel_col:
    student_options = df[["Student_ID", "Student_Name"]].drop_duplicates().sort_values(by="Student_ID")
    student_options["Label"] = student_options["Student_ID"].astype(str) + " - " + student_options["Student_Name"].astype(str)
    selected_student_label = st.selectbox("Select Student Profile", options=student_options["Label"].tolist(), index=0)
    selected_student_id = selected_student_label.split(" - ")[0]

st.markdown("---")


# ==========================================
# SECTION 1: STUDENT PERFORMANCE DESK (DETAILED OVERVIEW)
# ==========================================
if selected_section == "📋 Student Performance Desk":
    student_df = df[df["Student_ID"] == selected_student_id]
    student_name = student_df["Student_Name"].iloc[0]
    student_department = student_df["Department"].iloc[0]
    student_semester = student_df["Semester"].iloc[0]
    student_gender = student_df["Gender"].iloc.get(0, "N/A") if "Gender" in student_df.columns else "N/A"
    
    avg_score = student_df["Final_Marks"].mean()
    avg_attendance = student_df["Attendance"].mean()
    pass_rate = (student_df["Final_Marks"] >= 50).mean() * 100
    cgpa = round((avg_score / 10), 2)
    
    ml_risk_prob = student_df["ML_Risk_Probability"].mean()
    ml_requires_attention = ml_risk_prob >= 0.5

    # Top Student Info Row
    top_c1, top_c2, top_c3, top_c4 = st.columns(4)
    with top_c1:
        st.markdown(f"""
            <div class='stat-card-blue'>
                <label>Student ID & Name</label>
                <h3>{selected_student_id}</h3>
                <p style='margin:2px 0 0 0; font-size:14px; color:#ffffff; font-weight:600;'>{student_name}</p>
            </div>
        """, unsafe_allow_html=True)
    with top_c2:
        st.markdown(f"""
            <div class='stat-card'>
                <label>Department & Semester</label>
                <h3>{student_department}</h3>
                <p style='margin:2px 0 0 0; font-size:13px; color:#64748b; font-weight:600;'>Semester {student_semester}</p>
            </div>
        """, unsafe_allow_html=True)
    with top_c3:
        st.markdown(f"""
            <div class='stat-card'>
                <label>Overall GPA / Score</label>
                <h3>{cgpa} / 10.0</h3>
                <p style='margin:2px 0 0 0; font-size:13px; color:#64748b; font-weight:600;'>Avg Mark: {avg_score:.2f}</p>
            </div>
        """, unsafe_allow_html=True)
    with top_c4:
        risk_color_txt = "#ef4444" if ml_requires_attention else "#10b981"
        risk_label_txt = "Attention Required" if ml_requires_attention else "Stable Standing"
        st.markdown(f"""
            <div class='stat-card'>
                <label>ML Risk Assessment</label>
                <h3 style='color: {risk_color_txt}; font-size:18px;'>{risk_label_txt}</h3>
                <p style='margin:2px 0 0 0; font-size:13px; color:#64748b; font-weight:600;'>Probability: {ml_risk_prob:.2f}</p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Detailed Mid Grid: Avatar & Metrics Breakdown vs Subject Performance Table
    grid_left, grid_right = st.columns([1.2, 2.8])

    with grid_left:
        st.markdown("<div class='chart-card' style='text-align: center;'>", unsafe_allow_html=True)
        st.image(f"https://api.dicebear.com/7.x/avataaars/svg?seed={student_name}", width=110)
        st.markdown(f"<h4 style='margin: 8px 0 2px 0; color: #0f172a;'>{student_name}</h4>", unsafe_allow_html=True)
        st.markdown(f"<p style='color: #64748b; font-size: 12px; margin-bottom: 14px;'>Student Unique File Profile</p>", unsafe_allow_html=True)
        
        st.markdown(f"<div class='profile-badge'><label>Attendance Average</label><span>{avg_attendance:.1f}%</span></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='profile-badge'><label>Course Pass Rate</label><span>{pass_rate:.1f}%</span></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='profile-badge'><label>Enrolled Subjects</label><span>{len(student_df)} Modules</span></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='profile-badge'><label>Performance Trend</label><span>{student_df['performance_trend'].iloc[0]}</span></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with grid_right:
        st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
        st.markdown("##### 📚 Comprehensive Subject Analytics Matrix")
        
        detailed_matrix = student_df[["Subject", "Attendance", "Quiz_Score", "Assignment_Score", "Midterm_Marks", "Final_Marks", "Grade", "ML_Risk_Probability"]].copy()
        detailed_matrix["ML Status"] = np.where(detailed_matrix["ML_Risk_Probability"] >= 0.5, "⚠️ Flagged", "✅ Stable")
        detailed_matrix = detailed_matrix[["Subject", "Attendance", "Quiz_Score", "Assignment_Score", "Midterm_Marks", "Final_Marks", "Grade", "ML Status"]]
        detailed_matrix.columns = ["Subject Name", "Attendance %", "Quiz", "Assignment", "Midterm", "Final", "Grade", "Status"]
        
        st.dataframe(detailed_matrix, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Lower Grid: Comparative Chart & AI Recommendation Engine
    low_c1, low_c2 = st.columns(2)

    with low_c1:
        st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
        st.markdown("##### 📊 Subject Score Breakdown Analysis")
        fig_bar = px.bar(
            student_df, y="Subject",
            x=["Quiz_Score", "Assignment_Score", "Midterm_Marks", "Final_Marks"],
            barmode="group", orientation="h",
            color_discrete_sequence=["#93c5fd", "#60a5fa", "#3b82f6", "#0052FF"]
        )
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=280, margin=dict(t=10, b=10, l=10, r=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with low_c2:
        st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
        st.markdown("##### 💡 AI & Machine Learning Advisory Engine")
        
        for idx, row in student_df.iterrows():
            prob = row["ML_Risk_Probability"]
            subj = row["Subject"]
            if prob >= 0.5:
                st.error(f"⚠️ **{subj}** (Risk Index: `{prob:.2f}`)\n\n*Action Required:* Trailing assessment scores or low attendance recorded. Faculty advisor review recommended.")
            else:
                st.success(f"✅ **{subj}** (Risk Index: `{prob:.2f}`)\n\n*Status:* Performance is stable and aligned with academic excellence benchmarks.")
        
        st.markdown("</div>", unsafe_allow_html=True)


# ==========================================
# SECTION 2: ANALYTICS & ML INSIGHTS
# ==========================================
elif selected_section == "📊 Analytics & ML Insights":
    st.markdown("### Performance & Machine Learning Analytics")
    
    s1, s2, s3, s4, s5 = st.columns(5)
    with s1:
        st.markdown(f"<div class='stat-card-blue'><label>Total Students</label><h3>{df['Student_ID'].nunique():,}</h3></div>", unsafe_allow_html=True)
    with s2:
        st.markdown(f"<div class='stat-card'><label>Average Marks</label><h3>{df['Final_Marks'].mean():.2f}</h3></div>", unsafe_allow_html=True)
    with s3:
        st.markdown(f"<div class='stat-card'><label>Pass Rate</label><h3>{(df['Final_Marks'] >= 50).mean()*100:.1f}%</h3></div>", unsafe_allow_html=True)
    with s4:
        st.markdown(f"<div class='stat-card'><label>Flagged Risk</label><h3>{df['ML_Attention_Predicted'].sum()}</h3></div>", unsafe_allow_html=True)
    with s5:
        st.markdown(f"<div class='stat-card'><label>Departments</label><h3>{df['Department'].nunique()}</h3></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    c_left, c_right = st.columns([2, 1])
    with c_left:
        st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
        st.markdown("##### Departmental Score Trends vs Target Benchmarks")
        dept_trend = df.groupby("Department")[["Midterm_Marks", "Final_Marks"]].mean().reset_index()
        fig_trend = px.bar(
            dept_trend, x="Department", y=["Midterm_Marks", "Final_Marks"],
            barmode="group", color_discrete_sequence=["#0052FF", "#93c5fd"]
        )
        fig_trend.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=300, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_trend, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c_right:
        st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
        st.markdown("##### Attendance Risk Distribution")
        att_cats = np.where(df["Attendance"] >= 75, "Good (≥75%)", "Low (<75%)")
        att_df = pd.Series(att_cats).value_counts().reset_index()
        att_df.columns = ["Status", "Count"]
        fig_donut = px.pie(att_df, values="Count", names="Status", hole=0.65, color_discrete_sequence=["#0052FF", "#ef4444"])
        fig_donut.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=300, margin=dict(t=10, b=10, l=10, r=10), showlegend=False)
        st.plotly_chart(fig_donut, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    m_col1, m_col2 = st.columns(2)
    with m_col1:
        st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
        st.markdown("##### 🌲 scikit-learn Model Evaluation (Slide 6)")
        rf_metrics_df = pd.DataFrame({
            "Class": ["No Risk (0)", "Requires Attention (1)"],
            "Precision": [rf_report['0']['precision'], rf_report['1']['precision']],
            "Recall": [rf_report['0']['recall'], rf_report['1']['recall']],
            "F1-Score": [rf_report['0']['f1-score'], rf_report['1']['f1-score']]
        })
        st.dataframe(rf_metrics_df.style.format(precision=2), use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with m_col2:
        st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
        st.markdown("##### 🔍 Random Forest Feature Importances (Slide 11)")
        feature_importances = pd.DataFrame({
            "Academic Feature": feature_cols,
            "Importance Score": rf_model.feature_importances_
        }).sort_values(by="Importance Score", ascending=True)
        fig_importance = px.bar(
            feature_importances, x="Importance Score", y="Academic Feature",
            orientation="h", color="Importance Score", color_continuous_scale="Blues"
        )
        fig_importance.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=190, margin=dict(t=10, b=10, l=10, r=10), showlegend=False)
        st.plotly_chart(fig_importance, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ==========================================
# SECTION 3: INSTITUTIONAL OVERVIEW
# ==========================================
elif selected_section == "🏠 Institutional Overview":
    st.markdown("### Institutional Summary & Key Metrics")
    
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(f"<div class='stat-card'><label>Total Students</label><h3>{df['Student_ID'].nunique()}</h3></div>", unsafe_allow_html=True)
    k2.markdown(f"<div class='stat-card'><label>Total Faculty</label><h3>5</h3></div>", unsafe_allow_html=True)
    k3.markdown(f"<div class='stat-card'><label>Subjects Offered</label><h3>{df['Subject'].nunique()}</h3></div>", unsafe_allow_html=True)
    k4.markdown(f"<div class='stat-card'><label>Average Mark</label><h3>{df['Final_Marks'].mean():.2f}</h3></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    c_left, c_right = st.columns(2)
    with c_left:
        st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
        st.markdown("##### Average Marks by Department")
        dept_avg = df.groupby("Department")["Final_Marks"].mean().reset_index()
        fig_dept = px.bar(dept_avg, x="Department", y="Final_Marks", color="Department", color_discrete_sequence=["#0052FF", "#3b82f6", "#60a5fa", "#93c5fd"])
        fig_dept.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
        st.plotly_chart(fig_dept, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c_right:
        st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
        st.markdown("##### Risk Distribution Breakdown")
        risk_counts = df["ML_Attention_Predicted"].map({1: "Requires Attention", 0: "Stable"}).value_counts().reset_index()
        risk_counts.columns = ["Status", "Count"]
        fig_donut = px.pie(risk_counts, values="Count", names="Status", hole=0.5, color_discrete_sequence=["#10b981", "#ef4444"])
        fig_donut.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_donut, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
