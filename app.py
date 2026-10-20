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

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Student Performance Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Enterprise-Grade UI/UX CSS & Styling ---
st.markdown("""
    <style>
        /* Global App Theme */
        .stApp {
            background-color: #0b0f17 !important;
            color: #f8fafc !important;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }

        /* Standardized Metric Card */
        .metric-card {
            background-color: #131b2e;
            border: 1px solid #1e293b;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            margin-bottom: 20px;
        }
        .metric-card-primary {
            background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 100%);
            border: 1px solid #3b82f6;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.25);
            margin-bottom: 20px;
        }
        .metric-title {
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #94a3b8;
            font-weight: 700;
            margin-bottom: 4px;
        }
        .metric-card-primary .metric-title {
            color: #bfdbfe;
        }
        .metric-value {
            font-size: 26px;
            font-weight: 800;
            color: #f8fafc;
            margin: 0;
        }
        .metric-card-primary .metric-value {
            color: #ffffff;
        }

        /* Standardized Content / Chart Container */
        .content-box {
            background-color: #131b2e;
            border: 1px solid #1e293b;
            border-radius: 12px;
            padding: 24px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            margin-bottom: 24px;
        }
        .box-header {
            font-size: 15px;
            font-weight: 700;
            color: #f8fafc;
            margin-bottom: 16px;
            letter-spacing: 0.02em;
            border-bottom: 1px solid #1e293b;
            padding-bottom: 10px;
        }

        /* Profile Details Badges */
        .profile-badge {
            background-color: #1a2642;
            border: 1px solid #28385e;
            border-radius: 8px;
            padding: 12px 16px;
            margin-bottom: 10px;
        }
        .profile-label {
            font-size: 10px;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-weight: 700;
            display: block;
            margin-bottom: 2px;
        }
        .profile-val {
            font-size: 15px;
            color: #f8fafc;
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

# --- Top Navigation & Selection Bar ---
nav_col, sel_col = st.columns([1.6, 1.4])

with nav_col:
    selected_section = st.radio(
        "Navigation",
        ["📋 Student Performance Desk", "📊 Analytics & ML Insights", "🏠 Institutional Overview"],
        index=0,
        horizontal=True
    )

with sel_col:
    student_options = df[["Student_ID", "Student_Name"]].drop_duplicates().sort_values(by="Student_ID")
    student_options["Label"] = student_options["Student_ID"].astype(str) + " — " + student_options["Student_Name"].astype(str)
    selected_student_label = st.selectbox("Select Student Profile", options=student_options["Label"].tolist(), index=0)
    selected_student_id = selected_student_label.split(" — ")[0]

st.markdown("<hr style='margin: 10px 0 20px 0; border-color: #1e293b;'>", unsafe_allow_html=True)


# ==========================================
# SECTION 1: STUDENT PERFORMANCE DESK
# ==========================================
if selected_section == "📋 Student Performance Desk":
    student_df = df[df["Student_ID"] == selected_student_id]
    student_name = student_df["Student_Name"].iloc[0]
    student_department = student_df["Department"].iloc[0]
    student_semester = student_df["Semester"].iloc[0]
    
    avg_score = student_df["Final_Marks"].mean()
    avg_attendance = student_df["Attendance"].mean()
    pass_rate = (student_df["Final_Marks"] >= 50).mean() * 100
    cgpa = round((avg_score / 10), 2)
    
    ml_risk_prob = student_df["ML_Risk_Probability"].mean()
    ml_requires_attention = ml_risk_prob >= 0.5

    # Standardized 4 KPI Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
            <div class='metric-card-primary'>
                <div class='metric-title'>Selected Student ID</div>
                <div class='metric-value'>{selected_student_id}</div>
            </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-title'>Department / Sem</div>
                <div class='metric-value' style='font-size:20px;'>{student_department} (S{student_semester})</div>
            </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-title'>Calculated CGPA</div>
                <div class='metric-value'>{cgpa} / 10</div>
            </div>
        """, unsafe_allow_html=True)
    with m4:
        risk_color = "#f87171" if ml_requires_attention else "#34d399"
        risk_text = "Attention Required" if ml_requires_attention else "Stable Standing"
        st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-title'>ML Risk Status</div>
                <div class='metric-value' style='font-size:16px; color:{risk_color};'>{risk_text}</div>
            </div>
        """, unsafe_allow_html=True)

    # Standardized Layout Grid (Profile Details vs Subject Analytics Matrix)
    col_left, col_right = st.columns([1.2, 2.8])

    with col_left:
        st.markdown("<div class='content-box'>", unsafe_allow_html=True)
        st.markdown("<div class='box-header'>👤 Student Profile</div>", unsafe_allow_html=True)
        
        st.image(f"https://api.dicebear.com/7.x/avataaars/svg?seed={student_name}", width=90)
        st.markdown(f"<h4 style='margin: 8px 0 2px 0; color: #f8fafc;'>{student_name}</h4>", unsafe_allow_html=True)
        st.markdown(f"<p style='color: #94a3b8; font-size: 12px; margin-bottom: 16px;'>Enrolled Student Record</p>", unsafe_allow_html=True)

        st.markdown(f"""
            <div class='profile-badge'>
                <span class='profile-label'>Attendance Rate</span>
                <span class='profile-val'>{avg_attendance:.1f}%</span>
            </div>
            <div class='profile-badge'>
                <span class='profile-label'>Course Pass Rate</span>
                <span class='profile-val'>{pass_rate:.1f}%</span>
            </div>
            <div class='profile-badge'>
                <span class='profile-label'>Performance Trend</span>
                <span class='profile-val'>{student_df['performance_trend'].iloc[0]}</span>
            </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        st.markdown("<div class='content-box'>", unsafe_allow_html=True)
        st.markdown("<div class='box-header'>📚 Subject Analytics Matrix</div>", unsafe_allow_html=True)
        
        matrix_df = student_df[["Subject", "Attendance", "Quiz_Score", "Assignment_Score", "Midterm_Marks", "Final_Marks", "Grade", "ML_Risk_Probability"]].copy()
        matrix_df["Status"] = np.where(matrix_df["ML_Risk_Probability"] >= 0.5, "⚠️ Flagged", "✅ Stable")
        matrix_df = matrix_df[["Subject", "Attendance", "Quiz_Score", "Assignment_Score", "Midterm_Marks", "Final_Marks", "Grade", "Status"]]
        matrix_df.columns = ["Subject Name", "Attendance %", "Quiz", "Assignment", "Midterm", "Final", "Grade", "ML Status"]
        
        st.dataframe(matrix_df, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Standardized Lower Grid (Subject Breakdown Chart vs AI Recommendation Engine)
    low_1, low_2 = st.columns(2)

    with low_1:
        st.markdown("<div class='content-box'>", unsafe_allow_html=True)
        st.markdown("<div class='box-header'>📊 Subject Score Comparison</div>", unsafe_allow_html=True)
        fig_bar = px.bar(
            student_df, y="Subject",
            x=["Quiz_Score", "Assignment_Score", "Midterm_Marks", "Final_Marks"],
            barmode="group", orientation="h",
            color_discrete_sequence=["#60a5fa", "#3b82f6", "#2563eb", "#1d4ed8"]
        )
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#f8fafc"), height=260, margin=dict(t=10, b=10, l=10, r=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with low_2:
        st.markdown("<div class='content-box'>", unsafe_allow_html=True)
        st.markdown("<div class='box-header'>💡 AI & Machine Learning Advisory Engine</div>", unsafe_allow_html=True)
        
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
    s1, s2, s3, s4, s5 = st.columns(5)
    with s1:
        st.markdown(f"""
            <div class='metric-card-primary'>
                <div class='metric-title'>Total Cohort</div>
                <div class='metric-value'>{df['Student_ID'].nunique():,}</div>
            </div>
        """, unsafe_allow_html=True)
    with s2:
        st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-title'>Average Marks</div>
                <div class='metric-value'>{df['Final_Marks'].mean():.2f}</div>
            </div>
        """, unsafe_allow_html=True)
    with s3:
        st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-title'>Pass Rate</div>
                <div class='metric-value'>{(df['Final_Marks'] >= 50).mean()*100:.1f}%</div>
            </div>
        """, unsafe_allow_html=True)
    with s4:
        st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-title'>Flagged Risk</div>
                <div class='metric-value'>{df['ML_Attention_Predicted'].sum()}</div>
            </div>
        """, unsafe_allow_html=True)
    with s5:
        st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-title'>Departments</div>
                <div class='metric-value'>{df['Department'].nunique()}</div>
            </div>
        """, unsafe_allow_html=True)

    c_left, c_right = st.columns([2, 1])
    with c_left:
        st.markdown("<div class='content-box'>", unsafe_allow_html=True)
        st.markdown("<div class='box-header'>📈 Departmental Score Trends vs Target Benchmarks</div>", unsafe_allow_html=True)
        dept_trend = df.groupby("Department")[["Midterm_Marks", "Final_Marks"]].mean().reset_index()
        fig_trend = px.bar(
            dept_trend, x="Department", y=["Midterm_Marks", "Final_Marks"],
            barmode="group", color_discrete_sequence=["#2563eb", "#93c5fd"]
        )
        fig_trend.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#f8fafc"), height=290, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_trend, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c_right:
        st.markdown("<div class='content-box'>", unsafe_allow_html=True)
        st.markdown("<div class='box-header'>🍩 Attendance Risk Distribution</div>", unsafe_allow_html=True)
        att_cats = np.where(df["Attendance"] >= 75, "Good (≥75%)", "Low (<75%)")
        att_df = pd.Series(att_cats).value_counts().reset_index()
        att_df.columns = ["Status", "Count"]
        fig_donut = px.pie(att_df, values="Count", names="Status", hole=0.65, color_discrete_sequence=["#2563eb", "#ef4444"])
        fig_donut.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#f8fafc"), height=290, margin=dict(t=10, b=10, l=10, r=10), showlegend=False)
        st.plotly_chart(fig_donut, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    m_col1, m_col2 = st.columns(2)
    with m_col1:
        st.markdown("<div class='content-box'>", unsafe_allow_html=True)
        st.markdown("<div class='box-header'>🌲 scikit-learn Model Evaluation (Slide 6)</div>", unsafe_allow_html=True)
        rf_metrics_df = pd.DataFrame({
            "Class": ["No Risk (0)", "Requires Attention (1)"],
            "Precision": [rf_report['0']['precision'], rf_report['1']['precision']],
            "Recall": [rf_report['0']['recall'], rf_report['1']['recall']],
            "F1-Score": [rf_report['0']['f1-score'], rf_report['1']['f1-score']]
        })
        st.dataframe(rf_metrics_df.style.format(precision=2), use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with m_col2:
        st.markdown("<div class='content-box'>", unsafe_allow_html=True)
        st.markdown("<div class='box-header'>🔍 Random Forest Feature Importances (Slide 11)</div>", unsafe_allow_html=True)
        feature_importances = pd.DataFrame({
            "Academic Feature": feature_cols,
            "Importance Score": rf_model.feature_importances_
        }).sort_values(by="Importance Score", ascending=True)
        fig_importance = px.bar(
            feature_importances, x="Importance Score", y="Academic Feature",
            orientation="h", color="Importance Score", color_continuous_scale="Purples"
        )
        fig_importance.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#f8fafc"), height=195, margin=dict(t=10, b=10, l=10, r=10), showlegend=False)
        st.plotly_chart(fig_importance, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ==========================================
# SECTION 3: INSTITUTIONAL OVERVIEW
# ==========================================
elif selected_section == "🏠 Institutional Overview":
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"<div class='metric-card'><div class='metric-title'>Total Students</div><div class='metric-value'>{df['Student_ID'].nunique()}</div></div>", unsafe_allow_html=True)
    with k2:
        st.markdown(f"<div class='metric-card'><div class='metric-title'>Total Faculty</div><div class='metric-value'>5</div></div>", unsafe_allow_html=True)
    with k3:
        st.markdown(f"<div class='metric-card'><div class='metric-title'>Subjects Offered</div><div class='metric-value'>{df['Subject'].nunique()}</div></div>", unsafe_allow_html=True)
    with k4:
        st.markdown(f"<div class='metric-card'><div class='metric-title'>Average Mark</div><div class='metric-value'>{df['Final_Marks'].mean():.2f}</div></div>", unsafe_allow_html=True)

    c_left, c_right = st.columns(2)
    with c_left:
        st.markdown("<div class='content-box'>", unsafe_allow_html=True)
        st.markdown("<div class='box-header'>📊 Average Marks by Department</div>", unsafe_allow_html=True)
        dept_avg = df.groupby("Department")["Final_Marks"].mean().reset_index()
        fig_dept = px.bar(dept_avg, x="Department", y="Final_Marks", color="Department", color_discrete_sequence=["#2563eb", "#3b82f6", "#60a5fa", "#93c5fd"])
        fig_dept.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#f8fafc"), showlegend=False, height=280)
        st.plotly_chart(fig_dept, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c_right:
        st.markdown("<div class='content-box'>", unsafe_allow_html=True)
        st.markdown("<div class='box-header'>🍩 Risk Distribution Breakdown</div>", unsafe_allow_html=True)
        risk_counts = df["ML_Attention_Predicted"].map({1: "Requires Attention", 0: "Stable"}).value_counts().reset_index()
        risk_counts.columns = ["Status", "Count"]
        fig_donut = px.pie(risk_counts, values="Count", names="Status", hole=0.5, color_discrete_sequence=["#34d399", "#f87171"])
        fig_donut.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#f8fafc"), height=280)
        st.plotly_chart(fig_donut, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
