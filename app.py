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

# --- Creative Ultra-Modern SaaS CSS ---
st.markdown("""
    <style>
        /* Global Background & Typography */
        .stApp {
            background: #090d16 !important;
            color: #f1f5f9 !important;
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }

        /* Top Banner */
        .hero-banner {
            background: linear-gradient(135deg, rgba(30,41,59,0.8) 0%, rgba(15,23,42,0.9) 100%);
            border: 1px solid rgba(255,255,255,0.08);
            backdrop-filter: blur(12px);
            border-radius: 16px;
            padding: 20px 24px;
            margin-bottom: 24px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .hero-title {
            font-size: 22px;
            font-weight: 800;
            background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0;
        }
        .hero-subtitle {
            font-size: 12px;
            color: #94a3b8;
            margin-top: 4px;
        }

        /* Glassmorphism Metric Cards */
        .glass-card {
            background: rgba(18, 24, 38, 0.75);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 14px;
            padding: 18px 20px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
            backdrop-filter: blur(8px);
            transition: all 0.3s ease;
        }
        .glass-card:hover {
            border-color: rgba(99, 102, 241, 0.4);
            transform: translateY(-2px);
        }
        .card-label {
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #64748b;
            font-weight: 700;
        }
        .card-value {
            font-size: 24px;
            font-weight: 800;
            color: #f8fafc;
            margin-top: 4px;
        }

        /* Unified Container Box */
        .container-box {
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 16px;
            padding: 20px;
            margin-bottom: 20px;
        }
        .section-header {
            font-size: 14px;
            font-weight: 700;
            color: #e2e8f0;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-bottom: 16px;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        /* Status Pills */
        .status-pill-green {
            background: rgba(16, 185, 129, 0.15);
            color: #34d399;
            border: 1px solid rgba(16, 185, 129, 0.3);
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 11px;
            font-weight: 700;
        }
        .status-pill-red {
            background: rgba(244, 63, 94, 0.15);
            color: #fb7185;
            border: 1px solid rgba(244, 63, 94, 0.3);
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 11px;
            font-weight: 700;
        }

        /* Clean Table Overrides */
        .stDataFrame {
            border-radius: 12px !important;
            overflow: hidden !important;
        }
    </style>
""", unsafe_allow_html=True)

# --- 9-Step scikit-learn ML Workflow ---
@st.cache_resource
def run_ml_pipeline():
    excel_path = "data/Subject_Analytics_Dataset_200_Students.xlsx"
    
    if not os.path.exists(excel_path):
        st.error(f"⚠️ Dataset not found at '{excel_path}'. Please place the file in 'data/'.")
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

# --- Top Header & Controls ---
st.markdown("""
    <div class="hero-banner">
        <div>
            <div class="hero-title">🎓 AI STUDENT PERFORMANCE DESK</div>
            <div class="hero-subtitle">Machine Learning Academic Risk Prediction & Performance Analytics</div>
        </div>
    </div>
""", unsafe_allow_html=True)

nav_col, sel_col = st.columns([1.6, 1.4])

with nav_col:
    selected_section = st.radio(
        "Dashboard Navigation",
        ["📋 Student Overview", "📊 Analytics & ML Insights", "🏠 Institutional Summary"],
        index=0,
        horizontal=True
    )

with sel_col:
    student_options = df[["Student_ID", "Student_Name"]].drop_duplicates().sort_values(by="Student_ID")
    student_options["Label"] = student_options["Student_ID"].astype(str) + " — " + student_options["Student_Name"].astype(str)
    selected_student_label = st.selectbox("Select Student Profile", options=student_options["Label"].tolist(), index=0)
    selected_student_id = selected_student_label.split(" — ")[0]

st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 16px 0 24px 0;'>", unsafe_allow_html=True)


# ==========================================
# SECTION 1: CREATIVE STUDENT OVERVIEW
# ==========================================
if selected_section == "📋 Student Overview":
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

    # Top KPI Glass Cards
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""
            <div class="glass-card" style="border-left: 4px solid #38bdf8;">
                <div class="card-label">Student ID</div>
                <div class="card-value">{selected_student_id}</div>
                <div style="font-size:12px; color:#94a3b8; margin-top:2px;">{student_name}</div>
            </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
            <div class="glass-card" style="border-left: 4px solid #818cf8;">
                <div class="card-label">Department / Sem</div>
                <div class="card-value">{student_department}</div>
                <div style="font-size:12px; color:#94a3b8; margin-top:2px;">Semester {student_semester}</div>
            </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
            <div class="glass-card" style="border-left: 4px solid #c084fc;">
                <div class="card-label">Calculated GPA</div>
                <div class="card-value">{cgpa} <span style="font-size:14px; color:#64748b;">/ 10</span></div>
                <div style="font-size:12px; color:#94a3b8; margin-top:2px;">Avg Score: {avg_score:.1f}</div>
            </div>
        """, unsafe_allow_html=True)
    with k4:
        status_html = "<span class='status-pill-red'>⚠️ ATTENTION REQUIRED</span>" if ml_requires_attention else "<span class='status-pill-green'>✅ STABLE STANDING</span>"
        st.markdown(f"""
            <div class="glass-card" style="border-left: 4px solid {'#f43f5e' if ml_requires_attention else '#10b981'};">
                <div class="card-label">ML Risk Status</div>
                <div style="margin-top:8px;">{status_html}</div>
                <div style="font-size:12px; color:#94a3b8; margin-top:8px;">Risk Probability: <b>{ml_risk_prob:.2f}</b></div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Conditional AI Advisory Alert (Only shows if flagged)
    flagged_subjects = student_df[student_df["ML_Risk_Probability"] >= 0.5]
    if len(flagged_subjects) > 0:
        st.markdown("<div class='container-box' style='border-left: 4px solid #f43f5e;'>", unsafe_allow_html=True)
        st.markdown("<div class='section-header' style='color:#fb7185;'>⚠️ AI Academic Intervention Advisory</div>", unsafe_allow_html=True)
        for idx, row in flagged_subjects.iterrows():
            st.error(f"Module **{row['Subject']}** flagged with risk probability `{row['ML_Risk_Probability']:.2f}`. Low midterm or quiz trends detected.")
        st.markdown("</div>", unsafe_allow_html=True)

    # Main Grid: Radar Chart + Progression Journey vs Matrix
    row1_left, row1_right = st.columns([1.3, 1.7])

    with row1_left:
        st.markdown("<div class='container-box'>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>🕸️ Subject Competency Radar</div>", unsafe_allow_html=True)
        
        # Plotly Radar / Spider Chart
        categories = student_df["Subject"].tolist()
        fig_radar = go.Figure()

        fig_radar.add_trace(go.Scatterpolar(
            r=student_df["Final_Marks"].tolist(),
            theta=categories,
            fill='toself',
            name='Final Marks',
            line_color='#38bdf8',
            fillcolor='rgba(56, 189, 248, 0.25)'
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=student_df["Midterm_Marks"].tolist(),
            theta=categories,
            fill='toself',
            name='Midterm Marks',
            line_color='#c084fc',
            fillcolor='rgba(192, 132, 252, 0.15)'
        ))

        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], gridcolor='#1e293b'),
                angularaxis=dict(gridcolor='#1e293b', tickfont=dict(color='#f1f5f9', size=11))
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=30, r=30, t=20, b=20),
            height=320,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5, font=dict(color='#f1f5f9'))
        )
        st.plotly_chart(fig_radar, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with row1_right:
        st.markdown("<div class='container-box'>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>📈 Assessment Score Progression Journey</div>", unsafe_allow_html=True)
        
        # Transform data for progression line plot
        prog_df = student_df.melt(
            id_vars=["Subject"], 
            value_vars=["Quiz_Score", "Assignment_Score", "Midterm_Marks", "Final_Marks"],
            var_name="Evaluation Stage", 
            value_name="Score"
        )
        prog_df["Evaluation Stage"] = prog_df["Evaluation Stage"].str.replace("_", " ").str.title()

        fig_prog = px.line(
            prog_df, 
            x="Evaluation Stage", 
            y="Score", 
            color="Subject", 
            markers=True,
            color_discrete_sequence=["#38bdf8", "#818cf8", "#c084fc", "#f43f5e", "#10b981"]
        )
        fig_prog.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#f1f5f9"),
            height=320,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(gridcolor='#1e293b'),
            yaxis=dict(gridcolor='#1e293b', range=[0, 105]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_prog, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Unified Subject Analytics Matrix
    st.markdown("<div class='container-box'>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>📚 Complete Subject Performance Matrix</div>", unsafe_allow_html=True)
    
    matrix_df = student_df[["Subject", "Attendance", "Quiz_Score", "Assignment_Score", "Midterm_Marks", "Final_Marks", "Grade", "ML_Risk_Probability"]].copy()
    matrix_df["ML Risk Status"] = np.where(matrix_df["ML_Risk_Probability"] >= 0.5, "⚠️ Flagged", "✅ Stable")
    matrix_df = matrix_df[["Subject", "Attendance", "Quiz_Score", "Assignment_Score", "Midterm_Marks", "Final_Marks", "Grade", "ML Risk Status"]]
    matrix_df.columns = ["Subject Module", "Attendance %", "Quiz Score", "Assignment", "Midterm", "Final Marks", "Grade", "ML Status"]
    
    st.dataframe(matrix_df, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ==========================================
# SECTION 2: ANALYTICS & ML INSIGHTS
# ==========================================
elif selected_section == "📊 Analytics & ML Insights":
    st.markdown("### 📊 Cohort Performance & Machine Learning Verification")
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"<div class='glass-card'><div class='card-label'>Total Cohort</div><div class='card-value'>{df['Student_ID'].nunique():,}</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='glass-card'><div class='card-label'>Average Final Mark</div><div class='card-value'>{df['Final_Marks'].mean():.1f}</div></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='glass-card'><div class='card-label'>Pass Rate</div><div class='card-value'>{(df['Final_Marks'] >= 50).mean()*100:.1f}%</div></div>", unsafe_allow_html=True)
    with c4:
        st.markdown(f"<div class='glass-card'><div class='card-label'>Total Flagged Students</div><div class='card-value'>{df['ML_Attention_Predicted'].sum()}</div></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    g1, g2 = st.columns([2, 1])
    with g1:
        st.markdown("<div class='container-box'>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>🏛️ Departmental Average Score Benchmarks</div>", unsafe_allow_html=True)
        dept_avg = df.groupby("Department")[["Midterm_Marks", "Final_Marks"]].mean().reset_index()
        fig_dept = px.bar(
            dept_avg, x="Department", y=["Midterm_Marks", "Final_Marks"],
            barmode="group", color_discrete_sequence=["#38bdf8", "#818cf8"]
        )
        fig_dept.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#f1f5f9"), height=280, yaxis=dict(gridcolor='#1e293b'))
        st.plotly_chart(fig_dept, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with g2:
        st.markdown("<div class='container-box'>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>🍩 Attendance Risk Split</div>", unsafe_allow_html=True)
        att_cats = np.where(df["Attendance"] >= 75, "Good (≥75%)", "Low (<75%)")
        att_df = pd.Series(att_cats).value_counts().reset_index()
        att_df.columns = ["Status", "Count"]
        fig_donut = px.pie(att_df, values="Count", names="Status", hole=0.65, color_discrete_sequence=["#10b981", "#f43f5e"])
        fig_donut.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#f1f5f9"), height=280, showlegend=False)
        st.plotly_chart(fig_donut, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    m1, m2 = st.columns(2)
    with m1:
        st.markdown("<div class='container-box'>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>🌲 Random Forest vs Logistic Regression Metrics</div>", unsafe_allow_html=True)
        rf_metrics_df = pd.DataFrame({
            "Class": ["No Risk (0)", "Requires Attention (1)"],
            "Precision": [rf_report['0']['precision'], rf_report['1']['precision']],
            "Recall": [rf_report['0']['recall'], rf_report['1']['recall']],
            "F1-Score": [rf_report['0']['f1-score'], rf_report['1']['f1-score']]
        })
        st.dataframe(rf_metrics_df.style.format(precision=2), use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with m2:
        st.markdown("<div class='container-box'>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>🔍 Random Forest Feature Importance Ranking</div>", unsafe_allow_html=True)
        feature_importances = pd.DataFrame({
            "Academic Feature": feature_cols,
            "Importance Score": rf_model.feature_importances_
        }).sort_values(by="Importance Score", ascending=True)
        
        fig_importance = px.bar(
            feature_importances, x="Importance Score", y="Academic Feature",
            orientation="h", color="Importance Score", color_continuous_scale="Purples"
        )
        fig_importance.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#f1f5f9"), height=195, margin=dict(t=10, b=10, l=10, r=10), showlegend=False, xaxis=dict(gridcolor='#1e293b'))
        st.plotly_chart(fig_importance, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ==========================================
# SECTION 3: INSTITUTIONAL SUMMARY
# ==========================================
elif selected_section == "🏠 Institutional Summary":
    st.markdown("### 🏠 Institutional Summary Dashboard")
    
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"<div class='glass-card'><div class='card-label'>Total Enrolled Students</div><div class='card-value'>{df['Student_ID'].nunique()}</div></div>", unsafe_allow_html=True)
    with k2:
        st.markdown(f"<div class='glass-card'><div class='card-label'>Faculty Strength</div><div class='card-value'>5</div></div>", unsafe_allow_html=True)
    with k3:
        st.markdown(f"<div class='glass-card'><div class='card-label'>Offered Modules</div><div class='card-value'>{df['Subject'].nunique()}</div></div>", unsafe_allow_html=True)
    with k4:
        st.markdown(f"<div class='glass-card'><div class='card-label'>Overall Average Mark</div><div class='card-value'>{df['Final_Marks'].mean():.2f}</div></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    c_left, c_right = st.columns(2)
    with c_left:
        st.markdown("<div class='container-box'>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>📊 Department Performance Breakdown</div>", unsafe_allow_html=True)
        dept_avg = df.groupby("Department")["Final_Marks"].mean().reset_index()
        fig_dept = px.bar(dept_avg, x="Department", y="Final_Marks", color="Department", color_discrete_sequence=["#38bdf8", "#818cf8", "#c084fc", "#f43f5e"])
        fig_dept.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#f1f5f9"), showlegend=False, height=280, yaxis=dict(gridcolor='#1e293b'))
        st.plotly_chart(fig_dept, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c_right:
        st.markdown("<div class='container-box'>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>🍩 Overall Academic Standing Distribution</div>", unsafe_allow_html=True)
        risk_counts = df["ML_Attention_Predicted"].map({1: "Requires Attention", 0: "Stable Standing"}).value_counts().reset_index()
        risk_counts.columns = ["Status", "Count"]
        fig_donut = px.pie(risk_counts, values="Count", names="Status", hole=0.5, color_discrete_sequence=["#10b981", "#f43f5e"])
        fig_donut.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#f1f5f9"), height=280)
        st.plotly_chart(fig_donut, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
