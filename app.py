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

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI Student Performance & Risk Advisory | Google Enterprise AI",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- GOOGLE MATERIAL DESIGN & ENTERPRISE UI STYLING ---
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500;700&family=Roboto:wght@400;500;700&display=swap');

        /* Global Application Theme */
        .stApp {
            background-color: #f8f9fa !important;
            font-family: 'Google Sans', 'Roboto', sans-serif;
            color: #202124 !important;
        }

        /* Hide default Streamlit elements */
        header {visibility: hidden;}
        
        /* Google Cloud / Material Sidebar */
        section[data-testid="stSidebar"] {
            background-color: #ffffff !important;
            border-right: 1px solid #dadce0;
            padding-top: 16px;
        }
        section[data-testid="stSidebar"] .stMarkdown h2 {
            color: #1a73e8 !important;
            font-weight: 700;
            font-size: 18px;
            letter-spacing: -0.5px;
        }

        /* Material Elevation Cards */
        .material-card {
            background-color: #ffffff;
            border: 1px solid #dadce0;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 1px 2px 0 rgba(60,64,67,0.3), 0 1px 3px 1px rgba(60,64,67,0.15);
            margin-bottom: 20px;
            transition: box-shadow 0.3s ease;
        }
        .material-card:hover {
            box-shadow: 0 1px 3px 0 rgba(60,64,67,0.30), 0 4px 8px 3px rgba(60,64,67,0.15);
        }

        /* Google KPI Metrics */
        .kpi-container {
            background: #ffffff;
            border: 1px solid #dadce0;
            border-radius: 12px;
            padding: 18px 20px;
            box-shadow: 0 1px 2px 0 rgba(60,64,67,0.15);
            position: relative;
            overflow: hidden;
        }
        .kpi-container::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 4px;
            height: 100%;
            background-color: #1a73e8;
        }
        .kpi-title {
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            color: #5f6368;
            font-weight: 500;
        }
        .kpi-val {
            font-size: 28px;
            font-weight: 700;
            color: #202124;
            margin-top: 4px;
        }

        /* Material Status Chips */
        .chip-success {
            background-color: #e6f4ea;
            color: #137333;
            padding: 4px 12px;
            border-radius: 16px;
            font-size: 12px;
            font-weight: 500;
            display: inline-flex;
            align-items: center;
            gap: 4px;
        }
        .chip-danger {
            background-color: #fce8e6;
            color: #c5221f;
            padding: 4px 12px;
            border-radius: 16px;
            font-size: 12px;
            font-weight: 500;
            display: inline-flex;
            align-items: center;
            gap: 4px;
        }

        /* Prescriptive AI Action Cards */
        .action-box {
            background-color: #fef7e0;
            border: 1px solid #fadd85;
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 12px;
        }
        .action-box-title {
            font-weight: 600;
            color: #b06000;
            font-size: 14px;
            margin-bottom: 6px;
        }
        .action-step-item {
            font-size: 13px;
            color: #3c4043;
            margin-bottom: 4px;
            display: flex;
            align-items: flex-start;
            gap: 6px;
        }
    </style>
""", unsafe_allow_html=True)

# --- ROBUST ML PIPELINE & DATA INGESTION ---
@st.cache_resource
def load_and_train_ai_model():
    excel_path = "data/Subject_Analytics_Dataset_200_Students.xlsx"
    
    if not os.path.exists(excel_path):
        st.error(f"Critical System Error: Dataset not found at '{excel_path}'. Please verify directory setup.")
        st.stop()

    try:
        df = pd.read_excel(excel_path)
    except Exception as e:
        st.error(f"Critical Error parsing dataset: {e}")
        st.stop()

    # Data Sanitization & Normalization
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.strip()

    # Feature Engineering
    if "performance_trend" not in df.columns:
        df["performance_trend"] = np.where(df["Final_Marks"] >= df["Midterm_Marks"], "Improving/Stable", "Declining")

    if "missed_assessments" not in df.columns:
        df["missed_assessments"] = np.where(df["Attendance"] < 75, 2, 0)

    trend_numeric = df["performance_trend"].map({"Improving/Stable": 1, "Declining": -1}).fillna(0)
    feature_cols = ["Attendance", "Quiz_Score", "Assignment_Score", "Midterm_Marks", "Final_Marks", "missed_assessments"]
    
    X = df[feature_cols].copy()
    X["trend_numeric"] = trend_numeric

    # Risk Target Definition & Supervised ML Training
    composite_score = (df["Attendance"] * 0.25) + (df["Final_Marks"] * 0.35) + (df["Midterm_Marks"] * 0.25) + (df["Quiz_Score"] * 0.15)
    df["risk_target"] = np.where(composite_score < 58, 1, 0)
    y = df["risk_target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Baseline Model (Logistic Regression)
    lr_model = LogisticRegression(random_state=42, max_iter=300)
    lr_model.fit(X_train, y_train)
    lr_report = classification_report(y_test, lr_model.predict(X_test), output_dict=True)

    # Primary Production Model (Random Forest Classifier)
    rf_model = RandomForestClassifier(n_estimators=150, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_report = classification_report(y_test, rf_model.predict(X_test), output_dict=True)
    conf_matrix = confusion_matrix(y_test, rf_model.predict(X_test))

    # Real-time Inference on Full Cohort
    df["ML_Attention_Predicted"] = rf_model.predict(X)
    df["ML_Risk_Probability"] = rf_model.predict_proba(X)[:, 1]

    # Letter Grade Mapping
    def assign_grade(m):
        if m >= 80: return "A"
        elif m >= 70: return "B"
        elif m >= 60: return "C"
        elif m >= 50: return "D"
        else: return "F"

    df["Grade"] = df["Final_Marks"].apply(assign_grade)

    return df, rf_model, rf_report, lr_report, conf_matrix, feature_cols

df, rf_model, rf_report, lr_report, conf_matrix, feature_cols = load_and_train_ai_model()

# --- GOOGLE CLOUD STYLE SIDEBAR ---
with st.sidebar:
    st.markdown("## 🔍 Google AI Engine")
    st.markdown("<p style='font-size: 12px; color: #5f6368;'>Student Risk & Advisory Console</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Global Filters
    st.markdown("##### ⚙️ Cohort Filters")
    selected_dept = st.selectbox("Department Filter", options=["All Departments"] + sorted(df["Department"].unique().tolist()))
    selected_gender = st.selectbox("Gender Filter", options=["All", "Female", "Male"])
    
    st.markdown("---")
    st.markdown("##### 🚀 Navigation Workspace")
    workspace_mode = st.radio(
        "Select Workspace",
        ["🎯 Student 360° & Risk Desk", "📈 Cohort Intelligence", "🤖 ML Ops & Model Explainability"],
        label_visibility="collapsed"
    )
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.info("💡 **Tip:** Select a student profile in the main workspace to generate real-time prescriptive AI recovery roadmaps.")

# Apply Sidebar Filters to Dataset
active_df = df.copy()
if selected_dept != "All Departments":
    active_df = active_df[active_df["Department"] == selected_dept]
if selected_gender != "All":
    active_df = active_df[active_df["Gender"] == selected_gender]


# ==============================================================================
# WORKSPACE 1: STUDENT 360° & RISK DESK (INDIVIDUAL STUDENT-BASED DATA)
# ==============================================================================
if workspace_mode == "🎯 Student 360° & Risk Desk":
    st.markdown("### 🎯 Student 360° Performance & Prescriptive Risk Advisory")
    st.markdown("<p style='color: #5f6368; font-size: 14px;'>Granular academic profiles powered by real-time Random Forest risk scoring.</p>", unsafe_allow_html=True)
    
    # Student Selector Control
    student_options = active_df[["Student_ID", "Student_Name"]].drop_duplicates().sort_values(by="Student_ID")
    student_options["Label"] = student_options["Student_ID"].astype(str) + " — " + student_options["Student_Name"].astype(str)
    
    sel_student_label = st.selectbox("Search & Select Student Profile", options=student_options["Label"].tolist())
    sel_student_id = sel_student_label.split(" — ")[0]
    
    student_data = active_df[active_df["Student_ID"] == sel_student_id]
    s_name = student_data["Student_Name"].iloc[0]
    s_dept = student_data["Department"].iloc[0]
    s_sem = student_data["Semester"].iloc[0] if "Semester" in student_data.columns else "Year 1"
    
    avg_final = student_data["Final_Marks"].mean()
    avg_att = student_data["Attendance"].mean()
    risk_prob = student_data["ML_Risk_Probability"].mean()
    is_at_risk = risk_prob >= 0.5

    # Top KPI Metrics Row
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""
            <div class='kpi-container'>
                <div class='kpi-title'>Student Identity</div>
                <div class='kpi-val' style='font-size: 20px;'>{sel_student_id}</div>
                <div style='font-size: 12px; color: #5f6368; margin-top: 2px;'>{s_name}</div>
            </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
            <div class='kpi-container'>
                <div class='kpi-title'>Department / Semester</div>
                <div class='kpi-val' style='font-size: 18px;'>{s_dept}</div>
                <div style='font-size: 12px; color: #5f6368; margin-top: 2px;'>{s_sem}</div>
            </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
            <div class='kpi-container'>
                <div class='kpi-title'>Cumulative GPA Score</div>
                <div class='kpi-val'>{round(avg_final/10, 1)} <span style='font-size: 14px; color: #5f6368;'>/ 10</span></div>
                <div style='font-size: 12px; color: #5f6368; margin-top: 2px;'>Avg Marks: {avg_final:.1f}</div>
            </div>
        """, unsafe_allow_html=True)
    with k4:
        chip_html = "<span class='chip-danger'>⚠️ Attention Required</span>" if is_at_risk else "<span class='chip-success'>✅ Stable Standing</span>"
        st.markdown(f"""
            <div class='kpi-container'>
                <div class='kpi-title'>ML Risk Assessment</div>
                <div style='margin-top: 8px;'>{chip_html}</div>
                <div style='font-size: 12px; color: #5f6368; margin-top: 6px;'>Risk Probability: <b>{risk_prob:.2f}</b></div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Main Grid: Prescriptive AI Advisory vs Competency Visualization
    col_left, col_right = st.columns([1.3, 1.7])

    with col_left:
        st.markdown("<div class='material-card'>", unsafe_allow_html=True)
        st.markdown("<h4 style='font-size: 16px; font-weight: 600; color: #202124; margin-bottom: 12px;'>🛠️ Prescriptive AI Recovery Roadmap</h4>", unsafe_allow_html=True)
        
        flagged_modules = student_data[student_data["ML_Risk_Probability"] >= 0.5]
        
        if len(flagged_modules) > 0:
            for _, row in flagged_modules.iterrows():
                subj = row["Subject"]
                q_score = row["Quiz_Score"]
                m_score = row["Midterm_Marks"]
                att_rate = row["Attendance"]
                
                steps = []
                if q_score < 60:
                    steps.append(f"Quiz competency is low ({q_score}/100). Mandate weekly formative quiz modules.")
                if m_score < 60:
                    steps.append(f"Midterm mark is trailing ({m_score}/100). Schedule mandatory 1-on-1 faculty tutoring.")
                if att_rate < 75:
                    steps.append(f"Attendance is critical ({att_rate:.1f}%). Enforce attendance intervention protocol.")
                steps.append("Complete assigned remedial problem sets prior to final examinations.")
                
                steps_rendered = "".join([f"<div class='action-step-item'>🔹 {step}</div>" for step in steps])
                
                st.markdown(f"""
                    <div class='action-box'>
                        <div class='action-box-title'>⚠️ High Risk Flag: {subj} (Index: {row['ML_Risk_Probability']:.2f})</div>
                        {steps_rendered}
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div style='background-color: #e6f4ea; border: 1px solid #ceead6; border-radius: 8px; padding: 16px;'>
                    <div style='font-weight: 600; color: #137333; font-size: 14px;'>✅ Optimal Academic Standing</div>
                    <div style='font-size: 13px; color: #3c4043; margin-top: 4px;'>All enrolled modules are performing above risk thresholds. No active interventions required.</div>
                </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        st.markdown("<div class='material-card'>", unsafe_allow_html=True)
        st.markdown("<h4 style='font-size: 16px; font-weight: 600; color: #202124; margin-bottom: 12px;'>🕸️ Multi-Subject Competency Balance</h4>", unsafe_allow_html=True)
        
        # Plotly Radar Chart for Student Subject Performance
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=student_data["Final_Marks"].tolist(),
            theta=student_data["Subject"].tolist(),
            fill='toself',
            name='Final Marks',
            line_color='#1a73e8',
            fillcolor='rgba(26, 115, 232, 0.2)'
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=student_data["Midterm_Marks"].tolist(),
            theta=student_data["Subject"].tolist(),
            fill='toself',
            name='Midterm Marks',
            line_color='#9333ea',
            fillcolor='rgba(147, 51, 234, 0.1)'
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], gridcolor='#dadce0'),
                angularaxis=dict(gridcolor='#dadce0', tickfont=dict(size=11))
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=280,
            margin=dict(t=20, b=20, l=20, r=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_radar, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Detailed Student Subject Matrix Table
    st.markdown("<div class='material-card'>", unsafe_allow_html=True)
    st.markdown("<h4 style='font-size: 16px; font-weight: 600; color: #202124; margin-bottom: 12px;'>📚 Comprehensive Subject Analytics Matrix</h4>", unsafe_allow_html=True)
    
    display_matrix = student_data[["Subject", "Attendance", "Quiz_Score", "Assignment_Score", "Midterm_Marks", "Final_Marks", "Grade", "ML_Risk_Probability"]].copy()
    display_matrix["Status"] = np.where(display_matrix["ML_Risk_Probability"] >= 0.5, "⚠️ Flagged", "✅ Stable")
    display_matrix = display_matrix[["Subject", "Attendance", "Quiz_Score", "Assignment_Score", "Midterm_Marks", "Final_Marks", "Grade", "Status"]]
    display_matrix.columns = ["Module Name", "Attendance %", "Quiz", "Assignment", "Midterm", "Final Mark", "Grade", "ML Status"]
    
    st.dataframe(display_matrix, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ==============================================================================
# WORKSPACE 2: COHORT INTELLIGENCE (PROBLEM STATEMENT ANALYTICS)
# ==============================================================================
elif workspace_mode == "📈 Cohort Intelligence":
    st.markdown("### 📈 Institutional Cohort Intelligence & Risk Analytics")
    st.markdown("<p style='color: #5f6368; font-size: 14px;'>Macro-level performance indicators and departmental distributions.</p>", unsafe_allow_html=True)
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"<div class='kpi-container'><div class='kpi-title'>Filtered Cohort</div><div class='kpi-val'>{active_df['Student_ID'].nunique()}</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='kpi-container'><div class='kpi-title'>Cohort Avg Mark</div><div class='kpi-val'>{active_df['Final_Marks'].mean():.1f}</div></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='kpi-container'><div class='kpi-title'>Pass Rate</div><div class='kpi-val'>{(active_df['Final_Marks'] >= 50).mean()*100:.1f}%</div></div>", unsafe_allow_html=True)
    with c4:
        st.markdown(f"<div class='kpi-container'><div class='kpi-title'>Flagged At-Risk</div><div class='kpi-val'>{active_df['ML_Attention_Predicted'].sum()}</div></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    row_c1, row_c2 = st.columns(2)
    with row_c1:
        st.markdown("<div class='material-card'>", unsafe_allow_html=True)
        st.markdown("<h4 style='font-size: 16px; font-weight: 600; color: #202124; margin-bottom: 12px;'>🏛️ Departmental Score Benchmarks</h4>", unsafe_allow_html=True)
        dept_grp = active_df.groupby("Department")[["Midterm_Marks", "Final_Marks"]].mean().reset_index()
        fig_dept = px.bar(dept_grp, x="Department", y=["Midterm_Marks", "Final_Marks"], barmode="group", color_discrete_sequence=["#1a73e8", "#9333ea"])
        fig_dept.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=280, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_dept, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with row_c2:
        st.markdown("<div class='material-card'>", unsafe_allow_html=True)
        st.markdown("<h4 style='font-size: 16px; font-weight: 600; color: #202124; margin-bottom: 12px;'>🍩 Attendance Compliance Distribution</h4>", unsafe_allow_html=True)
        att_status = np.where(active_df["Attendance"] >= 75, "Compliant (≥75%)", "Non-Compliant (<75%)")
        att_series = pd.Series(att_status).value_counts().reset_index()
        att_series.columns = ["Category", "Count"]
        fig_pie = px.pie(att_series, values="Count", names="Category", hole=0.6, color_discrete_sequence=["#1a73e8", "#d93025"])
        fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=280, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_pie, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ==============================================================================
# WORKSPACE 3: ML OPS & MODEL EXPLAINABILITY
# ==============================================================================
elif workspace_mode == "🤖 ML Ops & Model Explainability":
    st.markdown("### 🤖 Machine Learning Model Validation & Feature Explainability")
    st.markdown("<p style='color: #5f6368; font-size: 14px;'>Supervised classification metrics and Random Forest feature importance rankings.</p>", unsafe_allow_html=True)
    
    m_col1, m_col2 = st.columns(2)
    
    with m_col1:
        st.markdown("<div class='material-card'>", unsafe_allow_html=True)
        st.markdown("<h4 style='font-size: 16px; font-weight: 600; color: #202124; margin-bottom: 12px;'>🌲 Random Forest Classification Report</h4>", unsafe_allow_html=True)
        rf_metrics_table = pd.DataFrame({
            "Class": ["Stable (0)", "At-Risk (1)"],
            "Precision": [rf_report['0']['precision'], rf_report['1']['precision']],
            "Recall": [rf_report['0']['recall'], rf_report['1']['recall']],
            "F1-Score": [rf_report['0']['f1-score'], rf_report['1']['f1-score']]
        })
        st.dataframe(rf_metrics_table.style.format(precision=2), use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with m_col2:
        st.markdown("<div class='material-card'>", unsafe_allow_html=True)
        st.markdown("<h4 style='font-size: 16px; font-weight: 600; color: #202124; margin-bottom: 12px;'>🔍 Feature Importance Weights</h4>", unsafe_allow_html=True)
        feat_imp_df = pd.DataFrame({
            "Feature": feature_cols,
            "Importance": rf_model.feature_importances_
        }).sort_values(by="Importance", ascending=True)
        
        fig_feat = px.bar(feat_imp_df, x="Importance", y="Feature", orientation="h", color_discrete_sequence=["#1a73e8"])
        fig_feat.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=220, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_feat, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
