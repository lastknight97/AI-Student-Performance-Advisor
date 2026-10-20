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
)

# --- Clean Light-Theme UI Custom CSS (Inspired by reference layout style, zero copied text/names) ---
st.markdown("""
    <style>
        /* Light Global Background */
        .stApp {
            background-color: #f4f6fb;
            color: #1e293b;
        }
        
        /* Top Greeting Banner */
        .welcome-banner {
            background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
            padding: 24px;
            border-radius: 16px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.03);
            margin-bottom: 24px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .welcome-banner h2 {
            margin: 0;
            font-size: 24px;
            font-weight: 800;
            color: #0f172a;
        }
        .welcome-banner p {
            margin: 4px 0 0 0;
            font-size: 13px;
            color: #64748b;
        }

        /* Modern Card Container */
        .dash-card {
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            padding: 18px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.02);
            margin-bottom: 16px;
        }
        .dash-card label {
            font-size: 11px;
            text-transform: uppercase;
            color: #64748b;
            font-weight: 700;
            display: block;
        }
        .dash-card h3 {
            margin: 6px 0 0 0;
            font-size: 22px;
            color: #0f172a;
            font-weight: 800;
        }

        /* Profile Details Badges */
        .profile-badge {
            background-color: #f8fafc;
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

    # Feature Engineering
    trend_numeric = df["performance_trend"].map({"Improving/Stable": 1, "Declining": -1}).fillna(0)
    feature_cols = ["Attendance", "Quiz_Score", "Assignment_Score", "Midterm_Marks", "Final_Marks", "missed_assessments"]
    
    X = df[feature_cols].copy()
    X["trend_numeric"] = trend_numeric

    # Binary Target: 0 = No Risk, 1 = Academic Attention Required
    composite_score = (df["Attendance"] * 0.25) + (df["Final_Marks"] * 0.35) + (df["Midterm_Marks"] * 0.25) + (df["Quiz_Score"] * 0.15)
    df["risk_target"] = np.where(composite_score < 58, 1, 0)
    y = df["risk_target"]

    # Train-Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Baseline Model: Logistic Regression
    lr_baseline = LogisticRegression(random_state=42, max_iter=300)
    lr_baseline.fit(X_train, y_train)
    lr_pred = lr_baseline.predict(X_test)
    lr_report = classification_report(y_test, lr_pred, output_dict=True)

    # Primary Model: Random Forest Classifier
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

# --- Top Header Banner ---
st.markdown("""
    <div class="welcome-banner">
        <div>
            <h2>Welcome back, Advisor!</h2>
            <p>Here is an overview of student performance metrics and machine learning analytics.</p>
        </div>
    </div>
""", unsafe_allow_html=True)

# --- Navigation & Student Selector Bar ---
nav_col, sel_col = st.columns([1.2, 1.8])

with nav_col:
    selected_section = st.radio(
        "Navigation",
        ["📊 Student Performance Desk", "🏠 Institutional Overview", "📈 Analytics & ML Verification"],
        index=2
    )

with sel_col:
    student_options = df[["Student_ID", "Student_Name"]].drop_duplicates().sort_values(by="Student_ID")
    student_options["Label"] = student_options["Student_ID"].astype(str) + " - " + student_options["Student_Name"].astype(str)
    selected_student_label = st.selectbox("Select Student Profile", options=student_options["Label"].tolist(), index=0)
    selected_student_id = selected_student_label.split(" - ")[0]

st.markdown("---")

# ==========================================
# SECTION: ANALYTICS & ML VERIFICATION
# ==========================================
if selected_section == "📈 Analytics & ML Verification":
    st.markdown("### 📊 Performance Analytics & Machine Learning Verification")
    
    # Top 3 Metric Cards row (matching reference layout style)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"<div class='dash-card'><label>Total Students</label><h3>{df['Student_ID'].nunique()}</h3></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='dash-card'><label>Average Mark</label><h3>{df['Final_Marks'].mean():.2f}</h3></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='dash-card'><label>Active Departments</label><h3>{df['Department'].nunique()}</h3></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 3-Column Chart Grid
    g1, g2, g3 = st.columns(3)

    with g1:
        st.markdown("##### Students per Department")
        dept_counts = df.groupby("Department")["Student_ID"].nunique().reset_index()
        fig1 = px.bar(
            dept_counts, x="Department", y="Student_ID", color="Department",
            color_discrete_sequence=["#4f46e5", "#6366f1", "#818cf8", "#a5b4fc"]
        )
        fig1.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False, height=280, margin=dict(t=10, b=10))
        st.plotly_chart(fig1, use_container_width=True)

    with g2:
        st.markdown("##### Attendance Risk Breakdown")
        att_cats = np.where(df["Attendance"] >= 75, "Good (≥75%)", "Low (<75%)")
        att_df = pd.Series(att_cats).value_counts().reset_index()
        att_df.columns = ["Status", "Count"]
        fig2 = px.pie(att_df, values="Count", names="Status", hole=0.6, color_discrete_sequence=["#10b981", "#ef4444"])
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=280, margin=dict(t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    with g3:
        st.markdown("##### Marks Sum by Grade Level")
        grade_df = df.groupby("Grade")["Final_Marks"].sum().reset_index()
        grade_order = ["F", "D", "C", "A", "B"]
        grade_df["Grade"] = pd.Categorical(grade_df["Grade"], categories=grade_order, ordered=True)
        grade_df = grade_df.sort_values("Grade")

        fig3 = px.bar(
            grade_df, y="Grade", x="Final_Marks", orientation="h", color="Grade",
            color_discrete_sequence=px.colors.sequential.Purples_r
        )
        fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False, height=280, margin=dict(t=10, b=10))
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")

    st.markdown("### 🌲 scikit-learn Model Evaluation (Slide 6)")
    
    m_col1, m_col2 = st.columns(2)
    with m_col1:
        st.markdown("##### Primary Model: Random Forest Classifier")
        rf_metrics_df = pd.DataFrame({
            "Class": ["No Risk (0)", "Requires Attention (1)"],
            "Precision": [rf_report['0']['precision'], rf_report['1']['precision']],
            "Recall": [rf_report['0']['recall'], rf_report['1']['recall']],
            "F1-Score": [rf_report['0']['f1-score'], rf_report['1']['f1-score']]
        })
        st.dataframe(rf_metrics_df.style.format(precision=2), use_container_width=True, hide_index=True)

    with m_col2:
        st.markdown("##### Baseline Model: Logistic Regression")
        lr_metrics_df = pd.DataFrame({
            "Class": ["No Risk (0)", "Requires Attention (1)"],
            "Precision": [lr_report['0']['precision'], lr_report['1']['precision']],
            "Recall": [lr_report['0']['recall'], lr_report['1']['recall']],
            "F1-Score": [lr_report['0']['f1-score'], lr_report['1']['f1-score']]
        })
        st.dataframe(lr_metrics_df.style.format(precision=2), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### 🔍 Random Forest Feature Importances (Slide 11)")

    feature_importances = pd.DataFrame({
        "Academic Feature": feature_cols,
        "Importance Score": rf_model.feature_importances_
    }).sort_values(by="Importance Score", ascending=True)

    fig_importance = px.bar(
        feature_importances,
        x="Importance Score",
        y="Academic Feature",
        orientation="h",
        color="Importance Score",
        color_continuous_scale="Purples"
    )
    fig_importance.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=350, margin=dict(t=10, b=10))
    st.plotly_chart(fig_importance, use_container_width=True)


# ==========================================
# SECTION: STUDENT PERFORMANCE DESK
# ==========================================
elif selected_section == "📊 Student Performance Desk":
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

    col_left, col_mid, col_right = st.columns([2.5, 2, 2])

    with col_left:
        st.markdown(f"#### 📝 Performance Summary — {student_name}")
        matrix_df = student_df[["Subject", "Quiz_Score", "Assignment_Score", "Midterm_Marks", "Final_Marks"]].copy()
        matrix_df.columns = ["Subject", "Quiz", "Assignment", "Midterm", "Final Marks"]
        st.dataframe(matrix_df, use_container_width=True, hide_index=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 📊 Subject Marks Comparison")
        
        fig_bar = px.bar(
            student_df, y="Subject",
            x=["Quiz_Score", "Assignment_Score", "Midterm_Marks", "Final_Marks"],
            barmode="group", orientation="h",
            color_discrete_sequence=["#c7d2fe", "#a5b4fc", "#818cf8", "#4338ca"]
        )
        fig_bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_mid:
        st.markdown("<div class='dash-card' style='text-align: center;'>", unsafe_allow_html=True)
        st.image(f"https://api.dicebear.com/7.x/avataaars/svg?seed={student_name}", width=110)
        st.markdown(f"<h3 style='margin: 8px 0 0 0; color: #0f172a;'>{student_name}</h3>", unsafe_allow_html=True)
        st.markdown(f"<p style='color: #64748b; font-size: 12px;'>ID: {selected_student_id} | {student_department}</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        b1, b2 = st.columns(2)
        with b1:
            st.markdown(f"<div class='profile-badge'><label>Attendance Rate</label><span>{avg_attendance:.1f}%</span></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='profile-badge'><label>Calculated GPA</label><span>{cgpa} / 10.0</span></div>", unsafe_allow_html=True)
        with b2:
            st.markdown(f"<div class='profile-badge'><label>Pass Rate</label><span>{pass_rate:.1f}%</span></div>", unsafe_allow_html=True)
            risk_color = "#ef4444" if ml_requires_attention else "#10b981"
            risk_text = "Requires Attention" if ml_requires_attention else "Stable"
            st.markdown(f"<div class='profile-badge'><label>ML Risk Status</label><span style='color: {risk_color};'>{risk_text} ({ml_risk_prob:.2f})</span></div>", unsafe_allow_html=True)

    with col_right:
        st.markdown("#### 💡 ML Recommendations")
        for idx, row in student_df.iterrows():
            prob = row["ML_Risk_Probability"]
            subj = row["Subject"]
            if prob >= 0.5:
                st.error(f"⚠️ **{subj}** (Risk: `{prob:.2f}`)\n\nReview recent topics and schedule support.")
            else:
                st.success(f"✅ **{subj}** (Risk: `{prob:.2f}`)\n\nPerformance is stable.")


# ==========================================
# SECTION: INSTITUTIONAL OVERVIEW
# ==========================================
elif selected_section == "🏠 Institutional Overview":
    st.markdown("### Institutional Summary & Performance Metrics")
    
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(f"<div class='dash-card'><label>Total Students</label><h3>{df['Student_ID'].nunique()}</h3></div>", unsafe_allow_html=True)
    k2.markdown("<div class='dash-card'><label>Total Faculty</label><h3>5</h3></div>", unsafe_allow_html=True)
    k3.markdown(f"<div class='dash-card'><label>Subjects Offered</label><h3>{df['Subject'].nunique()}</h3></div>", unsafe_allow_html=True)
    k4.markdown(f"<div class='dash-card'><label>Average Mark</label><h3>{df['Final_Marks'].mean():.2f}</h3></div>", unsafe_allow_html=True)

    st.markdown("---")
    
    c_left, c_right = st.columns(2)
    with c_left:
        st.markdown("#### Average Marks by Department")
        dept_avg = df.groupby("Department")["Final_Marks"].mean().reset_index()
        fig_dept = px.bar(dept_avg, x="Department", y="Final_Marks", color="Department")
        fig_dept.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
        st.plotly_chart(fig_dept, use_container_width=True)

    with c_right:
        st.markdown("#### Risk Distribution Breakdown")
        risk_counts = df["ML_Attention_Predicted"].map({1: "Requires Attention", 0: "Stable"}).value_counts().reset_index()
        risk_counts.columns = ["Status", "Count"]
        fig_donut = px.pie(risk_counts, values="Count", names="Status", hole=0.5, color_discrete_sequence=["#10b981", "#ef4444"])
        fig_donut.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_donut, use_container_width=True)
        
