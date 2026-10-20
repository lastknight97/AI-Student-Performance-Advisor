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

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="AI Student Performance Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Exact Dark-Theme Custom CSS matching the screenshot ---
st.markdown("""
    <style>
        /* Base Dark Background */
        .stApp {
            background-color: #0e1117;
            color: #ffffff;
        }
        
        /* Banner Header */
        .main-header {
            background: linear-gradient(135deg, #372ba0 0%, #4f3bcf 50%, #6841ea 100%);
            border-radius: 10px;
            padding: 22px 28px;
            margin-bottom: 25px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.4);
        }
        .main-header h1 {
            color: #ffffff !important;
            font-size: 22px !important;
            font-weight: 800 !important;
            margin: 0 !important;
            padding: 0 !important;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .main-header p {
            color: #c0b7fb !important;
            font-size: 11px !important;
            margin: 4px 0 0 0 !important;
            font-weight: 400;
        }

        /* Metric Cards */
        .metric-card {
            background-color: #161b22;
            border: 1px solid #21262d;
            border-radius: 8px;
            padding: 16px;
            text-align: center;
        }
        .metric-card label {
            color: #8b949e;
            font-size: 11px;
            text-transform: uppercase;
            font-weight: 700;
        }
        .metric-card h2 {
            color: #ffffff;
            margin: 6px 0 0 0;
            font-size: 22px;
            font-weight: 800;
        }

        /* Profile Details Badges */
        .profile-badge {
            background-color: #161b22;
            border: 1px solid #30363d;
            border-radius: 6px;
            padding: 8px 12px;
            margin-bottom: 8px;
        }
        .profile-badge label {
            color: #8b949e;
            font-size: 10px;
            text-transform: uppercase;
            font-weight: 700;
            display: block;
        }
        .profile-badge span {
            color: #ffffff;
            font-size: 14px;
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

    # Binary Classification Target: 0 = No Risk, 1 = Academic Attention Required
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

    # Assign ML Predictions & Probabilities across dataset
    df["ML_Attention_Predicted"] = rf_model.predict(X)
    df["ML_Risk_Probability"] = rf_model.predict_proba(X)[:, 1]

    # Grade Categorization
    def get_grade(marks):
        if marks >= 80: return "A"
        elif marks >= 70: return "B"
        elif marks >= 60: return "C"
        elif marks >= 50: return "D"
        else: return "F"

    df["Grade"] = df["Final_Marks"].apply(get_grade)

    return df, rf_model, rf_report, lr_report, conf_matrix, list(X.columns)

df, rf_model, rf_report, lr_report, conf_matrix, feature_cols = run_ml_pipeline()

# --- Banner Header ---
st.markdown("""
    <div class="main-header">
        <h1>🎓 AI STUDENT PERFORMANCE DASHBOARD</h1>
        <p>Machine Learning Academic Advisory & Performance Analytics System</p>
    </div>
""", unsafe_allow_html=True)

# --- Controls Bar (Navigation & Student Selector) ---
nav_col, sel_col = st.columns([1.2, 1.8])

with nav_col:
    selected_section = st.radio(
        "Navigation",
        ["📊 Student Performance Desk", "🏠 Institutional Overview", "📊 Analytics & ML Verification"],
        index=2
    )

with sel_col:
    student_options = df[["Student_ID", "Student_Name"]].drop_duplicates().sort_values(by="Student_ID")
    student_options["Label"] = student_options["Student_ID"].astype(str) + " - " + student_options["Student_Name"].astype(str)
    selected_student_label = st.selectbox("Select Student Profile", options=student_options["Label"].tolist(), index=0)
    selected_student_id = selected_student_label.split(" - ")[0]

st.markdown("---")


# ==========================================
# SECTION: ANALYTICS & ML VERIFICATION (Screenshot Layout)
# ==========================================
if selected_section == "📊 Analytics & ML Verification":
    st.markdown("### 📊 Interactive Analytics & Machine Learning Workflow")
    
    # 3-Column Chart Row
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.markdown("<h5 style='text-align: center; color: #ffffff;'>Students per Department</h5>", unsafe_allow_html=True)
        dept_counts = df.groupby("Department")["Student_ID"].nunique().reset_index()
        fig1 = px.bar(
            dept_counts, 
            x="Department", 
            y="Student_ID", 
            color="Department",
            color_discrete_sequence=["#2dd4bf", "#f97316", "#60a5fa", "#f472b6"]
        )
        fig1.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ffffff"),
            showlegend=False,
            margin=dict(l=10, r=10, t=10, b=10),
            height=320,
            yaxis=dict(gridcolor="#21262d")
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.markdown("<h5 style='text-align: center; color: #ffffff;'>Attendance Risk Breakdown</h5>", unsafe_allow_html=True)
        att_cats = np.where(df["Attendance"] >= 75, "Good (≥75%)", "Low (<75%)")
        att_df = pd.Series(att_cats).value_counts().reset_index()
        att_df.columns = ["Status", "Count"]
        
        fig2 = px.pie(
            att_df, 
            values="Count", 
            names="Status", 
            hole=0.6,
            color="Status",
            color_discrete_map={"Good (≥75%)": "#ef4444", "Low (<75%)": "#10b981"}
        )
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ffffff"),
            margin=dict(l=10, r=10, t=10, b=10),
            height=320
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col3:
        st.markdown("<h5 style='text-align: center; color: #ffffff;'>Marks Sum by Grade Level</h5>", unsafe_allow_html=True)
        grade_df = df.groupby("Grade")["Final_Marks"].sum().reset_index()
        grade_order = ["F", "D", "C", "A", "B"]
        grade_df["Grade"] = pd.Categorical(grade_df["Grade"], categories=grade_order, ordered=True)
        grade_df = grade_df.sort_values("Grade")

        fig3 = px.bar(
            grade_df, 
            y="Grade", 
            x="Final_Marks", 
            orientation="h",
            color="Grade",
            color_discrete_sequence=["#581c87", "#6b21a8", "#7e22ce", "#9333ea", "#a855f7"]
        )
        fig3.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ffffff"),
            showlegend=False,
            margin=dict(l=10, r=10, t=10, b=10),
            height=320,
            xaxis=dict(gridcolor="#21262d")
        )
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")

    # ML Model Evaluation Section
    st.markdown("### 🌲 scikit-learn Model Evaluation (Slide 6)")
    
    m_col1, m_col2 = st.columns(2)

    with m_col1:
        st.markdown("##### Primary Model: Random Forest Classifier")
        rf_metrics_df = pd.DataFrame({
            "Class": ["0  No Risk (0)", "1  Requires Attention (1)"],
            "Precision": [rf_report['0']['precision'], rf_report['1']['precision']],
            "Recall": [rf_report['0']['recall'], rf_report['1']['recall']],
            "F1-Score": [rf_report['0']['f1-score'], rf_report['1']['f1-score']]
        })
        st.dataframe(rf_metrics_df.style.format(precision=2), use_container_width=True, hide_index=True)

    with m_col2:
        st.markdown("##### Baseline Model: Logistic Regression")
        lr_metrics_df = pd.DataFrame({
            "Class": ["0  No Risk (0)", "1  Requires Attention (1)"],
            "Precision": [lr_report['0']['precision'], lr_report['1']['precision']],
            "Recall": [lr_report['0']['recall'], lr_report['1']['recall']],
            "F1-Score": [lr_report['0']['f1-score'], lr_report['1']['f1-score']]
        })
        st.dataframe(lr_metrics_df.style.format(precision=2), use_container_width=True, hide_index=True)

    st.markdown("---")

    # Feature Importance Section
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
    fig_importance.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ffffff"),
        margin=dict(l=10, r=10, t=10, b=10),
        height=380,
        xaxis=dict(gridcolor="#21262d")
    )
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
        st.markdown("#### 📊 Subject Score Breakdown")
        
        fig_bar = px.bar(
            student_df,
            y="Subject",
            x=["Quiz_Score", "Assignment_Score", "Midterm_Marks", "Final_Marks"],
            barmode="group",
            orientation="h",
            color_discrete_sequence=["#a5b4fc", "#818cf8", "#6366f1", "#4338ca"]
        )
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", 
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ffffff"),
            margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_mid:
        st.markdown("<div class='metric-card' style='text-align: center;'>", unsafe_allow_html=True)
        st.image(f"https://api.dicebear.com/7.x/avataaars/svg?seed={student_name}", width=110)
        st.markdown(f"<h3 style='margin: 8px 0 0 0; color: #ffffff;'>{student_name}</h3>", unsafe_allow_html=True)
        st.markdown(f"<p style='color: #8b949e; font-size: 12px;'>ID: {selected_student_id} | {student_department}</p>", unsafe_allow_html=True)
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
    k1.markdown(f"<div class='metric-card'><label>Total Students</label><h2>{df['Student_ID'].nunique()}</h2></div>", unsafe_allow_html=True)
    k2.markdown("<div class='metric-card'><label>Total Faculty</label><h2>5</h2></div>", unsafe_allow_html=True)
    k3.markdown(f"<div class='metric-card'><label>Subjects Offered</label><h2>{df['Subject'].nunique()}</h2></div>", unsafe_allow_html=True)
    k4.markdown(f"<div class='metric-card'><label>Average Mark</label><h2>{df['Final_Marks'].mean():.2f}</h2></div>", unsafe_allow_html=True)

    st.markdown("---")
    
    c_left, c_right = st.columns(2)
    with c_left:
        st.markdown("#### Average Marks by Department")
        dept_avg = df.groupby("Department")["Final_Marks"].mean().reset_index()
        fig_dept = px.bar(dept_avg, x="Department", y="Final_Marks", color="Department")
        fig_dept.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#ffffff"))
        st.plotly_chart(fig_dept, use_container_width=True)

    with c_right:
        st.markdown("#### Risk Distribution Breakdown")
        risk_counts = df["ML_Attention_Predicted"].map({1: "Requires Attention", 0: "Stable"}).value_counts().reset_index()
        risk_counts.columns = ["Status", "Count"]
        fig_donut = px.pie(risk_counts, values="Count", names="Status", hole=0.5, color_discrete_sequence=["#10b981", "#ef4444"])
        fig_donut.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#ffffff"))
        st.plotly_chart(fig_donut, use_container_width=True)
