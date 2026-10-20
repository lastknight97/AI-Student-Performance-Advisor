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

# --- Purple Aesthetic & Modern Dashboard Styling ---
st.markdown("""
    <style>
        /* Global Page Background */
        .main { background-color: #f3f0f7; }
        
        /* Header Banner Styling */
        .app-header {
            background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #4338ca 100%);
            color: #ffffff;
            padding: 18px 24px;
            border-radius: 12px;
            margin-bottom: 24px;
            box-shadow: 0 4px 14px rgba(49, 46, 129, 0.15);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .app-header h1 {
            margin: 0;
            font-size: 22px;
            font-weight: 800;
            color: #ffffff !important;
            letter-spacing: 0.5px;
        }
        .app-header p {
            margin: 2px 0 0 0;
            font-size: 12px;
            color: #c7d2fe !important;
            font-weight: 500;
        }
        
        /* Card Container Styling */
        .card-container {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(110, 68, 255, 0.06);
            border: 1px solid #e0e7ff;
            margin-bottom: 20px;
        }
        
        /* Profile Detail Badges */
        .profile-badge {
            background-color: #f5f3ff;
            border-radius: 8px;
            padding: 10px 14px;
            margin-bottom: 8px;
            border: 1px solid #ddd6fe;
        }
        .profile-badge label {
            font-size: 10px;
            color: #6d28d9;
            text-transform: uppercase;
            font-weight: 800;
            display: block;
            margin-bottom: 2px;
        }
        .profile-badge span {
            font-size: 14px;
            color: #1e1b4b;
            font-weight: 700;
        }
        
        /* Metric Badges */
        .metric-badge {
            background: #ffffff;
            padding: 16px;
            border-radius: 12px;
            border: 1px solid #e0e7ff;
            border-top: 4px solid #6366f1;
            box-shadow: 0 2px 8px rgba(0,0,0,0.04);
            text-align: center;
        }
        .metric-badge label {
            font-size: 11px;
            color: #6b7280;
            font-weight: 700;
            text-transform: uppercase;
            display: block;
        }
        .metric-badge h2 {
            margin: 6px 0 0 0;
            font-size: 24px;
            color: #1e1b4b !important;
            font-weight: 800;
        }
    </style>
""", unsafe_allow_html=True)

# --- ML Pipeline Workflow ---
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

    # Assign ML Predictions & Probabilities
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

# --- Top Header ---
st.markdown("""
    <div class="app-header">
        <div>
            <h1>🎓 AI STUDENT PERFORMANCE DASHBOARD</h1>
            <p>Machine Learning Academic Advisory & Performance Analytics System</p>
        </div>
    </div>
""", unsafe_allow_html=True)

# --- Top Selector Bar ---
top_col1, top_col2 = st.columns([2, 3])
with top_col1:
    selected_section = st.radio(
        "Navigation", 
        ["📊 Student Performance Desk", "🏠 Institutional Overview", "📈 Analytics & ML Verification"],
        horizontal=True
    )

with top_col2:
    student_options = df[["Student_ID", "Student_Name"]].drop_duplicates().sort_values(by="Student_ID")
    student_options["Label"] = student_options["Student_ID"].astype(str) + " - " + student_options["Student_Name"].astype(str)
    
    selected_student_label = st.selectbox("Select Student Profile", options=student_options["Label"].tolist(), index=0)
    selected_student_id = selected_student_label.split(" - ")[0]

st.markdown("---")


# ==========================================
# SECTION 1: STUDENT PERFORMANCE DESK
# ==========================================
if selected_section == "📊 Student Performance Desk":
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

    # --- Left Column: Marks Matrix & Subject Score Chart ---
    with col_left:
        st.markdown(f"#### 📝 Performance Summary — {student_name}")
        matrix_df = student_df[["Subject", "Quiz_Score", "Assignment_Score", "Midterm_Marks", "Final_Marks"]].copy()
        matrix_df.columns = ["Subject", "Quiz", "Assignment", "Midterm", "Final Marks"]
        st.dataframe(matrix_df, use_container_width=True, hide_index=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 📊 Subject Marks Comparison")
        
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
            margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # --- Middle Column: Student Profile & Attendance ---
    with col_mid:
        st.markdown("<div class='card-container' style='text-align: center;'>", unsafe_allow_html=True)
        st.image(f"https://api.dicebear.com/7.x/avataaars/svg?seed={student_name}", width=110)
        st.markdown(f"<h3 style='margin: 8px 0 0 0; color: #1e1b4b;'>{student_name}</h3>", unsafe_allow_html=True)
        st.markdown(f"<p style='color: #6b7280; font-size: 12px; margin-bottom: 12px;'>ID: {selected_student_id} | {student_department}</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        b1, b2 = st.columns(2)
        with b1:
            st.markdown(f"""
                <div class='profile-badge'>
                    <label>Attendance Rate</label>
                    <span>{avg_attendance:.1f}%</span>
                </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
                <div class='profile-badge'>
                    <label>Calculated GPA</label>
                    <span>{cgpa} / 10.0</span>
                </div>
            """, unsafe_allow_html=True)
        with b2:
            st.markdown(f"""
                <div class='profile-badge'>
                    <label>Pass Rate</label>
                    <span>{pass_rate:.1f}%</span>
                </div>
            """, unsafe_allow_html=True)
            risk_color = "#ef4444" if ml_requires_attention else "#10b981"
            risk_text = "Requires Attention" if ml_requires_attention else "Stable"
            st.markdown(f"""
                <div class='profile-badge'>
                    <label>ML Risk Status</label>
                    <span style='color: {risk_color};'>{risk_text} ({ml_risk_prob:.2f})</span>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("##### 📅 Monthly Attendance Breakdown")
        att_monthly = pd.DataFrame({
            "Month": ["Sep", "Oct", "Nov", "Dec"],
            "Days Present": [22, 24, 20, 23]
        })
        st.dataframe(att_monthly, use_container_width=True, hide_index=True)

    # --- Right Column: Recommendations & CGPA Trend ---
    with col_right:
        st.markdown("#### 💡 ML Performance Recommendations")
        for idx, row in student_df.iterrows():
            prob = row["ML_Risk_Probability"]
            subj = row["Subject"]
            
            if prob >= 0.5:
                st.error(f"⚠️ **{subj}** (Risk Prob: `{prob:.2f}`)\n\n*Action:* Low assessment scores/attendance detected. Schedule review sessions.")
            else:
                st.success(f"✅ **{subj}** (Risk Prob: `{prob:.2f}`)\n\n*Action:* Performance is on track. Maintain study schedule.")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 📈 Semester CGPA Progression")
        
        cgpa_df = pd.DataFrame({
            "Semester": ["Sem 1", "Sem 2", "Sem 3", "Sem 4"],
            "My CGPA": [cgpa - 0.2, cgpa - 0.1, cgpa, round(cgpa + 0.1, 2)],
            "Class Avg": [7.5, 7.6, 7.8, 7.9]
        })
        
        fig_cgpa = px.line(
            cgpa_df, x="Semester", y=["My CGPA", "Class Avg"], markers=True,
            color_discrete_sequence=["#4338ca", "#a5b4fc"]
        )
        fig_cgpa.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            yaxis_range=[0, 10], margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_cgpa, use_container_width=True)


# ==========================================
# SECTION 2: INSTITUTIONAL OVERVIEW
# ==========================================
elif selected_section == "🏠 Institutional Overview":
    st.markdown("### Institutional Summary & Performance Metrics")
    
    k1, k2, k3, k4 = st.columns(4)
    total_students = df["Student_ID"].nunique()
    total_faculty = 5
    subjects_count = df["Subject"].nunique()
    avg_marks = df["Final_Marks"].mean()

    with k1:
        st.markdown(f"<div class='metric-badge'><label>Total Students</label><h2>{total_students}</h2></div>", unsafe_allow_html=True)
    with k2:
        st.markdown(f"<div class='metric-badge'><label>Total Faculty</label><h2>{total_faculty}</h2></div>", unsafe_allow_html=True)
    with k3:
        st.markdown(f"<div class='metric-badge'><label>Subjects Offered</label><h2>{subjects_count}</h2></div>", unsafe_allow_html=True)
    with k4:
        st.markdown(f"<div class='metric-badge'><label>Average Mark</label><h2>{avg_marks:.2f}</h2></div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🏆 Dean's Honor Roll — Top Performers")
    top_students = df.groupby(["Student_ID", "Student_Name", "Department"])["Final_Marks"].mean().reset_index().sort_values(by="Final_Marks", ascending=False).head(3)
    
    t_col1, t_col2, t_col3 = st.columns(3)
    ranks = ["RANK 1", "RANK 2", "RANK 3"]
    
    for idx, (_, row) in enumerate(top_students.iterrows()):
        col = [t_col1, t_col2, t_col3][idx]
        with col:
            st.markdown(f"""
                <div class='card-container' style='text-align: center;'>
                    <img src='https://api.dicebear.com/7.x/avataaars/svg?seed={row["Student_Name"]}' width='80'/>
                    <h4 style='margin: 8px 0 2px 0; color: #1e1b4b;'>{row["Student_Name"]}</h4>
                    <p style='font-size: 12px; color: #6b7280; margin-bottom: 6px;'>ID: {row["Student_ID"]} | {row["Department"]}</p>
                    <hr style='margin: 6px 0;'>
                    <p style='font-size: 16px; font-weight: 800; color: #4338ca; margin: 0;'>Score: {row["Final_Marks"]:.2f}</p>
                    <span style='background-color: #e0e7ff; color: #3730a3; padding: 2px 8px; border-radius: 6px; font-size: 11px; font-weight: bold;'>{ranks[idx]}</span>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    
    c_left, c_right = st.columns(2)
    with c_left:
        st.markdown("#### Average Marks by Department")
        dept_avg = df.groupby("Department")["Final_Marks"].mean().reset_index()
        fig_dept = px.bar(
            dept_avg, x="Department", y="Final_Marks", color="Department",
            color_discrete_sequence=["#4338ca", "#6366f1", "#818cf8", "#a5b4fc"]
        )
        fig_dept.update_layout(showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_dept, use_container_width=True)

    with c_right:
        st.markdown("#### Overall Risk Status Breakdown")
        risk_counts = df["ML_Attention_Predicted"].map({1: "Requires Attention", 0: "Stable"}).value_counts().reset_index()
        risk_counts.columns = ["Status", "Count"]
        fig_donut = px.pie(
            risk_counts, values="Count", names="Status", hole=0.5,
            color="Status", color_discrete_map={"Requires Attention": "#ef4444", "Stable": "#10b981"}
        )
        fig_donut.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_donut, use_container_width=True)


# ==========================================
# SECTION 3: ANALYTICS & ML VERIFICATION
# ==========================================
elif selected_section == "📈 Analytics & ML Verification":
    st.markdown("### 📊 Interactive Analytics & Machine Learning Workflow")
    
    # Slicers Panel
    st.sidebar.title("🔍 Analytics Filters")
    selected_depts = st.sidebar.multiselect(
        "Select Departments", 
        options=df["Department"].unique().tolist(), 
        default=df["Department"].unique().tolist()
    )
    
    selected_sems = st.sidebar.multiselect(
        "Select Semesters", 
        options=df["Semester"].unique().tolist(), 
        default=df["Semester"].unique().tolist()
    )
    
    risk_filter = st.sidebar.selectbox(
        "Filter ML Risk Flag",
        options=["All Students", "Flagged (Requires Attention)", "Stable (No Risk)"]
    )

    filtered_df = df[(df["Department"].isin(selected_depts)) & (df["Semester"].isin(selected_sems))]
    if risk_filter == "Flagged (Requires Attention)":
        filtered_df = filtered_df[filtered_df["ML_Attention_Predicted"] == 1]
    elif risk_filter == "Stable (No Risk)":
        filtered_df = filtered_df[filtered_df["ML_Attention_Predicted"] == 0]

    # Analytics Charts Row 1
    g1, g2, g3 = st.columns(3)
    with g1:
        st.markdown("##### Students per Department")
        fig_g1 = px.bar(
            filtered_df.groupby("Department")["Student_ID"].nunique().reset_index(),
            x="Department", y="Student_ID", color="Department",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_g1.update_layout(showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_g1, use_container_width=True)

    with g2:
        st.markdown("##### Attendance Risk Breakdown")
        att_cat = np.where(filtered_df["Attendance"] < 75, "Low (<75%)", "Good (≥75%)")
        att_df = pd.Series(att_cat).value_counts().reset_index()
        att_df.columns = ["Category", "Count"]
        fig_g2 = px.pie(att_df, values="Count", names="Category", hole=0.5, color_discrete_sequence=["#ef4444", "#10b981"])
        fig_g2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_g2, use_container_width=True)

    with g3:
        st.markdown("##### Marks Sum by Grade Level")
        grade_df = filtered_df.groupby("Grade")["Final_Marks"].sum().reset_index().sort_values(by="Final_Marks")
        fig_g3 = px.bar(
            grade_df, y="Grade", x="Final_Marks", orientation="h", color="Grade",
            color_discrete_sequence=px.colors.sequential.Purples_r
        )
        fig_g3.update_layout(showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_g3, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🌲 scikit-learn Model Evaluation (Slide 6)")

    m_col1, m_col2 = st.columns(2)
    with m_col1:
        st.markdown("#### Primary Model: Random Forest Classifier")
        rf_metrics_df = pd.DataFrame({
            "Class": ["No Risk (0)", "Requires Attention (1)"],
            "Precision": [rf_report['0']['precision'], rf_report['1']['precision']],
            "Recall": [rf_report['0']['recall'], rf_report['1']['recall']],
            "F1-Score": [rf_report['0']['f1-score'], rf_report['1']['f1-score']]
        })
        st.dataframe(rf_metrics_df.style.format(precision=2), use_container_width=True)

    with m_col2:
        st.markdown("#### Baseline Model: Logistic Regression")
        lr_metrics_df = pd.DataFrame({
            "Class": ["No Risk (0)", "Requires Attention (1)"],
            "Precision": [lr_report['0']['precision'], lr_report['1']['precision']],
            "Recall": [lr_report['0']['recall'], lr_report['1']['recall']],
            "F1-Score": [lr_report['0']['f1-score'], lr_report['1']['f1-score']]
        })
        st.dataframe(lr_metrics_df.style.format(precision=2), use_container_width=True)

    st.markdown("---")
    st.markdown("#### 🔍 Random Forest Feature Importances (Slide 11)")
    
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
    fig_importance.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_importance, use_container_width=True)
