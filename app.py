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
    page_title="Students Performance & Risk Advisory Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- POWER BI UI THEME & STYLING CSS ---
st.markdown("""
    <style>
        .stApp {
            background-color: #f0f2f6 !important;
            font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
        }
        header {visibility: hidden;}
        
        /* Purple Sidebar Panel matching Screenshot */
        section[data-testid="stSidebar"] {
            background-color: #4c51bf !important;
            color: white !important;
            padding-top: 20px;
        }
        section[data-testid="stSidebar"] .stMarkdown h2, 
        section[data-testid="stSidebar"] label, 
        section[data-testid="stSidebar"] p {
            color: white !important;
        }

        /* Power BI Dashboard Containers */
        .dashboard-card {
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 16px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            margin-bottom: 16px;
        }

        /* Top KPI Summary Cards */
        .kpi-card {
            background-color: #ffffff;
            border-top: 4px solid #4c51bf;
            border-radius: 6px;
            padding: 14px 18px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .kpi-title {
            font-size: 11px;
            text-transform: uppercase;
            color: #718096;
            font-weight: 700;
            letter-spacing: 0.05em;
        }
        .kpi-value {
            font-size: 26px;
            font-weight: 800;
            color: #1a202c;
            margin-top: 2px;
        }

        /* Rank Cards */
        .rank-card {
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 12px;
            text-align: center;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }
        .rank-banner {
            background-color: #edf2f7;
            font-weight: 800;
            font-size: 13px;
            color: #2d3748;
            padding: 6px;
            border-radius: 4px;
            margin-top: 10px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        /* Actionable Advisory Cards */
        .action-card {
            background: #fff5f5;
            border-left: 4px solid #e53e3e;
            border-radius: 6px;
            padding: 12px 16px;
            margin-bottom: 10px;
        }
        .action-card-title {
            font-weight: 700;
            color: #c53030;
            font-size: 13px;
        }
        .action-step {
            font-size: 12px;
            color: #4a5568;
            margin-top: 4px;
        }
    </style>
""", unsafe_allow_html=True)

# --- ML PIPELINE & DATA PROCESSING ---
@st.cache_resource
def run_ml_pipeline_and_load_data():
    excel_path = "data/Subject_Analytics_Dataset_200_Students.xlsx"
    if os.path.exists(excel_path):
        df = pd.read_excel(excel_path)
        df.columns = df.columns.str.strip()
    else:
        np.random.seed(42)
        n = 200
        df = pd.DataFrame({
            "Student_ID": [f"STU{str(i).zfill(3)}" for i in range(1, n + 1)],
            "Student_Name": [f"Student {i}" for i in range(1, n + 1)],
            "Semester": np.random.choice(["Semester 1", "Semester 2", "Semester 3", "Semester 4"], size=n),
            "Year_Level": np.random.choice(["Year 1", "Year 2", "Year 3", "Year 4"], size=n),
            "Gender": np.random.choice(["Female", "Male"], size=n, p=[0.5, 0.5]),
            "Department": np.random.choice(["Language", "STEM", "Humanities", "Physical Ed"], size=n),
            "Nationality": np.random.choice(["Indian", "Korean", "Italian", "Japanese", "French", "Canadian"], size=n),
            "Subject": np.random.choice(["Mathematics", "Physics", "Chemistry", "Programming", "Database Systems"], size=n),
            "Age": np.random.randint(14, 19, size=n),
            "Attendance": np.random.uniform(60, 100, size=n),
            "Final_Marks": np.random.uniform(40, 98, size=n),
            "Midterm_Marks": np.random.uniform(35, 95, size=n),
            "Quiz_Score": np.random.uniform(50, 100, size=n),
            "Assignment_Score": np.random.uniform(50, 100, size=n)
        })

    # Ensure required columns exist
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.strip()

    if "Semester" not in df.columns and "Year_Level" in df.columns:
        df["Semester"] = df["Year_Level"]
    elif "Semester" not in df.columns:
        df["Semester"] = "Semester 1"

    if "Year_Level" not in df.columns:
        df["Year_Level"] = df["Semester"]

    if "Gender" not in df.columns:
        df["Gender"] = "Female"

    if "Department" not in df.columns:
        df["Department"] = "STEM"

    if "Nationality" not in df.columns:
        df["Nationality"] = "Indian"

    if "Student_Name" not in df.columns:
        df["Student_Name"] = df["Student_ID"]

    if "Subject" not in df.columns:
        df["Subject"] = "General Academics"

    # Feature Engineering
    if "performance_trend" not in df.columns:
        df["performance_trend"] = np.where(df["Final_Marks"] >= df["Midterm_Marks"], "Improving/Stable", "Declining")

    if "missed_assessments" not in df.columns:
        df["missed_assessments"] = np.where(df["Attendance"] < 75, 2, 0)

    trend_numeric = df["performance_trend"].map({"Improving/Stable": 1, "Declining": -1}).fillna(0)
    feature_cols = ["Attendance", "Quiz_Score", "Assignment_Score", "Midterm_Marks", "Final_Marks", "missed_assessments"]
    
    X = df[feature_cols].copy()
    X["trend_numeric"] = trend_numeric

    # Risk Target & ML Training
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

    df["ML_Attention_Predicted"] = rf_model.predict(X)
    df["ML_Risk_Probability"] = rf_model.predict_proba(X)[:, 1]

    def get_grade(marks):
        if marks >= 80: return "A"
        elif marks >= 70: return "B"
        elif marks >= 60: return "C"
        elif marks >= 50: return "D"
        else: return "F"

    df["Grade"] = df["Final_Marks"].apply(get_grade)

    return df, rf_model, rf_report, lr_report, feature_cols

df, rf_model, rf_report, lr_report, feature_cols = run_ml_pipeline_and_load_data()

# --- SIDEBAR (Purple Control Panel) ---
with st.sidebar:
    st.markdown("## Students Performance Dashboard")
    st.markdown("---")
    
    st.markdown("##### Date Range")
    date_col1, date_col2 = st.columns(2)
    with date_col1:
        start_date = st.date_input("Start", value=pd.to_datetime("2022-09-01"))
    with date_col2:
        end_date = st.date_input("End", value=pd.to_datetime("2025-07-15"))
        
    st.markdown("---")
    record_id = st.selectbox("RecordID", options=["All"] + sorted(df["Student_ID"].unique().tolist()))
    st.markdown("---")
    gender_filter = st.radio("Gender", options=["All"] + sorted(df["Gender"].unique().tolist()), horizontal=False)
    
    st.markdown("<br>", unsafe_allow_html=True)
    sb_col1, sb_col2 = st.columns(2)
    with sb_col1:
        if st.button("🔄 Reset", help="Reset Filters"):
            st.rerun()
    with sb_col2:
        st.button("ℹ️ Info", help="Dashboard Information")

# --- FILTER DATAFRAME ---
filtered_df = df.copy()
if record_id != "All":
    filtered_df = filtered_df[filtered_df["Student_ID"] == record_id]
if gender_filter != "All":
    filtered_df = filtered_df[filtered_df["Gender"] == gender_filter]

# --- MAIN DASHBOARD AREA ---

# Top Navigation Bar with View Selector
nav_tab = st.radio(
    "",
    ["OVERVIEW", "STUDENTS & TEACHERS", "ASSESSMENTS"],
    index=0,
    horizontal=True
)

st.markdown("<hr style='margin: 10px 0 20px 0; border-color: #cbd5e0;'>", unsafe_allow_html=True)

# 4 Top Summary KPI Cards
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.markdown(f"""
        <div class="kpi-card" style="border-top-color: #f6ad55;">
            <div>
                <div class="kpi-title">Total Students Enrolled</div>
                <div class="kpi-value">{filtered_df['Student_ID'].nunique():,}</div>
            </div>
            <div style="font-size: 28px;">🎓</div>
        </div>
    """, unsafe_allow_html=True)

with kpi2:
    st.markdown("""
        <div class="kpi-card" style="border-top-color: #ecc94b;">
            <div>
                <div class="kpi-title">Total Teachers</div>
                <div class="kpi-value">5</div>
            </div>
            <div style="font-size: 28px;">👨‍🏫</div>
        </div>
    """, unsafe_allow_html=True)

with kpi3:
    st.markdown("""
        <div class="kpi-card" style="border-top-color: #48bb78;">
            <div>
                <div class="kpi-title">Subjects Offered</div>
                <div class="kpi-value">8</div>
            </div>
            <div style="font-size: 28px;">📚</div>
        </div>
    """, unsafe_allow_html=True)

with kpi4:
    avg_marks_val = filtered_df['Final_Marks'].mean() if not filtered_df.empty else 0.0
    st.markdown(f"""
        <div class="kpi-card" style="border-top-color: #38bdf8;">
            <div>
                <div class="kpi-title">Average of Marks</div>
                <div class="kpi-value">{avg_marks_val:.2f}</div>
            </div>
            <div style="font-size: 28px;">📊</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- TAB 1: OVERVIEW & COHORT ANALYTICS ---
if nav_tab == "OVERVIEW":
    # ROW 1: Donut Chart & Top 3 Performing Students
    row1_col1, row1_col2, row1_col3, row1_col4 = st.columns([1.3, 1, 1, 1])

    with row1_col1:
        st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
        st.markdown("<p style='font-size: 12px; color: #718096; font-weight: 700; margin-bottom: 0;'>Pass % by Year Level</p>", unsafe_allow_html=True)
        
        year_pass = filtered_df.groupby("Year_Level")["Final_Marks"].apply(lambda x: (x >= 50).mean() * 100).reset_index(name="Pass_Rate")
        fig_donut = px.pie(
            year_pass, names="Year_Level", values="Pass_Rate", hole=0.65,
            color_discrete_sequence=["#4c51bf", "#6366f1", "#818cf8", "#c7d2fe"]
        )
        fig_donut.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=190, margin=dict(t=10, b=10, l=10, r=10), showlegend=False
        )
        fig_donut.update_traces(textinfo='percent', textfont_size=11)
        st.plotly_chart(fig_donut, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    top_students = filtered_df.sort_values(by="Final_Marks", ascending=False).drop_duplicates(subset=["Student_ID"]).head(3)
    dummy_students = [
        {"name": "Luis Meyer", "id": "STU007", "age": 15, "nat": "German", "tot": 87.15, "rank": "RANK 1"},
        {"name": "Harper Martinez", "id": "STU014", "age": 15, "nat": "American", "tot": 87.02, "rank": "RANK 2"},
        {"name": "Sophia Fischer", "id": "STU009", "age": 15, "nat": "German", "tot": 87.02, "rank": "RANK 3"}
    ]

    for idx, col_ref in enumerate([row1_col2, row1_col3, row1_col4]):
        s_data = dummy_students[idx]
        if idx < len(top_students):
            row = top_students.iloc[idx]
            s_name = row.get("Student_Name", s_data["name"])
            s_id = row.get("Student_ID", s_data["id"])
            s_tot = f"{row.get('Final_Marks', s_data['tot']):.2f}"
            s_nat = row.get("Nationality", s_data["nat"])
        else:
            s_name, s_id, s_tot, s_nat = s_data["name"], s_data["id"], f"{s_data['tot']:.2f}", s_data["nat"]

        with col_ref:
            st.markdown(f"""
                <div class="rank-card">
                    <p style="font-size: 11px; color: #718096; font-weight: 700; margin-bottom: 4px;">Top 3 Performing Students</p>
                    <div style="background: #e2e8f0; width: 45px; height: 45px; border-radius: 50%; margin: 6px auto; display: flex; align-items: center; justify-content: center; font-size: 18px;">👤</div>
                    <div style="font-weight: 800; font-size: 13px; color: #1a202c; margin-bottom: 6px;">{s_name}</div>
                    <hr style="border: none; border-top: 1px solid #e2e8f0; margin: 4px 0;">
                    <div style="font-size: 11px; text-align: left; color: #4a5568; line-height: 1.4;">
                        <div style="display: flex; justify-content: space-between;"><span>Student ID</span><b>{s_id}</b></div>
                        <div style="display: flex; justify-content: space-between;"><span>Age</span><b>15</b></div>
                        <div style="display: flex; justify-content: space-between;"><span>Nationality</span><b>{s_nat}</b></div>
                        <div style="display: flex; justify-content: space-between;"><span>Total %</span><b>{s_tot}</b></div>
                    </div>
                    <div class="rank-banner">{s_data['rank']}</div>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ROW 2: Bar & Line Analytical Charts
    row2_col1, row2_col2, row2_col3 = st.columns(3)

    with row2_col1:
        st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
        st.markdown("<p style='font-size: 12px; color: #718096; font-weight: 700; margin-bottom: 4px;'>Average Weighted Score by Department</p>", unsafe_allow_html=True)
        
        dept_score = filtered_df.groupby("Department")["Final_Marks"].mean().reset_index()
        fig_dept = px.bar(dept_score, x="Department", y="Final_Marks", color_discrete_sequence=["#4c51bf"])
        fig_dept.update_traces(marker_color="#4c51bf")
        fig_dept.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=210, margin=dict(t=10, b=10, l=10, r=10), xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor="#edf2f7"))
        st.plotly_chart(fig_dept, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with row2_col2:
        st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
        st.markdown("<p style='font-size: 12px; color: #718096; font-weight: 700; margin-bottom: 4px;'>Average Score by Nationality</p>", unsafe_allow_html=True)
        
        nat_score = filtered_df.groupby("Nationality")["Final_Marks"].mean().reset_index().sort_values(by="Final_Marks", ascending=True)
        fig_nat = px.bar(nat_score, x="Final_Marks", y="Nationality", orientation="h", color_discrete_sequence=["#4c51bf"])
        fig_nat.update_traces(marker_color="#4c51bf")
        fig_nat.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=210, margin=dict(t=10, b=10, l=10, r=10), xaxis=dict(showgrid=True, gridcolor="#edf2f7"), yaxis=dict(showgrid=False))
        st.plotly_chart(fig_nat, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with row2_col3:
        st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
        st.markdown("<p style='font-size: 12px; color: #718096; font-weight: 700; margin-bottom: 4px;'>Pass & Fail Rate Trend</p>", unsafe_allow_html=True)
        
        years = ["2022", "2023", "2024", "2025"]
        pass_rates = [69.23, 75.83, 80.00, 80.00]
        fail_rates = [30.77, 24.17, 20.00, 20.00]
        
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Bar(x=years, y=pass_rates, name="Pass %", marker_color="#4c51bf"))
        fig_trend.add_trace(go.Scatter(x=years, y=fail_rates, name="Fail %", mode="lines+markers", line=dict(color="#1a202c", width=2)))
        fig_trend.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=210, margin=dict(t=10, b=10, l=10, r=10), showlegend=False, xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor="#edf2f7"))
        st.plotly_chart(fig_trend, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# --- TAB 2: STUDENTS & TEACHERS (DEDICATED STUDENT-BASED SECTION & AI RECOVERY) ---
elif nav_tab == "STUDENTS & TEACHERS":
    st.markdown("### 👤 Student Profile & Prescriptive Recovery Advisory")
    
    selected_stu = st.selectbox("Select Student Profile to Examine", options=sorted(df["Student_ID"].unique().tolist()))
    stu_data = df[df["Student_ID"] == selected_stu]
    
    if not stu_data.empty:
        stu_row = stu_data.iloc[0]
        stu_name = stu_row["Student_Name"]
        stu_dept = stu_row["Department"]
        stu_avg = stu_data["Final_Marks"].mean()
        stu_att = stu_data["Attendance"].mean()
        stu_risk_prob = stu_data["ML_Risk_Probability"].mean()
        is_risk = stu_risk_prob >= 0.5
        
        p_col1, p_col2 = st.columns([1, 2.5])
        
        with p_col1:
            st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
            st.markdown(f"#### 👤 {stu_name}")
            st.markdown(f"**Student ID:** `{selected_stu}`")
            st.markdown(f"**Department:** {stu_dept}")
            st.markdown(f"**Average Attendance:** `{stu_att:.1f}%`")
            st.markdown(f"**Average Mark:** `{stu_avg:.1f}/100`")
            
            if is_risk:
                st.error(f"⚠️ **ML Status:** Flagged At-Risk (Index: `{stu_risk_prob:.2f}`)")
            else:
                st.success(f"✅ **ML Status:** Stable Standing (Index: `{stu_risk_prob:.2f}`)")
            st.markdown("</div>", unsafe_allow_html=True)

        with p_col2:
            st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
            st.markdown("#### 🛠️ AI Prescriptive Recovery Roadmap")
            
            flagged_subjects = stu_data[stu_data["ML_Risk_Probability"] >= 0.5]
            if len(flagged_subjects) > 0:
                for _, row in flagged_subjects.iterrows():
                    subj = row["Subject"]
                    quiz = row["Quiz_Score"]
                    midterm = row["Midterm_Marks"]
                    att = row["Attendance"]
                    
                    steps = []
                    if quiz < 60:
                        steps.append(f"📌 **Quiz Remediation:** Quiz score is `{quiz:.1f}/100`. Schedule weekly self-assessment tests.")
                    if midterm < 60:
                        steps.append(f"📌 **Midterm Review:** Midterm score is `{midterm:.1f}/100`. Book 1-on-1 faculty office hours.")
                    if att < 75:
                        steps.append(f"📌 **Attendance Recovery:** Attendance is `{att:.1f}%`. Enroll in mandatory tutorial sessions.")
                    steps.append("📌 **Practice Worksheet:** Complete 3 problem sets before final exams.")
                    
                    steps_text = "<br>".join(steps)
                    st.markdown(f"""
                        <div class='action-card'>
                            <div class='action-card-title'>⚠️ Module: {subj}</div>
                            <div class='action-step'>{steps_text}</div>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("✅ **Optimal Standing:** No active intervention or recovery steps required for this student.")
            
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
        st.markdown("#### 📚 Enrolled Modules & Score Matrix")
        matrix_view = stu_data[["Subject", "Attendance", "Quiz_Score", "Assignment_Score", "Midterm_Marks", "Final_Marks", "Grade", "ML_Risk_Probability"]].copy()
        matrix_view.columns = ["Module Name", "Attendance %", "Quiz Score", "Assignment", "Midterm", "Final Marks", "Grade", "Risk Probability"]
        st.dataframe(matrix_view.style.format({"Attendance %": "{:.1f}", "Risk Probability": "{:.2f}"}), use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

# --- TAB 3: ASSESSMENTS & ML EVALUATION ---
elif nav_tab == "ASSESSMENTS":
    st.markdown("### 📊 Assessment Analytics & Machine Learning Model Verification")
    
    a_col1, a_col2 = st.columns(2)
    
    with a_col1:
        st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
        st.markdown("#### 🌲 Random Forest Classifier Metrics")
        rf_df = pd.DataFrame({
            "Class": ["Stable (0)", "At-Risk (1)"],
            "Precision": [rf_report['0']['precision'], rf_report['1']['precision']],
            "Recall": [rf_report['0']['recall'], rf_report['1']['recall']],
            "F1-Score": [rf_report['0']['f1-score'], rf_report['1']['f1-score']]
        })
        st.dataframe(rf_df.style.format(precision=2), use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with a_col2:
        st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
        st.markdown("#### 🔍 Feature Importance Ranking")
        feat_df = pd.DataFrame({
            "Feature": feature_cols,
            "Importance": rf_model.feature_importances_
        }).sort_values(by="Importance", ascending=True)
        
        fig_imp = px.bar(feat_df, x="Importance", y="Feature", orientation="h", color_discrete_sequence=["#4c51bf"])
        fig_imp.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=200, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_imp, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
