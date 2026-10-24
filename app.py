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
    page_title="Student Academic Learning & Growth Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- HIGH-CONTRAST LIGHT ACADEMIC THEME (BLACK TEXT) ---
st.markdown("""
    <style>
        /* Global Text & Page Theme */
        .stApp {
            background-color: #f8fafc !important;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            color: #000000 !important;
        }

        /* Force black text across standard inputs and paragraphs */
        p, span, label, h1, h2, h3, h4, h5, h6, div {
            color: #000000 !important;
        }

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: #ffffff !important;
            border-right: 2px solid #cbd5e1;
            padding-top: 16px;
        }

        /* Card Container Styling */
        .student-card {
            background-color: #ffffff;
            border: 1.5px solid #cbd5e1;
            border-radius: 12px;
            padding: 18px 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            margin-bottom: 20px;
        }

        /* KPI Cards with High Contrast Black Text */
        .kpi-card {
            background: #ffffff;
            border: 1.5px solid #cbd5e1;
            border-left: 5px solid #2563eb;
            border-radius: 10px;
            padding: 16px 18px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.04);
        }
        .kpi-title {
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #334155 !important;
            font-weight: 800;
        }
        .kpi-val {
            font-size: 26px;
            font-weight: 800;
            color: #000000 !important;
            margin-top: 4px;
        }

        /* Educational Learning Support Boxes */
        .study-plan-card {
            background-color: #fffbe3;
            border: 1.5px solid #fde047;
            border-radius: 10px;
            padding: 16px;
            margin-bottom: 12px;
        }
        .study-plan-title {
            font-weight: 800;
            color: #854d0e !important;
            font-size: 14px;
            margin-bottom: 8px;
        }
        .study-step {
            font-size: 13px;
            color: #1e293b !important;
            margin-bottom: 6px;
            font-weight: 500;
        }

        /* Status Badges */
        .badge-stable {
            background-color: #dcfce7;
            color: #166534 !important;
            border: 1px solid #86efac;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 700;
        }
        .badge-support {
            background-color: #fee2e2;
            color: #991b1b !important;
            border: 1px solid #fca5a5;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 700;
        }

        /* Streamlit Dataframe High Contrast Light Table Override */
        .stDataFrame {
            background-color: #ffffff !important;
            border: 1.5px solid #cbd5e1 !important;
            border-radius: 10px !important;
        }
    </style>
""", unsafe_allow_html=True)

# --- DATA INGESTION & EDUCATIONAL ML MODEL ---
@st.cache_resource
def load_and_prepare_academic_data():
    excel_path = "data/Subject_Analytics_Dataset_200_Students.xlsx"
    
    if not os.path.exists(excel_path):
        # Synthetic Academic Data Fallback
        np.random.seed(42)
        n = 200
        df = pd.DataFrame({
            "Student_ID": [f"S{str(i).zfill(3)}" for i in range(1, n + 1)],
            "Student_Name": [f"Student {i}" for i in range(1, n + 1)],
            "Semester": np.random.choice(["Semester 1", "Semester 2", "Semester 3", "Semester 4"], size=n),
            "Gender": np.random.choice(["Female", "Male"], size=n, p=[0.5, 0.5]),
            "Department": np.random.choice(["CSE", "ECE", "EEE", "Mechanical"], size=n),
            "Subject": np.random.choice(["Mathematics", "Physics", "Chemistry", "Programming", "Database Systems"], size=n),
            "Attendance": np.random.uniform(60, 100, size=n),
            "Final_Marks": np.random.uniform(40, 98, size=n),
            "Midterm_Marks": np.random.uniform(35, 95, size=n),
            "Quiz_Score": np.random.uniform(50, 100, size=n),
            "Assignment_Score": np.random.uniform(50, 100, size=n)
        })
    else:
        df = pd.read_excel(excel_path)

    # Clean Strings
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.strip()

    if "performance_trend" not in df.columns:
        df["performance_trend"] = np.where(df["Final_Marks"] >= df["Midterm_Marks"], "Improving/Stable", "Needs Attention")

    if "missed_assessments" not in df.columns:
        df["missed_assessments"] = np.where(df["Attendance"] < 75, 2, 0)

    trend_num = df["performance_trend"].map({"Improving/Stable": 1, "Needs Attention": -1}).fillna(0)
    feature_cols = ["Attendance", "Quiz_Score", "Assignment_Score", "Midterm_Marks", "Final_Marks", "missed_assessments"]
    
    X = df[feature_cols].copy()
    X["trend_numeric"] = trend_num

    # Composite Student Academic Score Calculation
    total_score = (df["Attendance"] * 0.25) + (df["Final_Marks"] * 0.35) + (df["Midterm_Marks"] * 0.25) + (df["Quiz_Score"] * 0.15)
    df["needs_support_target"] = np.where(total_score < 58, 1, 0)
    y = df["needs_support_target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_report = classification_report(y_test, rf_model.predict(X_test), output_dict=True)

    df["Needs_Learning_Support"] = rf_model.predict(X)
    df["Support_Need_Index"] = rf_model.predict_proba(X)[:, 1]

    def calculate_grade(m):
        if m >= 80: return "A"
        elif m >= 70: return "B"
        elif m >= 60: return "C"
        elif m >= 50: return "D"
        else: return "F"

    df["Grade"] = df["Final_Marks"].apply(calculate_grade)

    return df, rf_model, rf_report, feature_cols

df, rf_model, rf_report, feature_cols = load_and_prepare_academic_data()

# --- SIDEBAR FILTERS ---
with st.sidebar:
    st.markdown("## 📚 Academic Navigation")
    st.markdown("Select options to view student learning results.")
    st.markdown("---")
    
    selected_dept = st.selectbox("Department Filter", options=["All Departments"] + sorted(df["Department"].unique().tolist()))
    selected_gender = st.selectbox("Gender Filter", options=["All", "Female", "Male"])
    
    st.markdown("---")
    st.markdown("##### 🏫 View Options")
    selected_section = st.radio(
        "Choose Dashboard View",
        ["📊 Student Overview Dashboard", "👤 Student Profile Study", "🏫 Class Learning Progress", "🤖 AI Model Verification"]
    )

# Filter Data
active_df = df.copy()
if selected_dept != "All Departments":
    active_df = active_df[active_df["Department"] == selected_dept]
if selected_gender != "All":
    active_df = active_df[active_df["Gender"] == selected_gender]


# ==============================================================================
# VIEW 1: STUDENT OVERVIEW DASHBOARD
# ==============================================================================
if selected_section == "📊 Student Overview Dashboard":
    st.markdown("### 📊 Student Learning Overview Dashboard")
    st.markdown("A complete summary of enrolled students, marks, attendance rates, and top performers.")
    
    # Overview Top KPIs
    o1, o2, o3, o4 = st.columns(4)
    with o1:
        st.markdown(f"""
            <div class='kpi-card'>
                <div class='kpi-title'>Total Enrolled Students</div>
                <div class='kpi-val'>{active_df['Student_ID'].nunique()}</div>
            </div>
        """, unsafe_allow_html=True)
    with o2:
        st.markdown(f"""
            <div class='kpi-card'>
                <div class='kpi-title'>Average Student Mark</div>
                <div class='kpi-val'>{active_df['Final_Marks'].mean():.1f} / 100</div>
            </div>
        """, unsafe_allow_html=True)
    with o3:
        st.markdown(f"""
            <div class='kpi-card'>
                <div class='kpi-title'>Overall Pass Rate</div>
                <div class='kpi-val'>{(active_df['Final_Marks'] >= 50).mean()*100:.1f}%</div>
            </div>
        """, unsafe_allow_html=True)
    with o4:
        st.markdown(f"""
            <div class='kpi-card' style='border-left-color: #ef4444;'>
                <div class='kpi-title'>Students Needing Support</div>
                <div class='kpi-val'>{active_df['Needs_Learning_Support'].sum()}</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Top Performing Students Ranking
    st.markdown("<div class='student-card'>", unsafe_allow_html=True)
    st.markdown("<h4 style='font-size: 16px; font-weight: 800; color: #000000;'>🏆 Top Academic Performers</h4>", unsafe_allow_html=True)
    
    top_students = active_df.groupby(["Student_ID", "Student_Name"])["Final_Marks"].mean().reset_index().sort_values(by="Final_Marks", ascending=False).head(3)
    
    r1, r2, r3 = st.columns(3)
    rank_cols = [r1, r2, r3]
    rank_labels = ["🥇 Rank 1", "🥈 Rank 2", "🥉 Rank 3"]
    
    for idx, row in top_students.iterrows():
        if idx < 3:
            with rank_cols[idx]:
                st.markdown(f"""
                    <div style='background-color: #f1f5f9; border: 1.5px solid #cbd5e1; border-radius: 8px; padding: 14px; text-align: center;'>
                        <div style='font-size: 14px; font-weight: 800; color: #1e293b;'>{rank_labels[idx]}</div>
                        <div style='font-size: 16px; font-weight: 800; color: #000000; margin-top: 4px;'>{row['Student_Name']}</div>
                        <div style='font-size: 12px; color: #334155; font-weight: 700;'>ID: {row['Student_ID']}</div>
                        <div style='font-size: 18px; font-weight: 800; color: #2563eb; margin-top: 6px;'>Average: {row['Final_Marks']:.1f}%</div>
                    </div>
                """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Department Score Distribution & Attendance Trend
    c_left, c_right = st.columns(2)
    with c_left:
        st.markdown("<div class='student-card'>", unsafe_allow_html=True)
        st.markdown("<h4 style='font-size: 15px; font-weight: 800; color: #000000;'>🏫 Average Marks by Department</h4>", unsafe_allow_html=True)
        dept_avg = active_df.groupby("Department")["Final_Marks"].mean().reset_index()
        fig_dept = px.bar(dept_avg, x="Department", y="Final_Marks", text_auto='.1f', color_discrete_sequence=["#2563eb"])
        fig_dept.update_traces(textfont=dict(color="#000000", size=12))
        fig_dept.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=260,
            xaxis=dict(tickfont=dict(color="#000000", size=12)),
            yaxis=dict(tickfont=dict(color="#000000", size=12), gridcolor="#e2e8f0", title=dict(text="Average Mark", font=dict(color="#000000")))
        )
        st.plotly_chart(fig_dept, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c_right:
        st.markdown("<div class='student-card'>", unsafe_allow_html=True)
        st.markdown("<h4 style='font-size: 15px; font-weight: 800; color: #000000;'>📅 Attendance Class Ratio</h4>", unsafe_allow_html=True)
        att_cats = np.where(active_df["Attendance"] >= 75, "Regular Attendance (≥75%)", "Low Attendance (<75%)")
        att_df = pd.Series(att_cats).value_counts().reset_index()
        att_df.columns = ["Status", "Count"]
        fig_pie = px.pie(att_df, values="Count", names="Status", color_discrete_sequence=["#16a34a", "#dc2626"])
        fig_pie.update_traces(textinfo="percent+label", textfont=dict(color="#000000", size=12))
        fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=260, showlegend=False)
        st.plotly_chart(fig_pie, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ==============================================================================
# VIEW 2: STUDENT PROFILE STUDY (INDIVIDUAL DEEP DIVE WITH CLEAR MARKINGS)
# ==============================================================================
elif selected_section == "👤 Student Profile Study":
    st.markdown("### 👤 Student Profile & Study Guidance")
    st.markdown("Examine individual student marks, attendance, and step-by-step learning support recommendations.")
    
    student_opts = active_df[["Student_ID", "Student_Name"]].drop_duplicates().sort_values(by="Student_ID")
    student_opts["Label"] = student_opts["Student_ID"].astype(str) + " — " + student_opts["Student_Name"].astype(str)
    
    selected_stu_label = st.selectbox("Select Student Record", options=student_opts["Label"].tolist())
    selected_stu_id = selected_stu_label.split(" — ")[0]
    
    stu_df = active_df[active_df["Student_ID"] == selected_stu_id]
    stu_name = stu_df["Student_Name"].iloc[0]
    stu_dept = stu_df["Department"].iloc[0]
    stu_sem = stu_df["Semester"].iloc[0] if "Semester" in stu_df.columns else "Semester 1"
    
    avg_final = stu_df["Final_Marks"].mean()
    avg_att = stu_df["Attendance"].mean()
    support_index = stu_df["Support_Need_Index"].mean()
    needs_help = support_index >= 0.5

    # 4 High Contrast Top Cards
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""
            <div class='kpi-card'>
                <div class='kpi-title'>Student Name & ID</div>
                <div class='kpi-val' style='font-size:18px;'>{stu_name}</div>
                <div style='font-size:12px; color:#334155; font-weight:700;'>ID: {selected_stu_id}</div>
            </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
            <div class='kpi-card'>
                <div class='kpi-title'>Department / Semester</div>
                <div class='kpi-val' style='font-size:18px;'>{stu_dept}</div>
                <div style='font-size:12px; color:#334155; font-weight:700;'>{stu_sem}</div>
            </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
            <div class='kpi-card'>
                <div class='kpi-title'>Average Mark</div>
                <div class='kpi-val'>{avg_final:.1f} / 100</div>
                <div style='font-size:12px; color:#334155; font-weight:700;'>Attendance: {avg_att:.1f}%</div>
            </div>
        """, unsafe_allow_html=True)
    with k4:
        badge_html = "<span class='badge-support'>⚠️ Learning Support Needed</span>" if needs_help else "<span class='badge-stable'>✅ Good Academic Standing</span>"
        st.markdown(f"""
            <div class='kpi-card' style='border-left-color: {"#ef4444" if needs_help else "#16a34a"};'>
                <div class='kpi-title'>Learning Status</div>
                <div style='margin-top:6px;'>{badge_html}</div>
                <div style='font-size:12px; color:#334155; font-weight:700; margin-top:4px;'>Support Need Index: <b>{support_index:.2f}</b></div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1.3, 1.7])

    with col_left:
        st.markdown("<div class='student-card'>", unsafe_allow_html=True)
        st.markdown("<h4 style='font-size: 15px; font-weight: 800; color: #000000;'>📖 Personalized Study Improvement Steps</h4>", unsafe_allow_html=True)
        
        flagged_subjects = stu_df[stu_df["Support_Need_Index"] >= 0.5]
        
        if len(flagged_subjects) > 0:
            for _, row in flagged_subjects.iterrows():
                subj = row["Subject"]
                q_score = row["Quiz_Score"]
                m_score = row["Midterm_Marks"]
                att_rate = row["Attendance"]
                
                steps = []
                if q_score < 60:
                    steps.append(f"Quiz mark is low ({q_score:.0f}/100). Review topic notes and practice weekly quiz questions.")
                if m_score < 60:
                    steps.append(f"Midterm mark is low ({m_score:.0f}/100). Attend teacher office hours for extra concept explanations.")
                if att_rate < 75:
                    steps.append(f"Class attendance is low ({att_rate:.1f}%). Ensure regular attendance to avoid missing key lessons.")
                steps.append("Complete assigned revision worksheets before final examinations.")
                
                rendered_steps = "".join([f"<div class='study-step'>📌 {step}</div>" for step in steps])
                
                st.markdown(f"""
                    <div class='study-plan-card'>
                        <div class='study-plan-title'>⚠️ Subject Needing Attention: {subj} (Need Score: {row['Support_Need_Index']:.2f})</div>
                        {rendered_steps}
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div style='background-color: #dcfce7; border: 1.5px solid #86efac; border-radius: 8px; padding: 16px;'>
                    <div style='font-weight: 800; color: #166534; font-size: 14px;'>✅ Strong Academic Performance</div>
                    <div style='font-size: 13px; color: #1e293b; margin-top: 4px;'>All subject marks meet or exceed standard requirements. Maintain current study habits!</div>
                </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        st.markdown("<div class='student-card'>", unsafe_allow_html=True)
        st.markdown("<h4 style='font-size: 15px; font-weight: 800; color: #000000;'>🕸️ Subject Mark Comparison (Clear Markings)</h4>", unsafe_allow_html=True)
        
        # High Contrast Plotly Radar Chart with Solid Black Text & Markings
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=stu_df["Final_Marks"].tolist(),
            theta=stu_df["Subject"].tolist(),
            fill='toself',
            name='Final Mark',
            line=dict(color='#2563eb', width=2),
            fillcolor='rgba(37, 99, 235, 0.25)'
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=stu_df["Midterm_Marks"].tolist(),
            theta=stu_df["Subject"].tolist(),
            fill='toself',
            name='Midterm Mark',
            line=dict(color='#9333ea', width=2),
            fillcolor='rgba(147, 51, 234, 0.15)'
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], gridcolor='#94a3b8', tickfont=dict(color='#000000', size=11, weight="bold")),
                angularaxis=dict(gridcolor='#94a3b8', tickfont=dict(color='#000000', size=12, weight="bold"))
            ),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=280, margin=dict(t=20, b=20, l=20, r=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5, font=dict(color='#000000', size=12))
        )
        st.plotly_chart(fig_radar, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Detailed High Contrast Subject Marks Table
    st.markdown("<div class='student-card'>", unsafe_allow_html=True)
    st.markdown("<h4 style='font-size: 15px; font-weight: 800; color: #000000;'>📖 Subject Marks Summary Table</h4>", unsafe_allow_html=True)
    
    table_df = stu_df[["Subject", "Attendance", "Quiz_Score", "Assignment_Score", "Midterm_Marks", "Final_Marks", "Grade", "Support_Need_Index"]].copy()
    table_df["Status"] = np.where(table_df["Support_Need_Index"] >= 0.5, "⚠️ Needs Help", "✅ Stable")
    table_df = table_df[["Subject", "Attendance", "Quiz_Score", "Assignment_Score", "Midterm_Marks", "Final_Marks", "Grade", "Status"]]
    table_df.columns = ["Subject Name", "Attendance %", "Quiz Mark", "Assignment Mark", "Midterm Mark", "Final Mark", "Letter Grade", "Learning Status"]
    
    # Styled Table with Clear Dark Text
    st.dataframe(
        table_df.style.format({
            "Attendance %": "{:.1f}", "Quiz Mark": "{:.0f}", "Assignment Mark": "{:.0f}",
            "Midterm Mark": "{:.0f}", "Final Mark": "{:.0f}"
        }),
        use_container_width=True,
        hide_index=True
    )
    st.markdown("</div>", unsafe_allow_html=True)


# ==============================================================================
# VIEW 3: CLASS LEARNING PROGRESS
# ==============================================================================
elif selected_section == "🏫 Class Learning Progress":
    st.markdown("### 🏫 Class Learning Progress & Score Trends")
    st.markdown("Classwide performance summaries across departments and subjects.")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"<div class='kpi-card'><div class='kpi-title'>Total Active Class Group</div><div class='kpi-val'>{active_df['Student_ID'].nunique()}</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='kpi-card'><div class='kpi-title'>Class Average Final Mark</div><div class='kpi-val'>{active_df['Final_Marks'].mean():.1f}</div></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='kpi-card'><div class='kpi-title'>Class Average Attendance</div><div class='kpi-val'>{active_df['Attendance'].mean():.1f}%</div></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    r1, r2 = st.columns(2)
    with r1:
        st.markdown("<div class='student-card'>", unsafe_allow_html=True)
        st.markdown("<h4 style='font-size: 15px; font-weight: 800; color: #000000;'>📊 Midterm vs Final Marks by Department</h4>", unsafe_allow_html=True)
        dept_grp = active_df.groupby("Department")[["Midterm_Marks", "Final_Marks"]].mean().reset_index()
        fig_bar = px.bar(dept_grp, x="Department", y=["Midterm_Marks", "Final_Marks"], barmode="group", color_discrete_sequence=["#2563eb", "#9333ea"])
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=280,
            xaxis=dict(tickfont=dict(color="#000000", size=12)),
            yaxis=dict(tickfont=dict(color="#000000", size=12), gridcolor="#cbd5e1"),
            legend=dict(font=dict(color="#000000"))
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with r2:
        st.markdown("<div class='student-card'>", unsafe_allow_html=True)
        st.markdown("<h4 style='font-size: 15px; font-weight: 800; color: #000000;'>📈 Grade Distribution Breakdown</h4>", unsafe_allow_html=True)
        grade_counts = active_df["Grade"].value_counts().reset_index()
        grade_counts.columns = ["Grade", "Count"]
        fig_grade = px.bar(grade_counts, x="Grade", y="Count", text_auto=True, color="Grade", color_discrete_sequence=px.colors.qualitative.Set2)
        fig_grade.update_traces(textfont=dict(color="#000000", size=12))
        fig_grade.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=280, showlegend=False,
            xaxis=dict(tickfont=dict(color="#000000", size=12)),
            yaxis=dict(tickfont=dict(color="#000000", size=12), gridcolor="#cbd5e1")
        )
        st.plotly_chart(fig_grade, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ==============================================================================
# VIEW 4: AI MODEL VERIFICATION
# ==============================================================================
elif selected_section == "🤖 AI Model Verification":
    st.markdown("### 🤖 Academic Support Model Verification")
    st.markdown("Verification metrics for the Machine Learning model predicting students who need learning support.")
    
    m1, m2 = st.columns(2)
    with m1:
        st.markdown("<div class='student-card'>", unsafe_allow_html=True)
        st.markdown("<h4 style='font-size: 15px; font-weight: 800; color: #000000;'>🌲 Machine Learning Classification Accuracy</h4>", unsafe_allow_html=True)
        metrics_df = pd.DataFrame({
            "Academic Category": ["Good Standing (0)", "Needs Learning Support (1)"],
            "Precision Score": [rf_report['0']['precision'], rf_report['1']['precision']],
            "Recall Score": [rf_report['0']['recall'], rf_report['1']['recall']],
            "F1-Score": [rf_report['0']['f1-score'], rf_report['1']['f1-score']]
        })
        st.dataframe(metrics_df.style.format(precision=2), use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with m2:
        st.markdown("<div class='student-card'>", unsafe_allow_html=True)
        st.markdown("<h4 style='font-size: 15px; font-weight: 800; color: #000000;'>🔍 Most Important Performance Factors</h4>", unsafe_allow_html=True)
        feat_df = pd.DataFrame({
            "Academic Indicator": feature_cols,
            "Importance Weight": rf_model.feature_importances_
        }).sort_values(by="Importance Weight", ascending=True)
        
        fig_feat = px.bar(feat_df, x="Importance Weight", y="Academic Indicator", orientation="h", color_discrete_sequence=["#2563eb"])
        fig_feat.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=220,
            xaxis=dict(tickfont=dict(color="#000000", size=11), gridcolor="#cbd5e1"),
            yaxis=dict(tickfont=dict(color="#000000", size=11))
        )
        st.plotly_chart(fig_feat, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
