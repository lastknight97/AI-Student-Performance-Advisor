import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Students Performance Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- EXACT POWER BI THEME & STYLING CSS ---
st.markdown("""
    <style>
        .stApp {
            background-color: #f0f2f6 !important;
            font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
        }
        header {visibility: hidden;}
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
        .dashboard-card {
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 16px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            margin-bottom: 16px;
        }
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
    </style>
""", unsafe_allow_html=True)

# --- DATA INGESTION & COLUMN MAPPING ---
@st.cache_resource
def load_dashboard_data():
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
            "Gender": np.random.choice(["Female", "Male"], size=n, p=[0.5, 0.5]),
            "Department": np.random.choice(["Language", "STEM", "Humanities", "Physical Ed"], size=n),
            "Nationality": np.random.choice(["Indian", "Korean", "Italian", "Japanese", "French", "Canadian"], size=n),
            "Age": np.random.randint(14, 19, size=n),
            "Attendance": np.random.uniform(60, 100, size=n),
            "Final_Marks": np.random.uniform(40, 98, size=n),
            "Midterm_Marks": np.random.uniform(35, 95, size=n),
        })

    # Ensure required columns exist safely
    if "Semester" not in df.columns and "Year_Level" in df.columns:
        df["Semester"] = df["Year_Level"]
    elif "Semester" not in df.columns:
        df["Semester"] = "Semester 1"

    if "Gender" not in df.columns:
        df["Gender"] = "Female"

    if "Department" not in df.columns:
        df["Department"] = "STEM"

    if "Nationality" not in df.columns:
        df["Nationality"] = "Indian"

    if "Student_Name" not in df.columns:
        df["Student_Name"] = df["Student_ID"]

    return df

df = load_dashboard_data()

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
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    sb_col1, sb_col2 = st.columns(2)
    with sb_col1:
        if st.button("🔄", help="Reset Filters"):
            st.rerun()
    with sb_col2:
        st.button("ℹ️", help="Dashboard Information")

# --- FILTER DATAFRAME ---
filtered_df = df.copy()
if record_id != "All":
    filtered_df = filtered_df[filtered_df["Student_ID"] == record_id]
if gender_filter != "All":
    filtered_df = filtered_df[filtered_df["Gender"] == gender_filter]

# --- MAIN DASHBOARD AREA ---

# Top Navigation Header Row
st.markdown("""
    <div style="display: flex; justify-content: flex-end; gap: 20px; font-weight: 700; font-size: 13px; color: #4a5568; margin-bottom: 12px; border-bottom: 2px solid #cbd5e0; padding-bottom: 8px;">
        <span style="color: #2b6cb0; border-bottom: 2px solid #2b6cb0; padding-bottom: 6px;">OVERVIEW</span>
        <span>STUDENTS & TEACHERS</span>
        <span>ASSESSMENTS</span>
    </div>
""", unsafe_allow_html=True)

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

# --- ROW 1: Donut Chart & Top 3 Performing Students ---
row1_col1, row1_col2, row1_col3, row1_col4 = st.columns([1.3, 1, 1, 1])

with row1_col1:
    st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 12px; color: #718096; font-weight: 700; margin-bottom: 0;'>Pass % by Semester</p>", unsafe_allow_html=True)
    
    sem_pass = filtered_df.groupby("Semester")["Final_Marks"].apply(lambda x: (x >= 50).mean() * 100).reset_index(name="Pass_Rate")
    fig_donut = px.pie(
        sem_pass, names="Semester", values="Pass_Rate", hole=0.65,
        color_discrete_sequence=["#4c51bf", "#6366f1", "#818cf8", "#c7d2fe"]
    )
    fig_donut.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=190, margin=dict(t=10, b=10, l=10, r=10),
        showlegend=False
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
    else:
        s_name = s_data["name"]
        s_id = s_data["id"]
        s_tot = f"{s_data['tot']:.2f}"

    with col_ref:
        st.markdown(f"""
            <div class="rank-card">
                <p style="font-size: 11px; color: #718096; font-weight: 700; margin-bottom: 4px;">Top 3 Performing Students</p>
                <div style="background: #e2e8f0; width: 50px; height: 50px; border-radius: 50%; margin: 8px auto; display: flex; align-items: center; justify-content: center; font-size: 20px;">👤</div>
                <div style="font-weight: 800; font-size: 14px; color: #1a202c; margin-bottom: 8px;">{s_name}</div>
                <hr style="border: none; border-top: 1px solid #e2e8f0; margin: 6px 0;">
                <div style="font-size: 11px; text-align: left; color: #4a5568; line-height: 1.5;">
                    <div style="display: flex; justify-content: space-between;"><span>Student ID</span><b>{s_id}</b></div>
                    <div style="display: flex; justify-content: space-between;"><span>Age</span><b>15</b></div>
                    <div style="display: flex; justify-content: space-between;"><span>Nationality</span><b>{s_data['nat']}</b></div>
                    <div style="display: flex; justify-content: space-between;"><span>Total %</span><b>{s_tot}</b></div>
                </div>
                <div class="rank-banner">{s_data['rank']}</div>
            </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- ROW 2: Bar & Line Analytical Charts ---
row2_col1, row2_col2, row2_col3 = st.columns(3)

with row2_col1:
    st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 12px; color: #718096; font-weight: 700; margin-bottom: 4px;'>Average Weighted Score by Department</p>", unsafe_allow_html=True)
    
    dept_score = filtered_df.groupby("Department")["Final_Marks"].mean().reset_index()
    fig_dept = px.bar(
        dept_score, x="Department", y="Final_Marks",
        color_discrete_sequence=["#4c51bf"]
    )
    fig_dept.update_traces(marker_color="#4c51bf")
    fig_dept.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=220, margin=dict(t=10, b=10, l=10, r=10),
        xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor="#edf2f7")
    )
    st.plotly_chart(fig_dept, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with row2_col2:
    st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 12px; color: #718096; font-weight: 700; margin-bottom: 4px;'>Average Score by Nationality</p>", unsafe_allow_html=True)
    
    nat_score = filtered_df.groupby("Nationality")["Final_Marks"].mean().reset_index().sort_values(by="Final_Marks", ascending=True)
    fig_nat = px.bar(
        nat_score, x="Final_Marks", y="Nationality", orientation="h",
        color_discrete_sequence=["#4c51bf"]
    )
    fig_nat.update_traces(marker_color="#4c51bf")
    fig_nat.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=220, margin=dict(t=10, b=10, l=10, r=10),
        xaxis=dict(showgrid=True, gridcolor="#edf2f7"), yaxis=dict(showgrid=False)
    )
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
    
    fig_trend.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=220, margin=dict(t=10, b=10, l=10, r=10),
        showlegend=False, xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor="#edf2f7")
    )
    st.plotly_chart(fig_trend, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
