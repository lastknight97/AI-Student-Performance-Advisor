import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Student Performance Advisor",
    page_icon="🎓",
    layout="wide",
)

# --- Custom CSS for Styling & Text Contrast ---
st.markdown("""
    <style>
        .main { background-color: #f8f9fa; }
        .metric-card {
            background-color: #ffffff;
            padding: 22px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.08);
            border-left: 6px solid #4f46e5;
            margin-bottom: 20px;
        }
        .metric-card h2, .metric-card p, .metric-card b {
            color: #1f2937 !important;
        }
    </style>
""", unsafe_allow_html=True)

# --- Data Loading & Deterministic Rule Engine ---
@st.cache_resource
def load_and_evaluate_data():
  possible_paths = [
      "data/Subject_Analytics_Dataset_200_Students.xlsx",
      "Subject_Analytics_Dataset_200_Students.xlsx"
  ]
  
  excel_path = None
  for path in possible_paths:
    if os.path.exists(path):
      excel_path = path
      break
      
  if not excel_path:
    st.error("⚠️ Excel dataset not found. Please ensure 'Subject_Analytics_Dataset_200_Students.xlsx' is uploaded in your repository.")
    st.stop()

  try:
    df = pd.read_excel(excel_path)
  except Exception as e:
    st.error(f"⚠️ Error reading Excel file: {e}")
    st.stop()

  # Clean column names and string spaces
  df.columns = df.columns.str.strip()
  for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].astype(str).str.strip()

  expected_cols = [
      "Student_ID", "Student_Name", "Department", "Semester", 
      "Subject", "Attendance", "Quiz_Score", "Assignment_Score", 
      "Midterm_Marks", "Final_Marks"
  ]

  missing_cols = [col for col in expected_cols if col not in df.columns]
  if missing_cols:
    st.error(f"❌ Missing columns in Excel file: {missing_cols}")
    st.stop()

  # --- Deterministic Academic Rule Engine ---
  def compute_academic_risk(row):
    risk_points = 0
    if row["Attendance"] < 65:
      risk_points += 2
    if row["Final_Marks"] < 50:
      risk_points += 2
    if row["Assignment_Score"] < 50:
      risk_points += 1
    if row["Midterm_Marks"] < 50:
      risk_points += 1
    
    return min(risk_points / 5.0, 1.0)

  df["Risk_Score"] = df.apply(compute_academic_risk, axis=1)
  
  if "performance_trend" not in df.columns:
    df["performance_trend"] = np.where(df["Final_Marks"] >= df["Midterm_Marks"], "improving", "declining")
  if "missed_assessments" not in df.columns:
    df["missed_assessments"] = np.where(df["Attendance"] < 75, 2, 0)

  return df

df = load_and_evaluate_data()

# --- Sidebar Navigation (Sections) ---
st.sidebar.title("🎓 Navigation")
st.sidebar.markdown("Select a dashboard section below:")
selected_section = st.sidebar.radio(
    "Go to", 
    ["📊 Overview", "👤 Student Profile", "📈 Analytics"]
)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Why Risk Indices?** Risk metrics help educators identify struggling students early for proactive remediation.")

# ==========================================
# SECTION 1: OVERVIEW
# ==========================================
if selected_section == "📊 Overview":
  st.title("🎓 AI Student Performance Advisor")
  st.markdown("Comprehensive academic overview and student performance monitoring.")
  st.markdown("---")
  
  st.subheader("Institutional Summary")
  
  col1, col2, col3, col4 = st.columns(4)
  total_students = df["Student_ID"].nunique()
  avg_attendance = df["Attendance"].mean()
  avg_final = df["Final_Marks"].mean()
  
  student_max_risk = df.groupby(["Student_ID", "Student_Name"])["Risk_Score"].max().reset_index()
  high_risk_count = int((student_max_risk["Risk_Score"] >= 0.5).sum())
  
  col1.metric("Active Students", f"{total_students}")
  col2.metric("Avg Attendance", f"{avg_attendance:.1f}%")
  col3.metric("Avg Final Score", f"{avg_final:.1f}")
  col4.metric("High Risk Students", f"{high_risk_count}", delta
