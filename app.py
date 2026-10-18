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

  # --- Deterministic Academic Risk Engine ---
  def compute_academic_risk(row):
    risk_points = 0
    if row["Attendance"] < 65:
      risk_points += 2
    if row["Final_Marks"] < 50:
      risk_points += 2
    if row["Assignment_Score"] < 50:
      risk_points += 1
    if row["Midterm_Marks"] < 50:
      risk_points +=
        
