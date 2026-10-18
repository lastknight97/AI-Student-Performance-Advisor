import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Student Performance CRM & Advisor",
    page_icon="🎓",
    layout="wide",
)

# --- Custom CSS for CRM Styling ---
st.markdown("""
    <style>
        .main { background-color: #f8f9fa; }
        .metric-card {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            border-left: 5px solid #4f46e5;
        }
    </style>
""", unsafe_allow_html=True)

# --- Data Loading & Preprocessing Pipeline ---
@st.cache_resource
def load_data_and_train_model():
  excel_path = "data/Subject_Analytics_Dataset_200_Students.xlsx"
  
  if not os.path.exists(excel_path):
    st.error(f"⚠️ File not found at: {excel_path}. Please check your folder structure.")
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

  # Dynamic Risk Label & Proxy Feature Generation
  if "risk" not in df.columns:
    df["risk"] = (
        (df["Final_Marks"] < 50) | 
        (df["Attendance"] < 65) | 
        (df["Assignment_Score"] < 50)
    ).astype(int)

  if "previous_avg" not in df.columns:
    df["previous_avg"] = (df["Midterm_Marks"] + df["Quiz_Score"]) / 2
  if "recent_avg" not in df.columns:
    df["recent_avg"] = (df["Final_Marks"] + df["Assignment_Score"]) / 2
  if "performance_trend" not in df.columns:
    df["performance_trend"] = np.where(df["Final_Marks"] >= df["Midterm_Marks"], "improving", "declining")
  if "missed_assessments" not in df.columns:
    df["missed_assessments"] = np.where(df["Attendance"] < 75, 2, 0)

  feature_cols = [
      "Attendance", "Assignment_Score", "Quiz_Score", "Midterm_Marks", 
      "Final_Marks", "previous_avg", "recent_avg", "missed_assessments", 
      "Subject", "Department"
  ]
  
  X = df[feature_cols]
  y = df["risk"]

  numeric_features = [
      "Attendance", "Assignment_Score", "Quiz_Score", "Midterm_Marks", 
      "Final_Marks", "previous_avg", "recent_avg", "missed_assessments"
  ]
  categorical_features = ["Subject", "Department"]

  preprocessor = ColumnTransformer(
      transformers=[
          ("num", StandardScaler(), numeric_features),
          ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
      ]
  )

  pipeline = Pipeline(
      steps=[
          ("preprocessor", preprocessor),
          ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
      ]
  )
  
  pipeline.fit(X, y)
  
  # Compute Risk Probabilities for all rows
  df["Risk_Probability"] = pipeline.predict_proba(X)[:, 1]
  return df, pipeline

# Load data and trained model
df, model = load_data_and_train_model()

# --- Top Navigation Bar / Header ---
st.title("🎓 AI Student Performance CRM & Advisor")
st.markdown("Enterprise-grade student telemetry, predictive risk scoring, and automated intervention tracking.")
st.markdown("---")

# Main CRM Tab Layout
tab1, tab2, tab3 = st.tabs([
    "📊 Executive Overview", 
    "👤 Student 360 Profile", 
    "📈 Department & Subject Analytics"
])

# ==========================================
# TAB 1: EXECUTIVE OVERVIEW (CRM Home)
# ==========================================
with tab1:
  st.subheader("Institutional Health & Risk Summary")
  
  col1, col2, col3, col4 = st.columns(4)
  total_students = df["Student_ID"].nunique()
  avg_attendance = df["Attendance"].mean()
  avg_final = df["Final_Marks"].mean()
  
  student_max_risk = df.groupby(["Student_ID", "Student_Name"])["Risk_Probability"].max().reset_index()
  high_risk_count = (student_max_risk["Risk_Probability"] >= 0.5).sum()
  
  col1.metric("Total Active Students", f"{total_students}")
  col2.metric("Average Institutional Attendance", f"{avg_attendance:.1f}%")
  col3.metric("Average Final Score", f"{avg_final:.1f}")
  col4.metric("Students at High Risk", f"{high_risk_count}", delta=f"-{high_risk_count} urgent", delta_color="inverse")
  
  st.markdown("---")
  
  c1, c2 = st.columns(2)
  
  with c1:
    st.markdown("#### Risk Distribution Breakdown")
    risk_df = student_max_risk.copy()
    risk_df["Status"] = np.where(risk_df["Risk_Probability"] >= 0.5, "At Risk", "Stable")
    status_counts = risk_df["Status"].value_counts().reset_index()
    status_counts.columns = ["Status", "Count"]
    
    fig_pie = px.pie(
        status_counts, 
        names="Status", 

      
