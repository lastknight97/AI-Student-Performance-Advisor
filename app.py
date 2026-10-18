import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# --- Page Configuration ---
st.set_page_config(
    page_title="AI-Based Student Performance Advisor",
    page_icon="🎓",
    layout="wide",
)

# --- Data Loading & Preprocessing Pipeline ---
@st.cache_resource
def load_data_and_train_model():
  excel_path = "data/Subject_Analytics_Dataset_200_Students.xlsx"
  
  if os.path.exists(excel_path):
    df = pd.read_excel(excel_path)
  else:
    st.error(
        f"⚠️ '{excel_path}' not found in the repository. Please verify that the "
        "file exists in the 'data/' folder."
    )
    st.stop()

  # Clean column names (strip trailing/leading whitespaces)
  df.columns = df.columns.str.strip()

  # Expected columns based on user input:
  # Student_ID, Student_Name, Department, Semester, Subject, Attendance, Quiz_Score, Assignment_Score, Midterm_Marks, Final_Marks
  expected_cols = [
      "Student_ID",
      "Student_Name",
      "Department",
      "Semester",
      "Subject",
      "Attendance",
      "Quiz_Score",
      "Assignment_Score",
      "Midterm_Marks",
      "Final_Marks",
  ]

  missing_cols = [col for col in expected_cols if col not in df.columns]
  if missing_cols:
    st.error(
        f"❌ Missing columns in Excel file: {missing_cols}\n\n"
        f"Your Excel columns are: {list(df.columns)}"
    )
    st.stop()

  # --- Feature Engineering & Dynamic Risk Label Generation ---
  # If a 'risk' label doesn't exist, we derive it from Final_Marks, Attendance, and Assignment_Score
  if "risk" not in df.columns:
    df["risk"] = (
        (df["Final_Marks"] < 50) | 
        (df["Attendance"] < 65) | 
        (df["Assignment_Score"] < 50)
    ).astype(int)

  # Create proxy features for historical/trend data if not present in columns
  if "previous_avg" not in df.columns:
    df["previous_avg"] = (df["Midterm_Marks"] + df["Quiz_Score"]) / 2
  if "recent_avg" not in df.columns:
    df["recent_avg"] = (df["Final_Marks"] + df["Assignment_Score"]) / 2
  if "performance_trend" not in df.columns:
    df["performance_trend"] = np.where(df["Final_Marks"] >= df["Midterm_Marks"], "improving", "declining")
  if "missed_assessments" not in df.columns:
    df["missed_assessments"] = np.where(df["Attendance"] < 75, 2, 0)

  # Define Features for Machine Learning Model
  feature_cols = [
      "Attendance",
      "Assignment_Score",
      "Quiz_Score",
      "Midterm_Marks",
      "Final_Marks",
      "previous_avg",
      "recent_avg",
      "missed_assessments",
      "Subject",
      "Department",
  ]
  
  X = df[feature_cols]
  y = df["risk"]

  # Preprocessing for Numeric & Categorical Columns
  numeric_features = [
      "Attendance",
      "Assignment_Score",
      "Quiz_Score",
      "Midterm_Marks",
      "Final_Marks",
      "previous_avg",
      "recent_avg",
      "missed_assessments",
  ]
  categorical_features = ["Subject", "Department"]

  preprocessor = ColumnTransformer(
      transformers=[
          ("num", StandardScaler(), numeric_features),
          ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
      ]
  )

  # Primary Machine Learning Pipeline: Random Forest Classifier
  pipeline = Pipeline(
      steps=[
          ("preprocessor", preprocessor),
          ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
      ]
  )
  
  pipeline.fit(X, y)
  return df, pipeline

# Load data and trained model
df, model = load_data_and_train_model()

# --- Sidebar Navigation & Student Selection ---
st.sidebar.title("🎓 Student Advisor Dashboard")
st.sidebar.markdown(
    "AI-driven academic risk assessment, telemetry tracking, and personalized interventions."
)

#
