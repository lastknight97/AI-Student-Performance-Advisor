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

# --- Data Loading & Preprocessing Pipeline ---
@st.cache_resource
def load_data_and_train_model():
  # Check multiple potential paths for the Excel file
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

# --- Top Header ---
st.title("🎓 AI Student Performance Advisor")
st.markdown("Student telemetry tracking, predictive risk scoring, and automated intervention guidance.")
st.markdown("---")

# Brief, Clean Tab Layout
tab1, tab2, tab3 = st.tabs([
    "📊 Overview", 
    "👤 Student Profile", 
    "📈 Analytics"
])

# ==========================================
# TAB 1: OVERVIEW
# ==========================================
with tab1:
  st.subheader("Institutional Summary")
  
  col1, col2, col3, col4 = st.columns(4)
  total_students = df["Student_ID"].nunique()
  avg_attendance = df["Attendance"].mean()
  avg_final = df["Final_Marks"].mean()
  
  student_max_risk = df.groupby(["Student_ID", "Student_Name"])["Risk_Probability"].max().reset_index()
  high_risk_count = int((student_max_risk["Risk_Probability"] >= 0.5).sum())
  
  col1.metric("Active Students", f"{total_students}")
  col2.metric("Avg Attendance", f"{avg_attendance:.1f}%")
  col3.metric("Avg Final Score", f"{avg_final:.1f}")
  col4.metric("High Risk Students", f"{high_risk_count}", delta=f"-{high_risk_count} urgent", delta_color="inverse")
  
  st.markdown("---")
  
  c1, c2 = st.columns(2)
  
  with c1:
    st.markdown("#### Risk Distribution")
    risk_df = student_max_risk.copy()
    risk_df["Status"] = np.where(risk_df["Risk_Probability"] >= 0.5, "At Risk", "Stable")
    status_counts = risk_df["Status"].value_counts().reset_index()
    status_counts.columns = ["Status", "Count"]
    
    fig_pie = px.pie(
        status_counts, 
        names="Status", 
        values="Count", 
        hole=0.4,
        color="Status",
        color_discrete_map={"At Risk": "#ef4444", "Stable": "#10b981"}
    )
    st.plotly_chart(fig_pie, use_container_width=True)
    
  with c2:
    st.markdown("#### Risk Count by Department")
    dept_risk = df.copy()
    dept_risk["Is_At_Risk"] = dept_risk["Risk_Probability"] >= 0.5
    dept_summary = dept_risk.groupby("Department")["Is_At_Risk"].sum().reset_index()
    
    fig_bar = px.bar(
        dept_summary,
        x="Department",
        y="Is_At_Risk",
        text="Is_At_Risk",
        color="Department",
        title="At-Risk Records per Department"
    )
    st.plotly_chart(fig_bar, use_container_width=True)

  st.markdown("---")
  st.markdown("### 🚨 Urgent Action Queue")
  
  top_risk_table = df[df["Risk_Probability"] >= 0.5][[
      "Student_ID", "Student_Name", "Department", "Subject", "Attendance", "Final_Marks", "Risk_Probability"
  ]].sort_values(by="Risk_Probability", ascending=False).head(10)
  
  top_risk_table["Risk Score (%)"] = (top_risk_table["Risk_Probability"] * 100).round(1)
  top_risk_table = top_risk_table.drop(columns=["Risk_Probability"])
  
  st.dataframe(top_risk_table, use_container_width=True)

# ==========================================
# TAB 2: STUDENT PROFILE
# ==========================================
with tab2:
  st.sidebar.header("🔍 Student Lookup")
  
  student_options = df[["Student_ID", "Student_Name"]].drop_duplicates().sort_values(by="Student_ID")
  student_options["Display_Label"] = student_options["Student_ID"].astype(str) + " - " + student_options["Student_Name"].astype(str)
  
  selected_display = st.sidebar.selectbox(
      "Select Student Profile", 
      options=student_options["Display_Label"].tolist()
  )
  
  selected_student_id = selected_display.split(" - ")[0]

  student_df = df[df["Student_ID"] == selected_student_id]
  student_name = student_df["Student_Name"].iloc[0]
  student_department = student_df["Department"].iloc[0]
  student_semester = student_df["Semester"].iloc[0]
  courses_chosen = ", ".join(student_df["Subject"].tolist())

  st.markdown(f"""
      <div class='metric-card'>
          <h2 style='color: #1f2937 !important; margin-bottom: 8px;'>👤 Profile: {student_name}</h2>
          <p style='color: #4b5563 !important; font-size: 16px; margin: 4px 0;'><b>Student ID:</b> {selected_student_id} &nbsp;|&nbsp; <b>Department:</b> {student_department} &nbsp;|&nbsp; <b>Semester:</b> {student_semester}</p>
          <p style='color: #4b5563 !important; font-size: 16px; margin: 4px 0;'><b>Courses Chosen:</b> {courses_chosen}</p>
      </div>
  """, unsafe_allow_html=True)
  st.markdown("")

  m1, m2, m3, m4 = st.columns(4)
  p_attendance = student_df["Attendance"].mean()
  p_final = student_df["Final_Marks"].mean()
  p_missed = student_df["missed_assessments"].sum()
  p_max_risk = float(student_df["Risk_Probability"].max())
  
  m1.metric("Attendance Rate", f"{p_attendance:.1f}%")
  m2.metric("Mean Final Marks", f"{p_final:.1f}")
  m3.metric("Est. Missed Tasks", int(p_missed))
  m4.metric("Max Risk Factor", f"{p_max_risk:.2f}", delta="At Risk" if p_max_risk >= 0.5 else "Stable", delta_color="inverse")

  st.markdown("---")
  st.subheader("📚 Subject Performance Matrix")
  
  display_df = student_df[[
      "Subject", "Attendance", "Quiz_Score", "Assignment_Score", 
      "Midterm_Marks", "Final_Marks", "performance_trend", "Risk_Probability"
  ]].copy()
  display_df["Risk Score"] = display_df["Risk_Probability"].apply(lambda x: f"{x:.2f}")
  display_df = display_df.drop(columns=["Risk_Probability"]).sort_values(by="Final_Marks", ascending=True)

  st.dataframe(display_df, use_container_width=True)

  sc1, sc2 = st.columns(2)
  with sc1:
    st.markdown("#### Subject Score Breakdown")
    fig_radar = px.bar(
        student_df,
        x="Subject",
        y=["Quiz_Score", "Assignment_Score", "Midterm_Marks", "Final_Marks"],
        barmode="group",
        title="Evaluation Metrics per Subject"
    )
    st.plotly_chart(fig_radar, use_container_width=True)

  with sc2:
    st.markdown("#### Attendance vs Final Score")
    fig_scc = px.scatter(
        student_df,
        x="Attendance",
        y="Final_Marks",
        color="performance_trend",
        size="Assignment_Score",
        hover_data=["Subject"],
        title="Performance Trajectory"
    )
    st.plotly_chart(fig_scc, use_container_width=True)

  st.markdown("---")
  st.subheader("💡 AI Advisor Intervention Notes")

  def generate_recommendation(row, prob):
    notes = []
    subj = row["Subject"]
    if prob >= 0.5:
      notes.append(f"⚠️ **{subj} requires urgent attention** (Risk Score: {prob:.2f}).")
    else:
      notes.append(f"✅ **{subj} is performing stably** (Risk Score: {prob:.2f}).")
    
    if row["Final_Marks"] < 50:
      notes.append(f"Final marks are low ({row['Final_Marks']}); recommend mandatory tutoring.")
    if row["Attendance"] < 75:
      notes.append(f"Attendance is trailing at {row['Attendance']}%.")
    return " ".join(notes)

  for idx, row in student_df.iterrows():
    prob = row["Risk_Probability"]
    with st.expander(f"Intervention Dossier: {row['Subject']} (Risk Index: {prob:.2f})"):
      st.markdown(generate_recommendation(row, prob))

# ==========================================
# TAB 3: ANALYTICS
# ==========================================
with tab3:
  st.subheader("Comparative Institutional Analytics")
  
  d1, d2 = st.columns(2)
  
  with d1:
    st.markdown("#### Department Final Marks Distribution")
    fig_box = px.box(
        df, 
        x="Department", 
        y="Final_Marks", 
        color="Department",
        title="Marks Distribution Across Departments"
    )
    st.plotly_chart(fig_box, use_container_width=True)
    
  with d2:
    st.markdown("#### Average Final Marks by Subject")
    subj_avg = df.groupby("Subject")["Final_Marks"].mean().reset_index().sort_values(by="Final_Marks")
    fig_subj = px.bar(
        subj_avg,
        x="Subject",
        y="Final_Marks",
        color="Final_Marks",
        color_continuous_scale="Viridis",
        title="Average Score per Subject"
    )
    st.plotly_chart(fig_subj, use_container_width=True)

  st.markdown("---")
  st.subheader("🔍 Machine Learning Feature Drivers")
  
  rf_model = model.named_steps["classifier"]
  preprocessor = model.named_steps["preprocessor"]

  cat_encoder = preprocessor.named_transformers_["cat"]
  encoded_cat_features = list(
      cat_encoder.get_feature_names_out(["Subject", "Department"])
  )
  all_features = [
      "Attendance", "Assignment_Score", "Quiz_Score", "Midterm_Marks", 
      "Final_Marks", "previous_avg", "recent_avg", "missed_assessments"
  ] + encoded_cat_features

  importances = rf_model.feature_importances_
  feat_imp_df = pd.DataFrame(
      {"Feature": all_features, "Importance": importances}
  ).sort_values(by="Importance", ascending=False)

  fig_imp = px.bar(
      feat_imp_df.head(10),
      x="Importance",
      y="Feature",
      orientation="h",
      title="Top 10 Feature Importances Driving Risk Engine",
  )
  fig_imp.update_layout(yaxis={"categoryorder": "total ascending"})
  st.plotly_chart(fig_imp, use_container_width=True)
    
