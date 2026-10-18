import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

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

# --- Data Loading & Machine Learning Risk Engine ---
@st.cache_resource
def load_and_evaluate_data():
  excel_path = "Subject_Analytics_Dataset_200_Students.xlsx"
  
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

  # --- Machine Learning Models (Logistic Regression Baseline + Random Forest) ---
  feature_cols = ["Attendance", "Quiz_Score", "Assignment_Score", "Midterm_Marks", "Final_Marks"]
  X = df[feature_cols]

  # Generate synthetic training targets based on overall academic performance distribution for ML training
  # (This serves as the target variable for our supervised models)
  composite_score = (X["Attendance"] * 0.3) + (X["Final_Marks"] * 0.4) + (X["Midterm_Marks"] * 0.3)
  
  def get_target_label(score):
    if score < 55:
      return "High"
    elif score < 70:
      return "Moderate"
    else:
      return "Low"

  y = composite_score.apply(get_target_label)

  # 1. Baseline Model: Logistic Regression
  lr_baseline = LogisticRegression(random_state=42, max_iter=200)
  lr_baseline.fit(X, y)
  df["Baseline_Pred"] = lr_baseline.predict(X)

  # 2. Main Model: Random Forest Classifier
  rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
  rf_model.fit(X, y)

  # Predict risk categories and probabilities using the Random Forest model
  df["Risk_Level"] = rf_model.predict(X)
  
  class_labels = list(rf_model.classes_)
  probs = rf_model.predict_proba(X)
  high_idx = class_labels.index("High") if "High" in class_labels else 0
  df["Risk_Score"] = probs[:, high_idx] if "High" in class_labels else 0.0

  df["performance_trend"] = np.where(df["Final_Marks"] >= df["Midterm_Marks"], "Improving / Steady", "Declining")
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

# ==========================================
# SECTION 1: OVERVIEW
# ==========================================
if selected_section == "📊 Overview":
  st.title("🎓 AI Student Performance Advisor")
  st.markdown("Comprehensive academic overview and machine learning risk monitoring.")
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
  col4.metric("High Status Students", f"{high_risk_count}", delta=f"-{high_risk_count} urgent", delta_color="inverse")
  
  st.markdown("---")
  
  c1, c2 = st.columns(2)
  
  with c1:
    st.markdown("#### Risk Distribution (Random Forest Model)")
    risk_df = student_max_risk.copy()
    risk_df["Status"] = np.where(risk_df["Risk_Score"] >= 0.5, "At Risk", "Stable")
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
    dept_risk["Is_At_Risk"] = dept_risk["Risk_Score"] >= 0.5
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
  st.markdown("### 🚨 Urgent Action Queue (ML Predicted)")
  
  top_risk_table = df[df["Risk_Level"] == "High"][[
      "Student_ID", "Student_Name", "Department", "Subject", "Attendance", "Final_Marks", "Risk_Level"
  ]].sort_values(by="Attendance", ascending=True).head(10)
  
  top_risk_table = top_risk_table.rename(columns={"Risk_Level": "Status"})
  st.dataframe(top_risk_table, use_container_width=True)

# ==========================================
# SECTION 2: STUDENT PROFILE (Lookup in Main Body)
# ==========================================
elif selected_section == "👤 Student Profile":
  st.title("👤 Student Profile")
  st.markdown("Comprehensive academic overview and machine learning risk monitoring.")
  st.markdown("---")
  
  st.markdown("### 🔍 Student Lookup")
  student_options = df[["Student_ID", "Student_Name"]].drop_duplicates().sort_values(by="Student_ID")
  student_options["Display_Label"] = student_options["Student_ID"].astype(str) + " - " + student_options["Student_Name"].astype(str)
  
  selected_display = st.selectbox(
      "Search or Select Student", 
      options=student_options["Display_Label"].tolist()
  )
  st.markdown("---")
  
  selected_student_id = selected_display.split(" - ")[0]

  student_df = df[df["Student_ID"] == selected_student_id]
  student_name = student_df["Student_Name"].iloc[0]
  student_department = student_df["Department"].iloc[0]
  student_semester = student_df["Semester"].iloc[0]
  courses_chosen = ", ".join(student_df["Subject"].tolist())

  st.markdown(f"""
      <div class='metric-card'>
          <h2 style='color: #1f2937 !important; margin-bottom: 8px;'>Profile: {student_name}</h2>
          <p style='color: #4b5563 !important; font-size: 16px; margin: 4px 0;'><b>Student ID:</b> {selected_student_id} &nbsp;|&nbsp; <b>Department:</b> {student_department} &nbsp;|&nbsp; <b>Semester:</b> {student_semester}</p>
          <p style='color: #4b5563 !important; font-size: 16px; margin: 4px 0;'><b>Courses Chosen:</b> {courses_chosen}</p>
      </div>
  """, unsafe_allow_html=True)
  st.markdown("")

  m1, m2, m3, m4 = st.columns(4)
  p_attendance = student_df["Attendance"].mean()
  p_final = student_df["Final_Marks"].mean()
  p_missed = student_df["missed_assessments"].sum()
  p_max_risk_label = student_df["Risk_Level"].iloc[0]
  
  m1.metric("Attendance Rate", f"{p_attendance:.1f}%")
  m2.metric("Mean Final Marks", f"{p_final:.1f}")
  m3.metric("Est. Missed Tasks", int(p_missed))
  m4.metric("Overall Academic Status", p_max_risk_label, delta="Needs Attention" if p_max_risk_label == "High" else "Stable", delta_color="inverse")

  st.markdown("---")
  st.subheader("📚 Subject Performance Matrix")
  
  display_df = student_df[[
      "Subject", "Attendance", "Quiz_Score", "Assignment_Score", 
      "Midterm_Marks", "Final_Marks", "performance_trend", "Risk_Level"
  ]].copy()
  display_df = display_df.rename(columns={"Risk_Level": "Risk Status", "performance_trend": "Progress Direction"}).sort_values(by="Final_Marks", ascending=True)

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
    st.markdown("#### Midterm vs. Final Marks Comparison")
    fig_comp_student = px.bar(
        student_df,
        x="Subject",
        y=["Midterm_Marks", "Final_Marks"],
        barmode="group",
        title="Exam Progress: Midterm vs Final Marks",
        color_discrete_map={"Midterm_Marks": "#3b82f6", "Final_Marks": "#10b981"}
    )
    st.plotly_chart(fig_comp_student, use_container_width=True)

  st.markdown("---")
  st.subheader("💡 AI Advisor Intervention Notes")

  def generate_recommendation(row, risk_level):
    notes = []
    subj = row["Subject"]
    if risk_level == "High":
      notes.append(f"⚠️ **{subj} requires urgent attention** (ML Status: High).")
    elif risk_level == "Moderate":
      notes.append(f"⚠️ **{subj} needs monitoring** (ML Status: Moderate).")
    else:
      notes.append(f"✅ **{subj} is performing well** (ML Status: Low).")
    
    if row["Final_Marks"] < 50:
      notes.append(f"Final marks are low ({row['Final_Marks']}); recommend mandatory tutoring.")
    if row["Attendance"] < 75:
      notes.append(f"Attendance is trailing at {row['Attendance']}%.")
    return " ".join(notes)

  for idx, row in student_df.iterrows():
    risk_lvl = row["Risk_Level"]
    with st.expander(f"Intervention Plan: {row['Subject']} (Status: {risk_lvl})"):
      st.markdown(generate_recommendation(row, risk_lvl))

# ==========================================
# SECTION 3: ANALYTICS
# ==========================================
elif selected_section == "📈 Analytics":
  st.title("📈 Institutional Analytics")
  st.markdown("Comprehensive academic overview and machine learning risk monitoring.")
  st.markdown("---")
  
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
  st.subheader("📊 What Drives Academic Risk? (Average Comparison)")
  st.markdown("Comparing the average scores of stable students versus those flagged at risk by the ML model.")

  df["Status"] = np.where(df["Risk_Level"] == "High", "At Risk", "Stable")
  comparison_df = df[["Attendance", "Quiz_Score", "Assignment_Score", "Midterm_Marks", "Final_Marks", "Status"]].groupby("Status").mean().reset_index()
  
  melted_comp = comparison_df.melt(id_vars="Status", var_name="Metric", value_name="Average_Score")

  fig_comp = px.bar(
      melted_comp,
      x="Metric",
      y="Average_Score",
      color="Status",
      barmode="group",
      title="Average Performance: Stable vs. At-Risk Students",
      color_discrete_map={"At Risk": "#ef4444", "Stable": "#10b981"}
  )
  st.plotly_chart(fig_comp, use_container_width=True)
    
