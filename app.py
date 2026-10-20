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
    page_title="AI Student Performance Advisor",
    page_icon="🎓",
    layout="wide",
)

# --- Custom CSS for Styling, Contrast & Round Floating Chat Widget ---
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
        /* Floating Chat Widget Container Styling */
        .floating-chat-container {
            position: fixed;
            bottom: 95px;
            right: 30px;
            width: 400px;
            height: 520px;
            background-color: white;
            border-radius: 12px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.2);
            z-index: 99999;
            padding: 15px;
            display: flex;
            flex-direction: column;
            border: 1px solid #e5e7eb;
        }
        /* Round Floating Action Button (FAB) for Peter */
        .floating-btn-wrapper {
            position: fixed;
            bottom: 25px;
            right: 25px;
            z-index: 999999;
        }
        div[data-testid="stButton"] button[kind="primary"] {
            border-radius: 50% !important;
            width: 60px !important;
            height: 60px !important;
            padding: 0px !important;
            background-color: #4f46e5 !important;
            color: white !important;
            font-size: 24px !important;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3) !important;
            border: none !important;
            display: flex;
            align-items: center;
            justify-content: center;
        }
    </style>
""", unsafe_allow_html=True)

# --- Data Loading & Machine Learning Workflow (Slide 6) ---
@st.cache_resource
def load_and_evaluate_data():
  excel_path = "data/Subject_Analytics_Dataset_200_Students.xlsx"
  
  if not os.path.exists(excel_path):
    st.error(f"⚠️ Excel dataset not found at '{excel_path}'. Please ensure the file is placed inside the 'data' folder.")
    st.stop()

  try:
    df = pd.read_excel(excel_path)
  except Exception as e:
    st.error(f"⚠️ Error reading Excel file: {e}")
    st.stop()

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

  feature_cols = ["Attendance", "Quiz_Score", "Assignment_Score", "Midterm_Marks", "Final_Marks"]
  X = df[feature_cols]

  composite_score = (X["Attendance"] * 0.3) + (X["Final_Marks"] * 0.4) + (X["Midterm_Marks"] * 0.3)
  
  def get_target_label(score):
    if score < 55:
      return "High"
    elif score < 70:
      return "Moderate"
    else:
      return "Low"

  y = composite_score.apply(get_target_label)

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  lr_baseline = LogisticRegression(random_state=42, max_iter=200)
  lr_baseline.fit(X_train, y_train)
  lr_pred = lr_baseline.predict(X_test)
  lr_report = classification_report(y_test, lr_pred, output_dict=True)

  rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
  rf_model.fit(X_train, y_train)
  rf_pred = rf_model.predict(X_test)
  
  rf_report = classification_report(y_test, rf_pred, output_dict=True)
  conf_matrix = confusion_matrix(y_test, rf_pred, labels=["High", "Moderate", "Low"])

  df["Risk_Level"] = rf_model.predict(X)
  
  class_labels = list(rf_model.classes_)
  probs = rf_model.predict_proba(X)
  high_idx = class_labels.index("High") if "High" in class_labels else 0
  df["Risk_Score"] = probs[:, high_idx] if "High" in class_labels else 0.0

  df["performance_trend"] = np.where(df["Final_Marks"] >= df["Midterm_Marks"], "Improving / Steady", "Declining")
  if "missed_assessments" not in df.columns:
    df["missed_assessments"] = np.where(df["Attendance"] < 75, 2, 0)

  return df, rf_model, rf_report, lr_report, conf_matrix

df, rf_model, rf_report, lr_report, conf_matrix = load_and_evaluate_data()

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

  st.markdown("---")
  st.subheader("📐 Model Evaluation Metrics")
  col_m1, col_m2 = st.columns(2)
  with col_m1:
    st.markdown("#### 🌲 Random Forest Classifier Metrics")
    rf_metrics_df = pd.DataFrame({
        "Class": ["High", "Moderate", "Low"],
        "Precision": [rf_report['High']['precision'], rf_report['Moderate']['precision'], rf_report['Low']['precision']],
        "Recall": [rf_report['High']['recall'], rf_report['Moderate']['recall'], rf_report['Low']['recall']],
        "F1-Score": [rf_report['High']['f1-score'], rf_report['Moderate']['f1-score'], rf_report['Low']['f1-score']]
    })
    st.dataframe(rf_metrics_df.style.format(precision=2), use_container_width=True)

  with col_m2:
    st.markdown("#### 📉 Logistic Regression Baseline Metrics")
    lr_metrics_df = pd.DataFrame({
        "Class": ["High", "Moderate", "Low"],
        "Precision": [lr_report['High']['precision'], lr_report['Moderate']['precision'], lr_report['Low']['precision']],
        "Recall": [lr_report['High']['recall'], lr_report['Moderate']['recall'], lr_report['Low']['recall']],
        "F1-Score": [lr_report['High']['f1-score'], lr_report['Moderate']['f1-score'], lr_report['Low']['f1-score']]
    })
    st.dataframe(lr_metrics_df.style.format(precision=2), use_container_width=True)

  st.markdown("#### 🗂️ Random Forest Confusion Matrix")
  cm_df = pd.DataFrame(
      conf_matrix, 
      index=["Actual High", "Actual Moderate", "Actual Low"], 
      columns=["Pred High", "Pred Moderate", "Pred Low"]
  )
  st.dataframe(cm_df, use_container_width=True)

  st.markdown("---")
  st.subheader("🔍 AI Model Insights: What the Model Weighs Most")
  feature_importances = pd.DataFrame({
      "Academic_Factor": ["Attendance", "Quiz_Score", "Assignment_Score", "Midterm_Marks", "Final_Marks"],
      "Importance": rf_model.feature_importances_
  }).sort_values(by="Importance", ascending=True)

  feature_labels = {
      "Attendance": "Attendance Rate",
      "Quiz_Score": "Quiz Scores",
      "Assignment_Score": "Assignment Scores",
      "Midterm_Marks": "Midterm Exam Marks",
      "Final_Marks": "Final Exam Marks"
  }
  feature_importances["Friendly_Name"] = feature_importances["Academic_Factor"].map(feature_labels)

  fig_importance = px.bar(
      feature_importances,
      x="Importance",
      y="Friendly_Name",
      orientation="h",
      title="Relative Importance of Academic Metrics in Risk Evaluation",
      color="Importance",
      color_continuous_scale="Blues"
  )
  fig_importance.update_layout(xaxis_title="Impact Weight", yaxis_title="Academic Metric")
  st.plotly_chart(fig_importance, use_container_width=True)


# ==========================================
# FLOATING CHATBOT WIDGET ("PETER")
# ==========================================

if "chat_open" not in st.session_state:
  st.session_state.chat_open = False

if "messages" not in st.session_state:
  st.session_state.messages = [
      {"role": "assistant", "content": "👋 Hi! I'm **Peter**, your AI Assistant. Ask me about any student record or type *'recommendations'* to fetch research-backed improvement steps!"}
  ]

# Render the round floating action button in the bottom right corner
with st.container():
  st.markdown('<div class="floating-btn-wrapper">', unsafe_allow_html=True)
  if st.button("💬", key="chat_toggle_btn", type="primary"):
    st.session_state.chat_open = not st.session_state.chat_open
    st.rerun()
  st.markdown('</div>', unsafe_allow_html=True)

# Render Peter's chat popup window if open
if st.session_state.chat_open:
  with st.container():
    st.markdown('<div class="floating-chat-container">', unsafe_allow_html=True)
    
    # Header & Close button
    ch1, ch2 = st.columns([4, 1])
    with ch1:
      st.markdown("**💬 Peter (AI Advisor)**")
    with ch2:
      if st.button("✖", key="close_chat"):
        st.session_state.chat_open = False
        st.rerun()
        
    st.markdown("---")
    
    # Chat history container
    chat_container = st.container(height=340)
    with chat_container:
      for message in st.session_state.messages:
        with st.chat_message(message["role"]):
          st.markdown(message["content"])

    # Chat input box inside Peter's window
    if prompt := st.chat_input("Ask Peter about student data or strategies...", key="floating_chat_input"):
      st.session_state.messages.append({"role": "user", "content": prompt})
      
      query_lower = prompt.lower()
      response = ""

      # Research-backed strategy handler
      if "recommend" in query_lower or "strategy" in query_lower or "improve" in query_lower or "step" in query_lower:
        response = "🌐 **Peter's Evidence-Based Strategies for At-Risk Learners:**\n\n"
        response += "1. **Targeted Tutoring & Peer Dyads:** Pair students with high-performing peers using Class-Wide Peer Tutoring (CWPT).\n"
        response += "2. **Differentiated Mastery Learning:** Offer flexible assignment completion pathways and formative temperature checks.\n"
        response += "3. **Metacognitive Goal Setting:** Encourage weekly self-evaluation logs to track test prep and deadlines.\n"
        response += "4. **Attendance Monitoring:** Establish weekly counselor check-ins for chronically absent students."

      # High risk summary handler
      elif "high" in query_lower and ("risk" in query_lower or "student" in query_lower or "status" in query_lower):
        high_risk_students = df[df["Risk_Level"] == "High"][["Student_ID", "Student_Name", "Department", "Subject"]].drop_duplicates().head(5)
        response = "🚨 **High Risk Students Summary:**\n\nHere are students currently flagged at **High** risk:\n"
        for idx, row in high_risk_students.iterrows():
          response += f"- **{row['Student_Name']}** (ID: {row['Student_ID']}, Dept: {row['Department']})\n"
        response += "\n*Peter's Recommended Steps:* Assign targeted tutoring, schedule counselor check-ins, and create individualized learning goals."

      # Specific student lookup handler
      else:
        found_student = None
        for _, row in df.drop_duplicates(subset=["Student_ID"]).iterrows():
          if str(row["Student_ID"]) in query_lower or row["Student_Name"].lower() in query_lower:
            found_student = row["Student_ID"]
            break
        
        if found_student:
          s_data = df[df["Student_ID"] == found_student]
          s_name = s_data["Student_Name"].iloc[0]
          s_dept = s_data["Department"].iloc[0]
          s_risk = s_data["Risk_Level"].iloc[0]
          s_att = s_data["Attendance"].mean()
          s_final = s_data["Final_Marks"].mean()

          response = f"👤 **Profile: {s_name} (ID: {found_student})**\n"
          response += f"- **Department:** {s_dept}\n- **Overall Status:** {s_risk} Risk\n- **Avg Attendance:** {s_att:.1f}%\n- **Avg Final Marks:** {s_final:.1f}\n\n"
          
          if s_risk == "High":
            response += "⚠️ **Peter's Tailored Plan:**\n- *Attendance:* Conduct counselor check-ins.\n- *Academics:* Assign peer-tutoring dyads and guided assignment practice."
          elif s_risk == "Moderate":
            response += "⚠️ **Peter's Tailored Plan:**\n- Provide worked examples for midterm topics and encourage weekly goal-setting."
          else:
            response += "✅ **Peter's Tailored Plan:**\n- Performing well! Maintain current habits and offer optional enrichment tasks."
        else:
          response = "I'm Peter! I can look up student records and provide research-backed strategies. Try typing a student's name, ID, or ask *" + '"What are recommended steps for high risk students?"*'

      st.session_state.messages.append({"role": "assistant", "content": response})
      st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)
