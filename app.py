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
    page_title="Trace Academy - Students Performance Overview",
    page_icon="🎓",
    layout="wide",
)

# --- Modern DataFlip Purple Theme CSS ---
st.markdown("""
    <style>
        /* Base Page Background */
        .main { background-color: #f3f0f7; }
        
        /* Card Container Styling */
        .desk-card {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 14px;
            box-shadow: 0 4px 12px rgba(110, 68, 255, 0.08);
            border: 1px solid #e9e1f5;
            margin-bottom: 20px;
        }
        
        /* Profile Details Badge Grid */
        .profile-badge {
            background-color: #f7f4fb;
            border-radius: 8px;
            padding: 10px 14px;
            margin-bottom: 8px;
            border: 1px solid #ebe2f7;
        }
        .profile-badge label {
            font-size: 11px;
            color: #7e6e9b;
            text-transform: uppercase;
            font-weight: 700;
            display: block;
        }
        .profile-badge span {
            font-size: 15px;
            color: #2d1854;
            font-weight: 700;
        }
        
        /* Table / Pill styling */
        .status-pill {
            background-color: #f0ebf8;
            color: #6e44ff;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
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

# --- Top Header & Brand Bar ---
header_col1, header_col2 = st.columns([2, 3])
with header_col1:
  st.title("🎓 DataFlip | TRACE ACADEMY")
  st.caption("STUDENTS PERFORMANCE OVERVIEW")

with header_col2:
  # Global Student Selector
  student_options = df[["Student_ID", "Student_Name"]].drop_duplicates().sort_values(by="Student_ID")
  student_options["Label"] = student_options["Student_ID"].astype(str) + " - " + student_options["Student_Name"].astype(str)
  
  selected_student_label = st.selectbox("Select Student", options=student_options["Label"].tolist(), index=0)
  selected_student_id = selected_student_label.split(" - ")[0]

# --- Sidebar Navigation & Trained Peter Assistant ---
st.sidebar.title("📌 Navigation")
selected_section = st.sidebar.radio(
    "Go to", 
    ["📊 Performance Overview", "👤 Student Desk", "📈 Institutional Analytics"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("🤖 Peter (AI Advisor)")

if "messages" not in st.session_state:
  st.session_state.messages = [
      {"role": "assistant", "content": "👋 I'm **Peter**! Ask me about any student ID, department risk, or interventions."}
  ]

# Sidebar Chat Log
with st.sidebar.container(height=220):
  for message in st.session_state.messages:
    with st.chat_message(message["role"]):
      st.markdown(message["content"])

# Sidebar Form Input
with st.sidebar.form(key="peter_sidebar_form", clear_on_submit=True):
  user_text = st.text_input("Ask Peter...", placeholder="Name, ID, or recommendations...", label_visibility="collapsed")
  submitted = st.form_submit_button("Ask Peter", use_container_width=True)

  if submitted and user_text:
    st.session_state.messages.append({"role": "user", "content": user_text})
    query_lower = user_text.lower()
    
    if any(k in query_lower for k in ["recommend", "strategy", "improve", "step", "action"]):
      response = "🌐 **Peter's Trained Intervention Strategies:**\n1. **Attendance (<75%):** Issue alert to counselor.\n2. **Marks (<50):** Assign peer tutoring (CWPT).\n3. **Goal Logs:** Conduct weekly preparation checks."
    elif any(k in query_lower for k in ["dept", "department", "ece", "cse", "aiml", "it"]):
      dept_counts = df[df["Risk_Level"] == "High"]["Department"].value_counts()
      response = "📊 **High Risk Students per Department:**\n"
      for d_name, count in dept_counts.items():
        response += f"- **{d_name}:** {count} students\n"
    else:
      found_student = None
      for _, row in df.drop_duplicates(subset=["Student_ID"]).iterrows():
        if str(row["Student_ID"]).lower() in query_lower or row["Student_Name"].lower() in query_lower:
          found_student = row["Student_ID"]
          break
      
      if found_student:
        s_data = df[df["Student_ID"] == found_student]
        s_name = s_data["Student_Name"].iloc[0]
        s_risk = s_data["Risk_Level"].iloc[0]
        s_att = s_data["Attendance"].mean()
        s_final = s_data["Final_Marks"].mean()
        response = f"👤 **Profile: {s_name} ({found_student})**\n- **Risk:** {s_risk}\n- **Attendance:** {s_att:.1f}%\n- **Avg Final Marks:** {s_final:.1f}"
      else:
        response = "I couldn't locate that student. Try searching for a Student ID (e.g., S029, S094)."

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()


# ==========================================
# SECTION 1: PERFORMANCE OVERVIEW (DataFlip Dashboard)
# ==========================================
if selected_section == "📊 Performance Overview":
  student_df = df[df["Student_ID"] == selected_student_id]
  student_name = student_df["Student_Name"].iloc[0]
  student_department = student_df["Department"].iloc[0]
  student_semester = student_df["Semester"].iloc[0]
  
  avg_score = student_df["Final_Marks"].mean()
  pass_rate = (student_df["Final_Marks"] >= 50).mean() * 100
  overall_grade = "A" if avg_score >= 75 else "B" if avg_score >= 60 else "C" if avg_score >= 50 else "F"

  # Grid Layout: Left (Student Desk) | Right (Department / Institutional Stats)
  left_col, right_col = st.columns([3, 2])

  with left_col:
    st.markdown("### STUDENTS PERFORMANCE DESK")
    
    # Profile Card Grid
    p_col1, p_col2, p_col3 = st.columns([1, 2, 2])
    with p_col1:
      # Avatar / Placeholder Image
      st.image("https://api.dicebear.com/7.x/avataaars/svg?seed=" + student_name, width=120)
    
    with p_col2:
      st.markdown(f"<div class='profile-badge'><label>Full Name</label><span>{student_name}</span></div>", unsafe_allow_html=True)
      st.markdown(f"<div class='profile-badge'><label>Level / Semester</label><span>Semester {student_semester}</span></div>", unsafe_allow_html=True)
      st.markdown(f"<div class='profile-badge'><label>Records</label><span>{len(student_df)} Subjects</span></div>", unsafe_allow_html=True)
      st.markdown(f"<div class='profile-badge'><label>Pass Rate</label><span>{pass_rate:.1f}%</span></div>", unsafe_allow_html=True)

    with p_col3:
      st.markdown(f"<div class='profile-badge'><label>Department</label><span>{student_department}</span></div>", unsafe_allow_html=True)
      st.markdown(f"<div class='profile-badge'><label>Student ID</label><span>{selected_student_id}</span></div>", unsafe_allow_html=True)
      st.markdown(f"<div class='profile-badge'><label>Avg Score</label><span>{avg_score:.2f}</span></div>", unsafe_allow_html=True)
      st.markdown(f"<div class='profile-badge'><label>Final Grade</label><span>{overall_grade}</span></div>", unsafe_allow_html=True)

    st.markdown("---")
    
    # Tabbed Assessment Desk (Like DataFlip mockup)
    tab_assessment, tab_subjects, tab_recommendations = st.tabs(["📝 Assessment", "📚 Subjects", "💡 Recommendations"])

    with tab_assessment:
      st.markdown("#### Subject Performance Summary")
      summary_df = student_df[["Subject", "Attendance", "Midterm_Marks", "Final_Marks", "Risk_Level"]].copy()
      summary_df.columns = ["Subject Name", "Attendance Rate (%)", "Midterm Score", "Final Score", "ML Status"]
      st.dataframe(summary_df, use_container_width=True)

    with tab_subjects:
      st.markdown("#### Score Breakdown per Subject")
      fig_scores = px.bar(
          student_df,
          x="Subject",
          y=["Quiz_Score", "Assignment_Score", "Midterm_Marks", "Final_Marks"],
          barmode="group",
          color_discrete_sequence=["#8b5cf6", "#a78bfa", "#c4b5fd", "#6d28d9"]
      )
      fig_scores.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
      st.plotly_chart(fig_scores, use_container_width=True)

    with tab_recommendations:
      st.markdown("#### Actionable Pedagogical Steps")
      for idx, row in student_df.iterrows():
        status = row["Risk_Level"]
        icon = "⚠️" if status in ["High", "Moderate"] else "✅"
        with st.expander(f"{icon} {row['Subject']} (Status: {status})"):
          st.write(f"**Attendance:** {row['Attendance']}% | **Final Marks:** {row['Final_Marks']}")
          if row["Final_Marks"] < 50:
            st.write("• **Intervention:** Schedule mandatory peer-tutoring sessions.")
          if row["Attendance"] < 75:
            st.write("• **Intervention:** Issue an attendance climate check notice.")

  with right_col:
    st.markdown("### INSTITUTIONAL OVERVIEW")
    
    # Donut Charts (Records by Grade & Semester)
    dc1, dc2 = st.columns(2)
    with dc1:
      st.markdown("##### Records by Risk Grade")
      risk_counts = df["Risk_Level"].value_counts().reset_index()
      risk_counts.columns = ["Risk_Level", "Count"]
      fig_donut1 = px.pie(risk_counts, values="Count", names="Risk_Level", hole=0.6, color_discrete_sequence=["#ef4444", "#f59e0b", "#10b981"])
      fig_donut1.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10))
      st.plotly_chart(fig_donut1, use_container_width=True)

    with dc2:
      st.markdown("##### Records by Department")
      dept_counts = df["Department"].value_counts().reset_index()
      dept_counts.columns = ["Department", "Count"]
      fig_donut2 = px.pie(dept_counts, values="Count", names="Department", hole=0.6, color_discrete_sequence=["#6e44ff", "#8b5cf6", "#a78bfa", "#c4b5fd"])
      fig_donut2.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10))
      st.plotly_chart(fig_donut2, use_container_width=True)

    st.markdown("---")
    st.markdown("##### Department Final Marks Distribution")
    fig_dept_bar = px.bar(
        df.groupby("Department")["Final_Marks"].mean().reset_index(),
        x="Department",
        y="Final_Marks",
        color="Department",
        color_discrete_sequence=["#6e44ff", "#8b5cf6", "#a78bfa", "#c4b5fd"]
    )
    fig_dept_bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_dept_bar, use_container_width=True)


# ==========================================
# SECTION 2: STUDENT DESK (Full Detail View)
# ==========================================
elif selected_section == "👤 Student Desk":
  st.title("👤 Student Desk & Profile Lookup")
  st.markdown("---")
  
  student_df = df[df["Student_ID"] == selected_student_id]
  student_name = student_df["Student_Name"].iloc[0]

  st.subheader(f"Detailed Analytics: {student_name} ({selected_student_id})")
  
  m1, m2, m3, m4 = st.columns(4)
  m1.metric("Mean Attendance", f"{student_df['Attendance'].mean():.1f}%")
  m2.metric("Mean Final Marks", f"{student_df['Final_Marks'].mean():.1f}")
  m3.metric("Est. Missed Tasks", int(student_df['missed_assessments'].sum()))
  m4.metric("Academic Risk Status", student_df['Risk_Level'].iloc[0])

  st.markdown("---")
  st.dataframe(student_df, use_container_width=True)


# ==========================================
# SECTION 3: INSTITUTIONAL ANALYTICS
# ==========================================
elif selected_section == "📈 Institutional Analytics":
  st.title("📈 Model Verification & Analytics")
  st.markdown("---")

  c1, c2 = st.columns(2)
  with c1:
    st.markdown("#### 🌲 Random Forest Model Metrics")
    rf_metrics_df = pd.DataFrame({
        "Class": ["High", "Moderate", "Low"],
        "Precision": [rf_report['High']['precision'], rf_report['Moderate']['precision'], rf_report['Low']['precision']],
        "Recall": [rf_report['High']['recall'], rf_report['Moderate']['recall'], rf_report['Low']['recall']],
        "F1-Score": [rf_report['High']['f1-score'], rf_report['Moderate']['f1-score'], rf_report['Low']['f1-score']]
    })
    st.dataframe(rf_metrics_df.style.format(precision=2), use_container_width=True)

  with c2:
    st.markdown("#### 📉 Logistic Regression Baseline Metrics")
    lr_metrics_df = pd.DataFrame({
        "Class": ["High", "Moderate", "Low"],
        "Precision": [lr_report['High']['precision'], lr_report['Moderate']['precision'], lr_report['Low']['precision']],
        "Recall": [lr_report['High']['recall'], lr_report['Moderate']['recall'], lr_report['Low']['recall']],
        "F1-Score": [lr_report['High']['f1-score'], lr_report['Moderate']['f1-score'], lr_report['Low']['f1-score']]
    })
    st.dataframe(lr_metrics_df.style.format(precision=2), use_container_width=True)
