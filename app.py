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

# --- Trace Academy Purple Aesthetic CSS ---
st.markdown("""
    <style>
        .main { background-color: #f3f0f7; }
        
        /* Card Styling */
        .desk-card {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 14px;
            box-shadow: 0 4px 12px rgba(110, 68, 255, 0.08);
            border: 1px solid #e9e1f5;
            margin-bottom: 20px;
        }
        
        /* Profile Detail Badges */
        .profile-badge {
            background-color: #f7f4fb;
            border-radius: 8px;
            padding: 8px 12px;
            margin-bottom: 6px;
            border: 1px solid #ebe2f7;
        }
        .profile-badge label {
            font-size: 10px;
            color: #7e6e9b;
            text-transform: uppercase;
            font-weight: 700;
            display: block;
        }
        .profile-badge span {
            font-size: 14px;
            color: #2d1854;
            font-weight: 700;
        }
        
        /* Floating Chat Widget ("Peter") */
        .floating-chat-container {
            position: fixed !important;
            bottom: 85px !important;
            right: 25px !important;
            width: 350px !important;
            max-width: calc(100vw - 40px) !important;
            height: 460px !important;
            background-color: white !important;
            border-radius: 14px !important;
            box-shadow: 0 8px 24px rgba(110,68,255,0.25) !important;
            z-index: 999998 !important;
            padding: 12px !important;
            display: flex !important;
            flex-direction: column !important;
            border: 1px solid #e5e7eb !important;
        }
        
        .floating-btn-wrapper {
            position: fixed !important;
            bottom: 20px !important;
            right: 25px !important;
            z-index: 999999 !important;
        }
        .floating-btn-wrapper button {
            border-radius: 50% !important;
            width: 55px !important;
            height: 55px !important;
            background-color: #6e44ff !important;
            color: white !important;
            font-size: 22px !important;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3) !important;
            border: none !important;
        }
    </style>
""", unsafe_allow_html=True)

# --- ML Pipeline Workflow (Slide 6 Implementation) ---
@st.cache_resource
def run_ml_pipeline():
  # 1. Dataset Collection[span_3](start_span)[span_3](end_span)
  excel_path = "data/Subject_Analytics_Dataset_200_Students.xlsx"
  
  if not os.path.exists(excel_path):
    st.error(f"⚠️ Excel dataset not found at '{excel_path}'. Please ensure the file is in the 'data' folder.")
    st.stop()

  try:
    df = pd.read_excel(excel_path)
  except Exception as e:
    st.error(f"⚠️ Error reading dataset: {e}")
    st.stop()

  # 2. Data Cleaning & Validation[span_4](start_span)[span_4](end_span)
  df.columns = df.columns.str.strip()
  for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].astype(str).str.strip()

  # 3 & 4. Feature Engineering[span_5](start_span)[span_5](end_span)
  if "performance_trend" not in df.columns:
    df["performance_trend"] = np.where(df["Final_Marks"] >= df["Midterm_Marks"], 1, -1)
  else:
    df["performance_trend"] = df["performance_trend"].map({"Improving": 1, "Stable": 1, "Declining": -1}).fillna(0)

  if "missed_assessments" not in df.columns:
    df["missed_assessments"] = np.where(df["Attendance"] < 75, 2, 0)

  feature_cols = ["Attendance", "Quiz_Score", "Assignment_Score", "Midterm_Marks", "Final_Marks", "performance_trend", "missed_assessments"]
  
  # Binary Target: 0 = No Risk, 1 = Academic Attention Required[span_6](start_span)[span_6](end_span)
  composite_score = (df["Attendance"] * 0.25) + (df["Final_Marks"] * 0.35) + (df["Midterm_Marks"] * 0.25) + (df["Quiz_Score"] * 0.15)
  df["risk_target"] = np.where(composite_score < 58, 1, 0)

  X = df[feature_cols]
  y = df["risk_target"]

  # 5. Train-Test Split (80/20)[span_7](start_span)[span_7](end_span)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  # 6. Baseline Model: Logistic Regression[span_8](start_span)[span_8](end_span)
  lr_baseline = LogisticRegression(random_state=42, max_iter=300)
  lr_baseline.fit(X_train, y_train)
  lr_pred = lr_baseline.predict(X_test)
  lr_report = classification_report(y_test, lr_pred, output_dict=True)

  # 7. Main Model: Random Forest Classifier[span_9](start_span)[span_9](end_span)
  rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
  rf_model.fit(X_train, y_train)
  rf_pred = rf_model.predict(X_test)
  
  # 8. Evaluation Metrics[span_10](start_span)[span_10](end_span)
  rf_report = classification_report(y_test, rf_pred, output_dict=True)
  conf_matrix = confusion_matrix(y_test, rf_pred)

  # Model Risk Probability Score Assignment
  df["ML_Attention_Predicted"] = rf_model.predict(X)
  df["ML_Risk_Probability"] = rf_model.predict_proba(X)[:, 1]

  return df, rf_model, rf_report, lr_report, conf_matrix, feature_cols

df, rf_model, rf_report, lr_report, conf_matrix, feature_cols = run_ml_pipeline()

# --- Top Header & Brand Bar ---
header_col1, header_col2 = st.columns([2, 3])
with header_col1:
  st.title("🎓 TRACE ACADEMY")
  st.caption("STUDENTS PERFORMANCE OVERVIEW (ML-DRIVEN)")

with header_col2:
  student_options = df[["Student_ID", "Student_Name"]].drop_duplicates().sort_values(by="Student_ID")
  student_options["Label"] = student_options["Student_ID"].astype(str) + " - " + student_options["Student_Name"].astype(str)
  
  selected_student_label = st.selectbox("Select Student", options=student_options["Label"].tolist(), index=0)
  selected_student_id = selected_student_label.split(" - ")[0]

# --- Sidebar Navigation ---
st.sidebar.title("📌 Navigation")
selected_section = st.sidebar.radio(
    "Go to", 
    ["📊 Students Performance Desk", "📈 Model Verification & Analytics"]
)

# ==========================================
# SECTION 1: STUDENTS PERFORMANCE DESK
# ==========================================
if selected_section == "📊 Students Performance Desk":
  student_df = df[df["Student_ID"] == selected_student_id]
  student_name = student_df["Student_Name"].iloc[0]
  student_department = student_df["Department"].iloc[0]
  student_semester = student_df["Semester"].iloc[0]
  
  avg_score = student_df["Final_Marks"].mean()
  pass_rate = (student_df["Final_Marks"] >= 50).mean() * 100
  overall_grade = "A" if avg_score >= 75 else "B" if avg_score >= 60 else "C" if avg_score >= 50 else "F"
  
  ml_risk_prob = student_df["ML_Risk_Probability"].mean()
  ml_requires_attention = ml_risk_prob >= 0.5

  left_col, right_col = st.columns([3, 2])

  with left_col:
    st.markdown("### STUDENTS PERFORMANCE DESK")
    
    # Student Metadata Badges
    p_col1, p_col2, p_col3 = st.columns([1, 2, 2])
    with p_col1:
      st.image("https://api.dicebear.com/7.x/avataaars/svg?seed=" + student_name, width=115)
    
    with p_col2:
      st.markdown(f"<div class='profile-badge'><label>Full Name</label><span>{student_name}</span></div>", unsafe_allow_html=True)
      st.markdown(f"<div class='profile-badge'><label>Level / Semester</label><span>Semester {student_semester}</span></div>", unsafe_allow_html=True)
      st.markdown(f"<div class='profile-badge'><label>Records</label><span>{len(student_df)} Subjects</span></div>", unsafe_allow_html=True)
      st.markdown(f"<div class='profile-badge'><label>Pass Rate</label><span>{pass_rate:.1f}%</span></div>", unsafe_allow_html=True)

    with p_col3:
      st.markdown(f"<div class='profile-badge'><label>Department</label><span>{student_department}</span></div>", unsafe_allow_html=True)
      st.markdown(f"<div class='profile-badge'><label>Student ID</label><span>{selected_student_id}</span></div>", unsafe_allow_html=True)
      st.markdown(f"<div class='profile-badge'><label>Avg Score</label><span>{avg_score:.2f}</span></div>", unsafe_allow_html=True)
      st.markdown(f"<div class='profile-badge'><label>ML Risk Status</label><span style='color: {'#ef4444' if ml_requires_attention else '#10b981'};'>{'Requires Attention' if ml_requires_attention else 'No Risk'} ({ml_risk_prob:.2f})</span></div>", unsafe_allow_html=True)

    st.markdown("---")
    
    # Trace Academy Tabbed Desk
    tab_assessment, tab_subjects, tab_recommendations = st.tabs(["📝 Assessment", "📚 Subjects", "💡 Recommendations"])

    with tab_assessment:
      st.markdown("#### Subject Performance Summary")
      summary_df = student_df[["Subject", "Attendance", "Midterm_Marks", "Final_Marks", "ML_Risk_Probability", "ML_Attention_Predicted"]].copy()
      summary_df["ML_Attention_Predicted"] = np.where(summary_df["ML_Attention_Predicted"] == 1, "Flagged", "Stable")
      summary_df.columns = ["Subject Name", "Attendance (%)", "Midterm", "Final Score", "ML Risk Prob", "ML Flag"]
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
      st.markdown("#### ML Recommendation Engine (Slide 7 & 8)")[span_11](start_span)[span_11](end_span)[span_12](start_span)[span_12](end_span)
      for idx, row in student_df.iterrows():
        prob = row["ML_Risk_Probability"]
        subj = row["Subject"]
        
        if prob >= 0.5:
          st.error(f"⚠️ **{subj}** — ML Risk Probability: `{prob:.2f}` (Requires Academic Attention)[span_13](start_span)[span_13](end_span)\n\n"
                   f"*System Recommendation:* Scores and attendance in {subj} are trailing behind. Review recent assessment topics and prioritize upcoming assessments[span_14](start_span)[span_14](end_span).")
        else:
          st.success(f"✅ **{subj}** — ML Risk Probability: `{prob:.2f}` (No Immediate Risk)\n\n"
                     f"*System Recommendation:* Maintain current study strategy. Performance is stable.")

  with right_col:
    st.markdown("### INSTITUTIONAL OVERVIEW")
    
    dc1, dc2 = st.columns(2)
    with dc1:
      st.markdown("##### Records by ML Risk")
      risk_counts = df["ML_Attention_Predicted"].map({1: "Requires Attention", 0: "No Risk"}).value_counts().reset_index()
      risk_counts.columns = ["Status", "Count"]
      fig_donut1 = px.pie(risk_counts, values="Count", names="Status", hole=0.6, color="Status", color_discrete_map={"Requires Attention": "#ef4444", "No Risk": "#10b981"})
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
# SECTION 2: MODEL VERIFICATION & ANALYTICS
# ==========================================
elif selected_section == "📈 Model Verification & Analytics":
  st.title("📈 Machine Learning Workflow Verification")
  st.markdown("Direct evaluation of models specified in Slide 2 & Slide 6[span_15](start_span)[span_15](end_span)[span_16](start_span)[span_16](end_span).")
  st.markdown("---")

  c1, c2 = st.columns(2)
  with c1:
    st.markdown("#### 🌲 Primary Model: Random Forest Classifier[span_17](start_span)[span_17](end_span)")
    rf_metrics_df = pd.DataFrame({
        "Class": ["No Risk (0)", "Attention (1)"],
        "Precision": [rf_report['0']['precision'], rf_report['1']['precision']],
        "Recall": [rf_report['0']['recall'], rf_report['1']['recall']],
        "F1-Score": [rf_report['0']['f1-score'], rf_report['1']['f1-score']]
    })
    st.dataframe(rf_metrics_df.style.format(precision=2), use_container_width=True)

  with c2:
    st.markdown("#### 📉 Baseline Model: Logistic Regression[span_18](start_span)[span_18](end_span)")
    lr_metrics_df = pd.DataFrame({
        "Class": ["No Risk (0)", "Attention (1)"],
        "Precision": [lr_report['0']['precision'], lr_report['1']['precision']],
        "Recall": [lr_report['0']['recall'], lr_report['1']['recall']],
        "F1-Score": [lr_report['0']['f1-score'], lr_report['1']['f1-score']]
    })
    st.dataframe(lr_metrics_df.style.format(precision=2), use_container_width=True)

  st.markdown("---")
  st.markdown("#### 🔍 Explainability & Feature Importance (Slide 11)[span_19](start_span)[span_19](end_span)")
  feature_importances = pd.DataFrame({
      "Feature": feature_cols,
      "Importance": rf_model.feature_importances_
  }).sort_values(by="Importance", ascending=True)

  fig_importance = px.bar(
      feature_importances,
      x="Importance",
      y="Feature",
      orientation="h",
      title="Random Forest Feature Importance Weights",
      color="Importance",
      color_continuous_scale="Purples"
  )
  st.plotly_chart(fig_importance, use_container_width=True)


# ==========================================
# FLOATING CHATBOT WIDGET ("PETER")
# ==========================================
if "chat_open" not in st.session_state:
  st.session_state.chat_open = False

if "messages" not in st.session_state:
  st.session_state.messages = [
      {"role": "assistant", "content": "👋 Hi! I'm **Peter**, your ML Performance Assistant. Ask me about student risk probabilities or type a student ID!"}
  ]

with st.container():
  st.markdown('<div class="floating-btn-wrapper">', unsafe_allow_html=True)
  if st.button("💬", key="chat_toggle_btn", type="primary"):
    st.session_state.chat_open = not st.session_state.chat_open
    st.rerun()
  st.markdown('</div>', unsafe_allow_html=True)

if st.session_state.chat_open:
  with st.container():
    st.markdown('<div class="floating-chat-container">', unsafe_allow_html=True)
    
    ch1, ch2 = st.columns([4, 1])
    with ch1:
      st.markdown("**💬 Peter (AI Advisor)**")
    with ch2:
      if st.button("✖", key="close_chat"):
        st.session_state.chat_open = False
        st.rerun()
        
    st.markdown("---")
    
    chat_container = st.container(height=260)
    with chat_container:
      for message in st.session_state.messages:
        with st.chat_message(message["role"]):
          st.markdown(message["content"])

    with st.form(key="peter_chat_form", clear_on_submit=True):
      cols_input = st.columns([4, 1])
      with cols_input[0]:
        user_text = st.text_input("Prompt Peter...", placeholder="Ask about student or ML risk...", label_visibility="collapsed")
      with cols_input[1]:
        submitted = st.form_submit_button("Send")

      if submitted and user_text:
        st.session_state.messages.append({"role": "user", "content": user_text})
        query_lower = user_text.lower()
        
        found_student = None
        for _, row in df.drop_duplicates(subset=["Student_ID"]).iterrows():
          if str(row["Student_ID"]).lower() in query_lower or row["Student_Name"].lower() in query_lower:
            found_student = row["Student_ID"]
            break
        
        if found_student:
          s_data = df[df["Student_ID"] == found_student]
          s_name = s_data["Student_Name"].iloc[0]
          s_prob = s_data["ML_Risk_Probability"].mean()
          s_flag = "Requires Attention" if s_prob >= 0.5 else "No Immediate Risk"

          response = f"👤 **Student Lookup: {s_name} ({found_student})**\n- **ML Risk Probability:** `{s_prob:.2f}`\n- **Predicted Status:** {s_flag}\n\n"
          if s_prob >= 0.5:
            response += "⚠️ *ML Insight:* High probability of academic strain. Recommend intervention."
          else:
            response += "✅ *ML Insight:* Low risk probability. Student is performing well."
        else:
          response = "I am Peter! Ask about a student name, ID (e.g. S029), or type 'risk' to query ML model outputs."

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)
      
