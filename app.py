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

# --- Data Loading & Model Training Pipeline ---
@st.cache_resource
def load_data_and_train_model():
  csv_path = "student_performance.csv"
  
  if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
  else:
    st.error(
        "⚠️ 'student_performance.csv' not found in the root directory. Please upload "
        "your 200-student dataset matching the required schema."
    )
    st.stop()

  # Define Features and Target
  feature_cols = [
      "attendance",
      "assignment_avg",
      "quiz_avg",
      "Midsem",
      "Comprehensive marks",
      "previous_avg",
      "recent_avg",
      "missed_assessments",
      "subject",
      "performance_trend",
  ]
  
  X = df[feature_cols]
  y = df["risk"]

  # Preprocessing for Numeric & Categorical Columns
  numeric_features = [
      "attendance",
      "assignment_avg",
      "quiz_avg",
      "Midsem",
      "Comprehensive marks",
      "previous_avg",
      "recent_avg",
      "missed_assessments",
  ]
  categorical_features = ["subject", "performance_trend"]

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
  
  # Train model on the dataset
  pipeline.fit(X, y)
  return df, pipeline

# Load data and trained model
df, model = load_data_and_train_model()

# --- Sidebar Navigation & Student Selection ---
st.sidebar.title("🎓 Student Advisor Dashboard")
st.sidebar.markdown(
    "AI-driven academic risk assessment, telemetry tracking, and personalized interventions."
)

student_list = sorted(df["student_id"].unique())
selected_student = st.sidebar.selectbox("Select Student ID", student_list)

# Filter Data for Selected Student
student_df = df[df["student_id"] == selected_student]

st.title(f"Performance Dossier: {selected_student}")
st.markdown("---")

# --- Overview Metrics ---
col1, col2, col3, col4 = st.columns(4)
overall_attendance = student_df["attendance"].mean()
overall_avg_score = student_df["recent_avg"].mean()
total_missed = student_df["missed_assessments"].sum()

# Predict risk probabilities using the trained model
risk_probs = model.predict_proba(student_df)[:, 1]
max_risk_prob = float(risk_probs.max())
overall_risk_status = "At Risk" if max_risk_prob >= 0.5 else "Stable"

col1.metric("Overall Attendance", f"{overall_attendance:.1f}%")
col2.metric("Recent Score Average", f"{overall_avg_score:.1f}")
col3.metric("Total Missed Assessments", int(total_missed))
col4.metric(
    "Max Risk Probability",
    f"{max_risk_prob:.2f}",
    delta=overall_risk_status,
    delta_color="inverse",
)

st.markdown("---")

# --- Subject-Wise Performance & Ranking ---
st.subheader("📚 Subject-Wise Breakdown & Priority Ranking")
display_df = student_df[[
    "subject",
    "attendance",
    "assignment_avg",
    "quiz_avg",
    "Midsem",
    "Comprehensive marks",
    "recent_avg",
    "performance_trend",
    "missed_assessments",
]].copy()
display_df["Risk Probability"] = [f"{p:.2f}" for p in risk_probs]
display_df = display_df.sort_values(by="recent_avg", ascending=True)

st.dataframe(display_df, use_container_width=True)

# --- Visualizations ---
col_left, col_right = st.columns(2)

with col_left:
  st.subheader("📊 Assessment Averages by Subject")
  fig_bar = px.bar(
      student_df,
      x="subject",
      y=["assignment_avg", "quiz_avg", "recent_avg"],
      barmode="group",
      title="Score Metrics across Subjects",
  )
  st.plotly_chart(fig_bar, use_container_width=True)

with col_right:
  st.subheader("📈 Attendance vs. Recent Score Trend")
  fig_scatter = px.scatter(
      student_df,
      x="attendance",
      y="recent_avg",
      color="performance_trend",
      size="missed_assessments",
      hover_data=["subject"],
      title="Attendance vs Recent Average (Size = Missed Assessments)",
  )
  st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown("---")

# --- Rule-Based Recommendation Engine ---
st.subheader("💡 Actionable Recommendations & Guidance")

def generate_recommendation(row, prob):
  recommendations = []
  subject = row["subject"]

  if prob >= 0.5:
    recommendations.append(
        f"**{subject} currently requires immediate attention.** (Model Risk "
        f"Probability: {prob:.2f})"
    )
  else:
    recommendations.append(
        f"**{subject} is stable.** (Model Risk Probability: {prob:.2f})"
    )

  if row["recent_avg"] < 60:
    recommendations.append(
        f"- Your recent scores in {subject} ({row['recent_avg']}) are "
        "declining. Review recent assessment topics and schedule instructor office hours."
    )
  if row["attendance"] < 70:
    recommendations.append(
        f"- Attendance stands at {row['attendance']}%, which is below the "
        "recommended institutional threshold (75%). Prioritize upcoming class participation."
    )
  if row["missed_assessments"] > 0:
    recommendations.append(
        f"- You have missed {int(row['missed_assessments'])} assessment(s). "
        "Clear pending submissions immediately."
    )
  if len(recommendations) == 1:
    recommendations.append(
        f"- Maintain your current study strategy for {subject}."
    )

  return " ".join(recommendations)

for idx, row in student_df.iterrows():
  prob = risk_probs[student_df.index.get_loc(idx)]
  with st.expander(f"Intervention Report: {row['subject']} (Risk Score: {prob:.2f})"):
    rec_text = generate_recommendation(row, prob)
    st.markdown(rec_text)

# --- Feature Importance Explainability ---
st.markdown("---")
st.subheader("🔍 Model Explainability (Feature Importances)")
rf_model = model.named_steps["classifier"]
preprocessor = model.named_steps["preprocessor"]

cat_encoder = preprocessor.named_transformers_["cat"]
encoded_cat_features = list(
    cat_encoder.get_feature_names_out(["subject", "performance_trend"])
)
all_features = [
    "attendance",
    "assignment_avg",
    "quiz_avg",
    "Midsem",
    "Comprehensive marks",
    "previous_avg",
    "recent_avg",
    "missed_assessments",
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
    title="Top 10 Feature Importances Driving Risk Predictions",
)
fig_imp.update_layout(yaxis={"categoryorder": "total ascending"})
st.plotly_chart(fig_imp, use_container_width=True)
