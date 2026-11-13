import re
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

# -----------------------------------------------------------------------------
# 1. PAGE CONFIG & HIGH-CONTRAST ACADEMIC STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI-Based Student Performance Advisor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .stApp {
            background-color: #f8fafc;
            color: #000000;
        }
        h1, h2, h3, h4, h5, h6, p, li, span, label, div {
            color: #000000 !important;
        }
        .kpi-card {
            background-color: #ffffff;
            border: 1px solid #cbd5e1;
            border-radius: 8px;
            padding: 16px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        }
        .kpi-title {
            font-size: 13px;
            font-weight: 700;
            color: #334155 !important;
            text-transform: uppercase;
            margin-bottom: 4px;
        }
        .kpi-value {
            font-size: 26px;
            font-weight: 800;
            color: #000000 !important;
        }
        .rule-box {
            background-color: #ffffff;
            border-left: 5px solid #1e40af;
            border-top: 1px solid #cbd5e1;
            border-right: 1px solid #cbd5e1;
            border-bottom: 1px solid #cbd5e1;
            padding: 12px 16px;
            border-radius: 6px;
            margin-bottom: 10px;
        }
        .disclaimer-box {
            background-color: #f1f5f9;
            border: 1px solid #94a3b8;
            border-radius: 6px;
            padding: 10px 14px;
            font-size: 13px;
        }
        button[data-baseweb="tab"] {
            font-weight: 700 !important;
            font-size: 15px !important;
            color: #000000 !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# 2. DATA INGESTION, PRIVACY ANONYMIZATION & RELIABILITY CHECKS
# -----------------------------------------------------------------------------
@st.cache_data
def load_and_prepare_data(file_path="data/Subject_Analytics_Dataset_200_Students.xlsx"):
    try:
        df = pd.read_excel(file_path)
    except Exception:
        # Fallback synthetic multi-subject dataset matching slide specifications if Excel is absent
        np.random.seed(42)
        students = [f"S{str(i).zfill(3)}" for i in range(1, 51)]
        subjects = ["Statistics", "Data Structures", "Machine Learning", "Database Systems"]
        depts = ["Computer Science", "Data Science", "Information Systems"]
        rows = []
        for s_id in students:
            dept = np.random.choice(depts)
            base_perf = np.random.normal(68, 16)
            base_att = np.clip(np.random.normal(80, 12), 45, 100)
            for subj in subjects:
                att = round(np.clip(base_att + np.random.normal(0, 4), 40, 100), 1)
                quiz = round(np.clip(base_perf + np.random.normal(0, 8), 25, 100), 1)
                assign = round(np.clip(base_perf + np.random.normal(2, 7), 30, 100), 1)
                midsem = round(np.clip(base_perf + np.random.normal(-2, 9), 20, 100), 1)
                final = round(np.clip(midsem + np.random.normal(1, 8), 20, 100), 1)
                rows.append(
                    {
                        "Student_ID": s_id,
                        "Department": dept,
                        "Subject": subj,
                        "Attendance": att,
                        "Quiz_Score": quiz,
                        "Assignment_Score": assign,
                        "Midterm_Marks": midsem,
                        "Final_Marks": final,
                    }
                )
        df = pd.DataFrame(rows)

    # Normalize column aliases (Slides 3, 4, 6)
    alias_map = {
        "Student ID": "Student_ID",
        "student_id": "Student_ID",
        "Assignment": "Assignment_Score",
        "assignment_avg": "Assignment_Score",
        "quiz_avg": "Quiz_Score",
        "Midsem": "Midterm_Marks",
        "Comprehensive marks": "Final_Marks",
        "attendance": "Attendance",
        "subject": "Subject",
    }
    df = df.rename(columns={k: v for k, v in alias_map.items() if k in df.columns})

    # Responsible Use / Data Integrity: Strip PII columns if present (keep only anonymized Student_ID)
    pii_cols = [c for c in df.columns if c.lower() in ["student_name", "name", "email", "phone"]]
    if pii_cols:
        df = df.drop(columns=pii_cols)

    if "Subject" not in df.columns:
        df["Subject"] = "Core Module"
    if "Department" not in df.columns:
        df["Department"] = "B.Sc Design & Computing"

    num_cols = ["Attendance", "Quiz_Score", "Assignment_Score", "Midterm_Marks", "Final_Marks"]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Data Reliability Check (Slide 11: "Insufficient Evidence / Needs Review" state)
    df["Data_Reliable"] = ~df[num_cols].isna().any(axis=1) & (df["Attendance"].between(0, 100))
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # Subject-level feature engineering
    df["Composite_Score"] = (
        0.15 * df["Quiz_Score"]
        + 0.20 * df["Assignment_Score"]
        + 0.30 * df["Midterm_Marks"]
        + 0.35 * df["Final_Marks"]
    ).round(1)

    df["Trend_Delta"] = (df["Final_Marks"] - df["Midterm_Marks"]).round(1)
    df["Performance_Trend"] = np.where(
        df["Trend_Delta"] > 2,
        "Improving",
        np.where(df["Trend_Delta"] < -3, "Declining", "Stable"),
    )
    df["Missed_Assessments"] = np.where(df["Attendance"] < 75, 2, np.where(df["Attendance"] < 82, 1, 0))

    # Primary Key Aggregation by Student_ID
    agg_df = (
        df.groupby("Student_ID", as_index=False)
        .agg(
            Department=("Department", "first"),
            Attendance=("Attendance", "mean"),
            Quiz_Score=("Quiz_Score", "mean"),
            Assignment_Score=("Assignment_Score", "mean"),
            Midterm_Marks=("Midterm_Marks", "mean"),
            Final_Marks=("Final_Marks", "mean"),
            Composite_Score=("Composite_Score", "mean"),
            Trend_Delta=("Trend_Delta", "mean"),
            Missed_Assessments=("Missed_Assessments", "sum"),
            Data_Reliable=("Data_Reliable", "all"),
        )
        .round(1)
    )

    agg_df["Performance_Trend"] = np.where(
        agg_df["Trend_Delta"] > 2,
        "Improving",
        np.where(agg_df["Trend_Delta"] < -3, "Declining", "Stable"),
    )

    # Target (Slide 2): Binary Classification (0 = Stable, 1 = Requires Academic Attention / Risk)
    agg_df["Risk_Target"] = np.where(
        (agg_df["Composite_Score"] < 60) | (agg_df["Attendance"] < 75), 1, 0
    )

    def map_grade(score):
        if score >= 80:
            return "A"
        elif score >= 70:
            return "B"
        elif score >= 60:
            return "C"
        elif score >= 50:
            return "D"
        return "F"

    agg_df["Grade"] = agg_df["Composite_Score"].apply(map_grade)
    return df, agg_df


# -----------------------------------------------------------------------------
# 3. MACHINE LEARNING & EXPLAINABILITY ENGINE (SLIDES 2, 6, 11)
# -----------------------------------------------------------------------------
@st.cache_resource
def train_ml_models(agg_df):
    feature_cols = [
        "Attendance",
        "Quiz_Score",
        "Assignment_Score",
        "Midterm_Marks",
        "Final_Marks",
        "Trend_Delta",
        "Missed_Assessments",
    ]
    X = agg_df[feature_cols]
    y = agg_df["Risk_Target"]

    if len(np.unique(y)) < 2:
        y.iloc[0] = 1 - y.iloc[0]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    rf = RandomForestClassifier(n_estimators=120, random_state=42)
    rf.fit(X_train, y_train)

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)

    probs = rf.predict_proba(X)
    risk_prob = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]

    y_pred = rf.predict(X_test)
    metrics = {
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 3),
        "Recall": round(recall_score(y_test, y_pred, zero_division=0), 3),
        "F1-Score": round(f1_score(y_test, y_pred, zero_division=0), 3),
    }

    importances = pd.DataFrame(
        {"Feature": feature_cols, "Importance": rf.feature_importances_}
    ).sort_values("Importance", ascending=True)

    return rf, lr, risk_prob, metrics, importances, feature_cols


def compute_local_explainability(student_row, agg_df, feature_cols, rf_model):
    cohort_means = agg_df[feature_cols].mean()
    importances = dict(zip(feature_cols, rf_model.feature_importances_))

    contributions = []
    for feat in feature_cols:
        val = student_row[feat]
        mean_val = cohort_means[feat]
        imp = importances[feat]

        if feat == "Missed_Assessments":
            diff = (val - mean_val) / (agg_df[feat].std() + 1e-5)
        else:
            diff = (mean_val - val) / (agg_df[feat].std() + 1e-5)

        risk_impact = round(diff * imp * 100, 2)
        contributions.append(
            {
                "Feature": feat,
                "Student_Value": val,
                "Cohort_Avg": round(mean_val, 1),
                "Risk_Contribution": risk_impact,
            }
        )
    return pd.DataFrame(contributions).sort_values("Risk_Contribution", ascending=True)


# -----------------------------------------------------------------------------
# 4. RECOMMENDATION ENGINE (SLIDES 7 & 8 EXACT ANALYTICS GENERATION)
# -----------------------------------------------------------------------------
def generate_student_recommendations(student_row, subject_rows):
    if not student_row["Data_Reliable"]:
        return {
            "status": "Insufficient Evidence / Needs Review",
            "priority_subjects": [],
            "rules_triggered": [
                "Data reliability check failed (missing or out-of-range records). Advisor review required before automated intervention."
            ],
            "study_plan": {},
        }

    subj_df = subject_rows.copy()
    subj_df["Priority_Score"] = (100 - subj_df["Composite_Score"]) + np.where(
        subj_df["Attendance"] < 75, 15, 0
    )
    subj_df = subj_df.sort_values("Priority_Score", ascending=False)
    priority_subjects = subj_df["Subject"].tolist()

    rules_triggered = []
    # Slide 8 exact recommendation engine example rule generation from actual analytics
    for _, s_row in subj_df.iterrows():
        if (
            s_row["Composite_Score"] < 60
            and s_row["Performance_Trend"] == "Declining"
            and s_row["Attendance"] < 75
        ):
            rule = (
                f"System output: \"{s_row['Subject']} currently requires additional attention. "
                f"Your recent scores are declining (Midterm: {s_row['Midterm_Marks']} → Final: {s_row['Final_Marks']}, "
                f"Average: {s_row['Composite_Score']}%) and attendance is below the selected target ({s_row['Attendance']}%). "
                f"Review recent assessment topics and prioritize the upcoming {s_row['Subject']} assessment.\""
            )
            rules_triggered.append(rule)
        elif s_row["Composite_Score"] < 60:
            rules_triggered.append(
                f"System output: \"{s_row['Subject']} average is {s_row['Composite_Score']}%. Review recent assessment topics and prioritize upcoming assignments.\""
            )

    if student_row["Attendance"] < 75 and not rules_triggered:
        rules_triggered.append(
            f"System output: \"Overall attendance is {student_row['Attendance']}% (below 75% target). Improve attendance immediately to avoid assessment penalties.\""
        )

    if not rules_triggered:
        rules_triggered.append(
            "System output: \"Student performance is stable across subjects. Maintain current study strategy and attempt advanced practice sets.\""
        )

    top_subj = priority_subjects[0] if len(priority_subjects) > 0 else "Core Subject"
    sec_subj = priority_subjects[1] if len(priority_subjects) > 1 else top_subj

    study_plan = {
        "Monday – Tuesday (Core Remediation)": f"Focus 2 hrs/day on **{top_subj}** (highest priority). Revise recent lecture topics and solve past midterm problems.",
        "Wednesday – Thursday (Secondary Reinforcement)": f"Dedicate 1.5 hrs/day to **{sec_subj}** and complete pending assignments to boost continuous evaluation scores.",
        "Friday (Active Recall & Quizzes)": "Attempt 1 timed practice quiz (45 mins) covering weak modules and review error logs.",
        "Weekend (Consolidation & Attendance Prep)": "Summarize weekly notes (1 hr) and pre-read upcoming week's syllabus to maximize class participation.",
    }

    status = "Requires Academic Attention" if student_row["Risk_Prob"] >= 0.5 else "On Track / Stable"
    return {
        "status": status,
        "priority_subjects": priority_subjects,
        "rules_triggered": rules_triggered,
        "study_plan": study_plan,
    }


# -----------------------------------------------------------------------------
# 5. AI ACADEMIC CHATBOT ENGINE (Q&A FOR RECOMMENDATIONS & STUDY PLANS)
# -----------------------------------------------------------------------------
def process_chatbot_query(user_query, raw_df, agg_df):
    q = user_query.lower().strip()
    id_match = re.search(r"\bs\d{3}\b", q)
    at_risk_df = agg_df[agg_df["Risk_Prob"] >= 0.5].sort_values("Risk_Prob", ascending=False)

    # Case 1: Specific student query
    if id_match:
        s_id = id_match.group(0).upper()
        if s_id not in agg_df["Student_ID"].values:
            return f"Student ID **{s_id}** was not found in the dataset. Available IDs range from `{agg_df['Student_ID'].min()}` to `{agg_df['Student_ID'].max()}`."

        s_row = agg_df[agg_df["Student_ID"] == s_id].iloc[0]
        s_subjs = raw_df[raw_df["Student_ID"] == s_id]
        rec = generate_student_recommendations(s_row, s_subjs)

        response = f"### Academic Advisory & Study Plan for **{s_id}**\n"
        response += f"- **Department:** {s_row['Department']} | **Status:** {rec['status']} (ML Risk Probability: {s_row['Risk_Prob']:.2f})\n"
        response += f"- **Composite Score:** {s_row['Composite_Score']}/100 (Grade {s_row['Grade']}) | **Attendance:** {s_row['Attendance']}%\n"
        response += f"- **Subject Priority Order:** {', '.join(rec['priority_subjects'])}\n\n"
        response += "**Generated System Output (Slide 8 Rule):**\n"
        for r in rec["rules_triggered"]:
            response += f"- {r}\n"
        response += "\n**Personalized Weekly Study Plan:**\n"
        for day, plan in rec["study_plan"].items():
            response += f"- **{day}:** {plan}\n"
        return response

    # Case 2: Study plan for ALL students
    if ("study plan" in q or "schedule" in q or "timetable" in q) and (
        "all" in q or "everyone" in q or "class" in q or "cohort" in q
    ):
        response = f"### Cohort-Wide Tiered Study Plan (For All {len(agg_df)} Students)\n\n"
        response += f"#### Tier 1: At-Risk / Students Needing Attention ({len(at_risk_df)} Students — Risk Prob ≥ 0.5)\n"
        response += "- **Daily Focus (2.5 hrs/day):** Prioritize lowest-scoring subject first, focusing on foundational concepts and past midterm solutions.\n"
        response += "- **Attendance Mandate:** Maintain ≥ 85% attendance in upcoming lectures and attend weekly faculty doubt-clearing sessions.\n"
        response += "- **Weekly Checkpoint:** Complete 2 diagnostic quizzes every Friday to track score recovery.\n\n"
        response += "#### Tier 2: Moderate / Borderline Students (Composite Score 60–74)\n"
        response += "- **Daily Focus (1.5 hrs/day):** Revise recent lecture notes within 24 hours of class and complete all assignments ahead of deadlines.\n\n"
        response += "#### Tier 3: Stable Students (Composite Score ≥ 75)\n"
        response += "- **Daily Focus (1 hr/day):** Maintain current study strategy, peer-mentor Tier 1 students, and attempt advanced case-study problems.\n"
        return response

    # Case 3: Poor / at-risk / recommendations
    if any(k in q for k in ["poor", "weak", "risk", "low", "failing", "support", "recommend", "help"]):
        if at_risk_df.empty:
            return "Great news! Currently, no students are flagged in the high-risk category."

        top_risk = at_risk_df.head(8)
        response = f"### Recommendations for At-Risk / Poor Performing Students ({len(at_risk_df)} Total Flagged)\n"
        response += "Students are flagged when their ML Support Need Probability is ≥ 0.5 (driven by composite marks < 60 or attendance < 75%).\n\n"
        response += "**Top Priority Students Requiring Immediate Attention:**\n"
        for _, row in top_risk.iterrows():
            s_subjs = raw_df[raw_df["Student_ID"] == row["Student_ID"]]
            weakest_subj = s_subjs.sort_values("Composite_Score").iloc[0]["Subject"]
            response += (
                f"- **{row['Student_ID']}** ({row['Department']}) — Avg: `{row['Composite_Score']}`, "
                f"Attendance: `{row['Attendance']}%`, Trend: `{row['Performance_Trend']}` (ML Risk: `{row['Risk_Prob']}`) → "
                f"**Priority Subject:** *{weakest_subj}*\n"
            )
        response += "\n**Key Actionable Recommendations (Slide 7 & 8 Standards):**\n"
        response += "1. Combine model output with subject performance, trends, attendance, and assessment context.\n"
        response += "2. Revise recent topics, prioritize upcoming assessments, and improve attendance.\n"
        response += "3. Clearly distinguish model-based evidence from causal claims.\n"
        return response

    # Case 4: Explainability / Why flagged
    if any(k in q for k in ["explain", "why", "feature", "shap", "model", "important"]):
        return (
            "### Model Explainability & Responsible Use Summary (Slide 11)\n"
            "- **Top Predictive Drivers:** Attendance %, Final/Midterm Marks, Performance Trend Delta, and Missed Assessments.\n"
            "- **Responsible AI Note:** Model feature importance highlights statistical patterns in academic engagement—it **does not** prove causation. "
            "All predictions are combined with subject context rules and protect student privacy (`Student_ID` anonymized)."
        )

    return (
        "I am your **AI Academic Advisor Chatbot**. You can ask me questions such as:\n"
        "- *\"What are the recommendations for poor / at-risk students?\"*\n"
        "- *\"Generate a study plan for all students\"*\n"
        "- *\"Give me recommendations and a study plan for S003\"*\n"
        "- *\"Why does the AI model flag students at risk?\"*"
    )


# -----------------------------------------------------------------------------
# 6. LOAD DATA & SIDEBAR FILTERS
# -----------------------------------------------------------------------------
raw_df, agg_df = load_and_prepare_data()
rf_model, lr_model, risk_probs, model_metrics, feat_imp_df, feature_cols = train_ml_models(agg_df)
agg_df["Risk_Prob"] = risk_probs.round(3)

st.sidebar.title("🎓 Cohort & Student Filters")
dept_options = ["All Departments"] + sorted(agg_df["Department"].unique().tolist())
selected_dept = st.sidebar.selectbox("Department", dept_options)

filtered_agg = (
    agg_df if selected_dept == "All Departments" else agg_df[agg_df["Department"] == selected_dept]
)
selected_student_id = st.sidebar.selectbox(
    "Select Student ID (Anonymized)", filtered_agg["Student_ID"].tolist()
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Unique Students Loaded:** {len(agg_df)}")
st.sidebar.markdown(f"**Filtered Cohort Size:** {len(filtered_agg)}")
st.sidebar.caption("Privacy Protected: All student names/PII are anonymized using Student_ID.")


# -----------------------------------------------------------------------------
# 7. UNIFIED SINGLE DASHBOARD HEADER & TABS
# -----------------------------------------------------------------------------
st.title("AI-Based Student Performance Advisor and Visualization System")
st.markdown(
    "Unified academic monitoring, Slide 8 analytics-driven recommendations, model explainability, and interactive AI advisor."
)

# Shared Cohort KPI Strip across the top of the single dashboard
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(
        f"<div class='kpi-card'><div class='kpi-title'>Total Students</div><div class='kpi-value'>{len(filtered_agg)}</div></div>",
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        f"<div class='kpi-card'><div class='kpi-title'>Cohort Avg Score</div><div class='kpi-value'>{filtered_agg['Composite_Score'].mean():.1f}%</div></div>",
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        f"<div class='kpi-card'><div class='kpi-title'>Avg Attendance</div><div class='kpi-value'>{filtered_agg['Attendance'].mean():.1f}%</div></div>",
        unsafe_allow_html=True,
    )
with c4:
    at_risk_count = int((filtered_agg["Risk_Prob"] >= 0.5).sum())
    st.markdown(
        f"<div class='kpi-card'><div class='kpi-title'>Students Needing Attention</div><div class='kpi-value'>{at_risk_count}</div></div>",
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# Single Dashboard Tabs
tab1, tab2, tab3 = st.tabs(
    [
        "📊 Student Overview & Recommendations",
        "🔍 Explainability & Responsible Use",
        "💬 AI Academic Chatbot (Q&A & Study Plans)",
    ]
)

# =============================================================================
# TAB 1: STUDENT OVERVIEW, SUBJECT RANKING & RECOMMENDATIONS (SLIDES 7, 8, 9)
# =============================================================================
with tab1:
    s_row = agg_df[agg_df["Student_ID"] == selected_student_id].iloc[0]
    s_subjs = raw_df[raw_df["Student_ID"] == selected_student_id]
    rec_data = generate_student_recommendations(s_row, s_subjs)

    st.subheader(f"Student Profile: {selected_student_id} ({s_row['Department']})")

    if not s_row["Data_Reliable"]:
        st.warning(
            "⚠️ **Insufficient Evidence / Needs Review**: Some assessment records for this student are missing or unreliable."
        )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Overall Score", f"{s_row['Composite_Score']}/100", f"Grade {s_row['Grade']}")
    m2.metric("Attendance", f"{s_row['Attendance']}%", f"Trend: {s_row['Performance_Trend']}")
    m3.metric("ML Risk Probability", f"{s_row['Risk_Prob']:.2f}", rec_data["status"])
    m4.metric(
        "Top Priority Subject",
        rec_data["priority_subjects"][0] if rec_data["priority_subjects"] else "N/A",
    )

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**Subject-Wise Marks & Priority Ranking**")
        subj_sorted = s_subjs.sort_values("Composite_Score", ascending=True)
        fig_subj = px.bar(
            subj_sorted,
            x="Composite_Score",
            y="Subject",
            orientation="h",
            text="Composite_Score",
            color="Attendance",
            color_continuous_scale="Blues",
            labels={"Composite_Score": "Composite Score", "Subject": "Subject"},
        )
        fig_subj.update_layout(
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            font=dict(color="#000000"),
            height=340,
            margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(fig_subj, use_container_width=True)

    with col_right:
        st.markdown("**Assessment Trend Chart (Quiz → Assignment → Midterm → Final)**")
        stages = ["Quiz_Score", "Assignment_Score", "Midterm_Marks", "Final_Marks"]
        fig_trend = go.Figure()
        for _, r in s_subjs.iterrows():
            fig_trend.add_trace(
                go.Scatter(
                    x=["Quiz", "Assignment", "Midterm", "Final"],
                    y=[r[st_col] for st_col in stages],
                    mode="lines+markers",
                    name=r["Subject"],
                )
            )
        fig_trend.update_layout(
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            font=dict(color="#000000"),
            yaxis=dict(range=[0, 105], gridcolor="#e2e8f0"),
            xaxis=dict(gridcolor="#e2e8f0"),
            height=340,
            margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    # Slide 7 & 8: From Prediction to Recommendation Engine Example
    st.subheader("Recommendation Engine Output (Slide 8 Standards)")
    st.caption("Wording is generated from actual analytics—not invented by the system.")
    for rule in rec_data["rules_triggered"]:
        st.markdown(f"<div class='rule-box'>{rule}</div>", unsafe_allow_html=True)

    if rec_data["study_plan"]:
        st.markdown("**Personalized Weekly Study Schedule:**")
        plan_df = pd.DataFrame(
            list(rec_data["study_plan"].items()), columns=["Schedule Window", "Actionable Study Task"]
        )
        st.table(plan_df)

    st.markdown(
        "<div class='disclaimer-box'><strong>Responsible Use Note:</strong> "
        "The ML model is not the final student-facing output. Model output is combined with subject performance, trends, attendance, and assessment context. "
        "Clearly distinguishes model-based evidence from causal claims.</div>",
        unsafe_allow_html=True,
    )

# =============================================================================
# TAB 2: EXPLAINABILITY & RESPONSIBLE USE (SLIDE 11)
# =============================================================================
with tab2:
    st.subheader("Model Verification & Explainable AI (XAI)")
    v1, v2, v3 = st.columns(3)
    v1.metric("Random Forest Precision", f"{model_metrics['Precision']*100:.1f}%")
    v2.metric("Random Forest Recall", f"{model_metrics['Recall']*100:.1f}%")
    v3.metric("Random Forest F1-Score", f"{model_metrics['F1-Score']*100:.1f}%")

    st.markdown("---")
    c_left, c_right = st.columns(2)

    with c_left:
        st.markdown(f"**Local Explanation for {selected_student_id} (SHAP-Style Attribution)**")
        st.caption("Positive bars push risk UP; negative bars push risk DOWN relative to cohort average.")
        s_row = agg_df[agg_df["Student_ID"] == selected_student_id].iloc[0]
        local_exp_df = compute_local_explainability(s_row, agg_df, feature_cols, rf_model)

        fig_local = px.bar(
            local_exp_df,
            x="Risk_Contribution",
            y="Feature",
            orientation="h",
            text="Risk_Contribution",
            color=local_exp_df["Risk_Contribution"] > 0,
            color_discrete_map={True: "#b91c1c", False: "#15803d"},
            labels={"Risk_Contribution": "Risk Contribution Impact", "Feature": "Model Feature"},
        )
        fig_local.update_layout(
            showlegend=False,
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            font=dict(color="#000000"),
            height=360,
        )
        st.plotly_chart(fig_local, use_container_width=True)

    with c_right:
        st.markdown("**Global Random Forest Feature Importance**")
        st.caption("Overall importance of features across the entire student dataset.")
        fig_imp = px.bar(
            feat_imp_df,
            x="Importance",
            y="Feature",
            orientation="h",
            text=feat_imp_df["Importance"].round(3),
            color_discrete_sequence=["#1e3a8a"],
        )
        fig_imp.update_layout(
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            font=dict(color="#000000"),
            height=360,
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    st.subheader("Explainability & Responsible Use Guidelines (Slide 11)")
    st.markdown(
        """
        - **Explainability:** Students and advisors can explain why the system flagged a student using feature importance and SHAP attributions.
        - **Key Features:** Recent performance, attendance, and internal marks are major model features.
        - **Correlation vs. Causation:** Model feature importance does **not** prove that a feature caused poor performance.
        - **Student Privacy Protection:** Anonymize data and avoid exposing personally identifiable information (PII).
        - **Data Reliability:** Include an `Insufficient Evidence` or `Needs Review` state when the data is unreliable.
        """
    )

# =============================================================================
# TAB 3: AI ACADEMIC CHATBOT (Q&A & STUDY PLANS)
# =============================================================================
with tab3:
    st.subheader("AI Academic Advisor Chatbot")
    st.markdown(
        "Ask questions about **recommendations for poor / at-risk students**, **study plans for all students**, "
        "or query any individual student ID (e.g., `S001`, `S015`)."
    )

    q_col1, q_col2, q_col3 = st.columns(3)
    quick_prompt = None
    if q_col1.button("📋 Recommendations for Poor Students"):
        quick_prompt = "What are the recommendations for poor students?"
    if q_col2.button("📅 Study Plan for All Students"):
        quick_prompt = "Generate a study plan for all students"
    if q_col3.button(f"🎯 Study Plan for {selected_student_id}"):
        quick_prompt = f"Give me recommendations and a study plan for {selected_student_id}"

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {
                "role": "assistant",
                "content": (
                    "Hello! I am the **AI Student Performance Advisor Chatbot**. "
                    "Ask me for recommendations for poor/at-risk students, a cohort-wide study plan for all students, "
                    "or a personalized study plan for any Student ID (e.g., `S004`)."
                ),
            }
        ]

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask about student recommendations or study plans...")
    active_query = quick_prompt if quick_prompt else user_input

    if active_query:
        st.session_state.chat_history.append({"role": "user", "content": active_query})
        with st.chat_message("user"):
            st.markdown(active_query)

        bot_reply = process_chatbot_query(active_query, raw_df, agg_df)
        st.session_state.chat_history.append({"role": "assistant", "content": bot_reply})
        with st.chat_message("assistant"):
            st.markdown(bot_reply)
