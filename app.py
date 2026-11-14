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

st.set_page_config(
    page_title="AI-Based Student Performance Advisor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .stApp { background-color: #f8fafc; color: #000000; }
        h1, h2, h3, h4, h5, h6, p, li, span, label, div { color: #000000 !important; }
        .kpi-container { display: flex; gap: 16px; width: 100%; margin-bottom: 20px; }
        .kpi-card { background-color: #ffffff; border: 1px solid #cbd5e1; border-radius: 8px; padding: 16px; flex: 1; box-shadow: 0 1px 3px rgba(0,0,0,0.05); min-height: 95px; display: flex; flex-direction: column; justify-content: center; }
        .kpi-title { font-size: 11px; font-weight: 700; color: #475569 !important; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px; }
        .kpi-value { font-size: 24px; font-weight: 800; color: #0f172a !important; line-height: 1.2; }
        .kpi-sub { font-size: 12px; color: #64748b !important; margin-top: 4px; }
        .rule-box { background-color: #ffffff; border-left: 5px solid #1e40af; border: 1px solid #cbd5e1; border-left-width: 5px; padding: 14px 18px; border-radius: 6px; margin-bottom: 12px; box-shadow: 0 1px 2px rgba(0,0,0,0.03); }
        .tech-box { background-color: #f8fafc; border: 1px solid #cbd5e1; border-left: 5px solid #0f172a; padding: 14px 18px; border-radius: 6px; margin-bottom: 12px; }
        .disclaimer-box { background-color: #f1f5f9; border: 1px solid #94a3b8; border-radius: 6px; padding: 12px 16px; font-size: 13px; }
        button[data-baseweb="tab"] { font-weight: 700 !important; font-size: 15px !important; color: #000000 !important; }
    </style>
""",
    unsafe_allow_html=True,
)


@st.cache_data
def load_and_prepare_data(file_path="Subject_Analytics_Dataset_500_Students-v2.xlsx"):
    try:
        df = pd.read_excel(file_path)
    except Exception:
        np.random.seed(42)
        students = [f"S{str(i).zfill(3)}" for i in range(1, 501)]
        subjects = ["Statistics", "Data Structures", "Machine Learning", "Database Systems"]
        depts = ["Computer Science", "Data Science", "Information Systems", "Software Engineering"]
        rows = []
        for s_id in students:
            dept = np.random.choice(depts)
            base_perf = np.random.normal(70, 15)
            base_att = np.clip(np.random.normal(82, 10), 40, 100)
            for subj in subjects:
                att = round(np.clip(base_att + np.random.normal(0, 5), 35, 100), 1)
                quiz = round(np.clip(base_perf + np.random.normal(0, 8), 20, 100), 1)
                assign = round(np.clip(base_perf + np.random.normal(2, 7), 25, 100), 1)
                midsem = round(np.clip(base_perf + np.random.normal(-3, 10), 15, 100), 1)
                comp = round(np.clip(midsem + np.random.normal(1, 9), 15, 100), 1)
                prev = round(np.clip(base_perf + np.random.normal(1, 6), 30, 100), 1)
                rec = round(np.clip((midsem + comp) / 2 + np.random.normal(0, 5), 20, 100), 1)
                missed = int(np.clip(np.random.poisson(1.5 if att < 75 else 0.4), 0, 8))
                rows.append(
                    {
                        "student_id": s_id,
                        "department": dept,
                        "subject": subj,
                        "attendance": att,
                        "assignment_avg": assign,
                        "quiz_avg": quiz,
                        "Midsem": midsem,
                        "Comprehensive marks": comp,
                        "previous_avg": prev,
                        "recent_avg": rec,
                        "missed_assessments": missed,
                    }
                )
        df = pd.DataFrame(rows)

    alias_map = {
        "Student ID": "student_id",
        "Student_ID": "student_id",
        "Subject": "subject",
        "Department": "department",
        "Attendance": "attendance",
        "attendance": "attendance",
        "assignment_avg": "assignment_avg",
        "quiz_avg": "quiz_avg",
        "Midsem": "Midsem",
        "Comprehensive marks": "Comprehensive marks",
        "previous_avg": "previous_avg",
        "recent_avg": "recent_avg",
        "missed_assessments": "missed_assessments",
    }
    df = df.rename(columns={k: v for k, v in alias_map.items() if k in df.columns})
    pii_cols = [c for c in df.columns if c.lower() in ["student_name", "name", "email", "phone"]]
    if pii_cols:
        df = df.drop(columns=pii_cols)

    if "subject" not in df.columns:
        df["subject"] = "Core Module"
    if "department" not in df.columns:
        df["department"] = "B.Sc Design & Computing"

    num_cols = [
        "attendance",
        "quiz_avg",
        "assignment_avg",
        "Midsem",
        "Comprehensive marks",
        "previous_avg",
        "recent_avg",
        "missed_assessments",
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Data_Reliable"] = ~df[num_cols].isna().any(axis=1) & (df["attendance"].between(0, 100))
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    df["Composite_Score"] = (
        0.10 * df["quiz_avg"]
        + 0.15 * df["assignment_avg"]
        + 0.25 * df["Midsem"]
        + 0.35 * df["Comprehensive marks"]
        + 0.15 * df["previous_avg"]
    ).round(1)

    df["Trend_Delta"] = (df["recent_avg"] - df["previous_avg"]).round(1)
    df["Performance_Trend"] = np.where(
        df["Trend_Delta"] > 2, "Improving", np.where(df["Trend_Delta"] < -3, "Declining", "Stable")
    )

    agg_df = (
        df.groupby("student_id", as_index=False)
        .agg(
            department=("department", "first"),
            attendance=("attendance", "mean"),
            quiz_avg=("quiz_avg", "mean"),
            assignment_avg=("assignment_avg", "mean"),
            Midsem=("Midsem", "mean"),
            Comprehensive_marks=("Comprehensive marks", "mean"),
            previous_avg=("previous_avg", "mean"),
            recent_avg=("recent_avg", "mean"),
            Composite_Score=("Composite_Score", "mean"),
            Trend_Delta=("Trend_Delta", "mean"),
            missed_assessments=("missed_assessments", "sum"),
            Data_Reliable=("Data_Reliable", "all"),
        )
        .round(1)
    )

    agg_df["Performance_Trend"] = np.where(
        agg_df["Trend_Delta"] > 2, "Improving", np.where(agg_df["Trend_Delta"] < -3, "Declining", "Stable")
    )
    agg_df["Risk_Target"] = np.where(
        (agg_df["Composite_Score"] < 58)
        | (agg_df["attendance"] < 72)
        | (agg_df["missed_assessments"] >= 3),
        1,
        0,
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


@st.cache_resource
def train_ml_models(agg_df):
    feature_cols = [
        "attendance",
        "quiz_avg",
        "assignment_avg",
        "Midsem",
        "Comprehensive_marks",
        "previous_avg",
        "recent_avg",
        "Trend_Delta",
        "missed_assessments",
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

    friendly_names = {
        "attendance": "Attendance %",
        "quiz_avg": "Quiz Average",
        "assignment_avg": "Assignment Average",
        "Midsem": "Mid-Semester Score",
        "Comprehensive_marks": "Comprehensive Exam Score",
        "previous_avg": "Previous Term Average",
        "recent_avg": "Recent Assessment Average",
        "Trend_Delta": "Score Trajectory Change",
        "missed_assessments": "Missed Assessments",
    }

    importances = pd.DataFrame(
        {
            "Factor": [friendly_names[c] for c in feature_cols],
            "Influence": rf.feature_importances_,
        }
    ).sort_values("Influence", ascending=True)
    return rf, lr, risk_prob, metrics, importances, feature_cols, friendly_names


def compute_local_explainability(student_row, agg_df, feature_cols, rf_model, friendly_names):
    cohort_means = agg_df[feature_cols].mean()
    importances = dict(zip(feature_cols, rf_model.feature_importances_))
    contributions = []
    for feat in feature_cols:
        val = student_row[feat]
        mean_val = cohort_means[feat]
        imp = importances.get(feat, 0.05)
        diff = (
            (val - mean_val) / (agg_df[feat].std() + 1e-5)
            if feat in ["missed_assessments"]
            else (mean_val - val) / (agg_df[feat].std() + 1e-5)
        )
        risk_impact = round(diff * imp * 100, 2)
        contributions.append(
            {
                "Factor": friendly_names[feat],
                "Student_Value": val,
                "Cohort_Avg": round(mean_val, 1),
                "Risk_Impact": risk_impact,
            }
        )
    return pd.DataFrame(contributions).sort_values("Risk_Impact", ascending=True)


def generate_student_recommendations(student_row, subject_rows):
    if not student_row["Data_Reliable"]:
        return {
            "status": "Insufficient Evidence / Needs Review",
            "priority_subjects": [],
            "rules_triggered": [
                "Academic records are incomplete or require verification before generating automated guidance."
            ],
            "study_plan": {},
        }
    subj_df = subject_rows.copy()
    subj_df["Priority_Score"] = (100 - subj_df["Composite_Score"]) + np.where(
        subj_df["attendance"] < 75, 15, 0
    )
    subj_df = subj_df.sort_values("Priority_Score", ascending=False)
    priority_subjects = subj_df["subject"].tolist()
    rules_triggered = []
    for _, s_row in subj_df.iterrows():
        if (
            s_row["Composite_Score"] < 60
            and s_row["Performance_Trend"] == "Declining"
            and s_row["attendance"] < 75
        ):
            rules_triggered.append(
                f'Advisor Guidance: "{s_row["subject"]} currently requires additional attention. Your recent scores are declining (Mid-Semester: {s_row["Midsem"]} → Comprehensive: {s_row["Comprehensive marks"]}, Average: {s_row["Composite_Score"]}%) and attendance is below the target threshold ({s_row["attendance"]}%). Review recent assessment topics and prioritize the upcoming {s_row["subject"]} evaluation."'
            )
        elif s_row["Composite_Score"] < 60:
            rules_triggered.append(
                f'Advisor Guidance: "{s_row["subject"]} average is {s_row["Composite_Score"]}%. Review recent assessment topics and prioritize upcoming coursework."'
            )
    if not rules_triggered:
        rules_triggered.append(
            'Advisor Guidance: "Academic performance is steady across all subjects. Maintain your current study routine and explore advanced practice exercises."'
        )
    top_subj = priority_subjects[0] if len(priority_subjects) > 0 else "Core Subject"
    sec_subj = priority_subjects[1] if len(priority_subjects) > 1 else top_subj
    study_plan = {
        "Monday – Tuesday (Primary Focus)": f"Dedicate 2 hrs/day on **{top_subj}** (top priority). Review recent lecture notes and practice past exam questions.",
        "Wednesday – Thursday (Secondary Reinforcement)": f"Spend 1.5 hrs/day on **{sec_subj}** and complete upcoming assignments ahead of deadlines.",
        "Friday (Self-Assessment & Quizzes)": "Take 1 timed practice quiz (45 mins) covering challenging topics and review mistakes.",
        "Weekend (Weekly Consolidation)": "Summarize key concepts (1 hr) and preview next week's syllabus to prepare for classes.",
    }
    status = "Needs Academic Support" if student_row["Risk_Prob"] >= 0.5 else "On Track"
    return {
        "status": status,
        "priority_subjects": priority_subjects,
        "rules_triggered": rules_triggered,
        "study_plan": study_plan,
    }


def process_chatbot_query(user_query, raw_df, agg_df):
    q = user_query.lower().strip()
    id_match = re.search(r"\bs\d{3}\b", q)
    at_risk_df = agg_df[agg_df["Risk_Prob"] >= 0.5].sort_values("Risk_Prob", ascending=False)
    if id_match:
        s_id = id_match.group(0).upper()
        if s_id not in agg_df["student_id"].values:
            return f"Student ID **{s_id}** was not found in the records."
        s_row = agg_df[agg_df["student_id"] == s_id].iloc[0]
        s_subjs = raw_df[raw_df["student_id"] == s_id]
        rec = generate_student_recommendations(s_row, s_subjs)
        response = f"### Academic Guidance & Study Plan for **{s_id}**\n- **Department:** {s_row['department']} | **Status:** {rec['status']} (Support Need Index: {s_row['Risk_Prob']:.2f})\n- **Overall Score:** {s_row['Composite_Score']}/100 (Grade {s_row['Grade']}) | **Attendance:** {s_row['attendance']}%\n- **Subject Priority Order:** {', '.join(rec['priority_subjects'])}\n\n**Actionable Recommendations:**\n"
        for r in rec["rules_triggered"]:
            response += f"- {r}\n"
        response += "\n**Personalized Weekly Study Schedule:**\n"
        for day, plan in rec["study_plan"].items():
            response += f"- **{day}:** {plan}\n"
        return response
    if ("study plan" in q or "schedule" in q or "timetable" in q) and (
        "all" in q or "everyone" in q or "class" in q or "cohort" in q
    ):
        return f"### Cohort-Wide Tiered Study Schedule ({len(agg_df)} Students)\n\n#### Group 1: Students Needing Immediate Support ({len(at_risk_df)} Students)\n- **Daily Focus (2.5 hrs/day):** Prioritize lowest-scoring subjects first and review foundational concepts.\n- **Attendance Target:** Maintain ≥ 85% class attendance and attend weekly doubt-clearing sessions.\n\n#### Group 2: Moderate Performers (Score 60–74)\n- **Daily Focus (1.5 hrs/day):** Review lecture notes within 24 hours and submit assignments early.\n\n#### Group 3: High Performers (Score ≥ 75)\n- **Daily Focus (1 hr/day):** Maintain current study habits and engage in peer mentoring."
    if any(k in q for k in ["poor", "weak", "risk", "low", "failing", "support", "recommend", "help"]):
        if at_risk_df.empty:
            return "All students are currently performing steadily."
        top_risk = at_risk_df.head(8)
        resp = f"### Priority Support Recommendations ({len(at_risk_df)} Students Flagged)\n"
        for _, row in top_risk.iterrows():
            resp += f"- **{row['student_id']}** ({row['department']}) — Overall Avg: `{row['Composite_Score']}`, Support Index: `{row['Risk_Prob']}`\n"
        return resp
    return "I am your **Academic Advisor Assistant**. Ask me for recommendations for students needing support, a cohort-wide study plan, or a personalized plan for any Student ID (e.g., `S004`)."
raw_df, agg_df = load_and_prepare_data()
rf_model, lr_model, risk_probs, model_metrics, feat_imp_df, feature_cols, friendly_names = train_ml_models(
    agg_df
)
agg_df["Risk_Prob"] = risk_probs.round(3)

st.sidebar.title("🎓 Cohort & Student Filters")
dept_options = ["All Departments"] + sorted(agg_df["department"].unique().tolist())
selected_dept = st.sidebar.selectbox("Department", dept_options)
filtered_agg = (
    agg_df if selected_dept == "All Departments" else agg_df[agg_df["department"] == selected_dept]
)
selected_student_id = st.sidebar.selectbox(
    "Select Student ID", filtered_agg["student_id"].tolist()
)

st.title("AI-Based Student Performance Advisor and Visualization System")
st.markdown(
    "Unified academic monitoring, data-driven study recommendations, performance factor insights, and interactive advisory assistant."
)

at_risk_count = int((filtered_agg["Risk_Prob"] >= 0.5).sum())
cohort_avg_score = f"{filtered_agg['Composite_Score'].mean():.1f}%"
cohort_avg_att = f"{filtered_agg['attendance'].mean():.1f}%"

st.markdown(
    f"""
    <div class="kpi-container">
        <div class="kpi-card"><div class="kpi-title">Total Students</div><div class="kpi-value">{len(filtered_agg)}</div><div class="kpi-sub">Active Records</div></div>
        <div class="kpi-card"><div class="kpi-title">Cohort Avg Score</div><div class="kpi-value">{cohort_avg_score}</div><div class="kpi-sub">Weighted Average</div></div>
        <div class="kpi-card"><div class="kpi-title">Avg Attendance</div><div class="kpi-value">{cohort_avg_att}</div><div class="kpi-sub">Overall Presence</div></div>
        <div class="kpi-card"><div class="kpi-title">Students Needing Attention</div><div class="kpi-value" style="color: #b91c1c !important;">{at_risk_count}</div><div class="kpi-sub">Priority Support Group</div></div>
    </div>
""",
    unsafe_allow_html=True,
)

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📊 Student Overview & Recommendations",
        "🔍 Performance Insights & Responsible Use",
        "💬 Academic Advisor Assistant",
        "⚙️ System Technicalities & Functionalities",
    ]
)

with tab1:
    s_row = agg_df[agg_df["student_id"] == selected_student_id].iloc[0]
    s_subjs = raw_df[raw_df["student_id"] == selected_student_id]
    rec_data = generate_student_recommendations(s_row, s_subjs)

    st.subheader(f"Student Profile: {selected_student_id} ({s_row['department']})")
    if not s_row["Data_Reliable"]:
        st.warning("⚠️ **Insufficient Evidence / Needs Review**")

    top_subj = rec_data["priority_subjects"][0] if rec_data["priority_subjects"] else "N/A"
    st.markdown(
        f"""
        <div class="kpi-container">
            <div class="kpi-card"><div class="kpi-title">Overall Score</div><div class="kpi-value">{s_row['Composite_Score']}/100</div><div class="kpi-sub">Grade {s_row['Grade']}</div></div>
            <div class="kpi-card"><div class="kpi-title">Attendance</div><div class="kpi-value">{s_row['attendance']}%</div><div class="kpi-sub">Trend: {s_row['Performance_Trend']}</div></div>
            <div class="kpi-card"><div class="kpi-title">Support Need Index</div><div class="kpi-value">{s_row['Risk_Prob']:.2f}</div><div class="kpi-sub">{rec_data['status']}</div></div>
            <div class="kpi-card"><div class="kpi-title">Top Priority Subject</div><div class="kpi-value" style="font-size: 18px !important; margin-top: 4px;">{top_subj}</div><div class="kpi-sub">Immediate Focus Area</div></div>
        </div>
    """,
        unsafe_allow_html=True,
    )

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("**Subject-Wise Marks & Priority Ranking**")
        subj_sorted = s_subjs.sort_values("Composite_Score", ascending=True)
        fig_subj = px.bar(
            subj_sorted,
            x="Composite_Score",
            y="subject",
            orientation="h",
            text="Composite_Score",
            color="attendance",
            color_continuous_scale="Blues",
            labels={"Composite_Score": "Overall Subject Score", "subject": "Subject"},
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
        st.markdown("**Assessment Progress Chart**")
        stages = ["quiz_avg", "assignment_avg", "Midsem", "Comprehensive marks"]
        fig_trend = go.Figure()
        for _, r in s_subjs.iterrows():
            fig_trend.add_trace(
                go.Scatter(
                    x=["Quiz", "Assignment", "Mid-Semester", "Comprehensive"],
                    y=[r[st_col] for st_col in stages],
                    mode="lines+markers",
                    name=r["subject"],
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

    st.subheader("Personalized Recommendations")
    st.caption("Guidance is generated directly from actual academic records.")
    for rule in rec_data["rules_triggered"]:
        st.markdown(f"<div class='rule-box'>{rule}</div>", unsafe_allow_html=True)

    if rec_data["study_plan"]:
        st.markdown("**Weekly Study Schedule:**")
        st.table(
            pd.DataFrame(
                list(rec_data["study_plan"].items()),
                columns=["Schedule Window", "Actionable Study Task"],
            )
        )

    st.markdown(
        "<div class='disclaimer-box'><strong>Advisory Note:</strong> Automated assessments combine subject scores, recent score trends, attendance, and coursework completion to highlight areas needing attention.</div>",
        unsafe_allow_html=True,
    )

with tab2:
    st.subheader("System Reliability & Key Performance Drivers")
    st.markdown(
        f"""
        <div class="kpi-container">
            <div class="kpi-card"><div class="kpi-title">Alert Accuracy</div><div class="kpi-value">{model_metrics['Precision']*100:.1f}%</div><div class="kpi-sub">Reliability Score</div></div>
            <div class="kpi-card"><div class="kpi-title">Detection Coverage</div><div class="kpi-value">{model_metrics['Recall']*100:.1f}%</div><div class="kpi-sub">Reliability Score</div></div>
            <div class="kpi-card"><div class="kpi-title">Overall Consistency</div><div class="kpi-value">{model_metrics['F1-Score']*100:.1f}%</div><div class="kpi-sub">Reliability Score</div></div>
        </div>
    """,
        unsafe_allow_html=True,
    )

    c_left, c_right = st.columns(2)
    with c_left:
        st.markdown(f"**Individual Factor Breakdown for {selected_student_id}**")
        st.caption(
            "Red bars indicate factors increasing support need; green bars indicate strengths relative to cohort average."
        )
        local_exp_df = compute_local_explainability(
            s_row, agg_df, feature_cols, rf_model, friendly_names
        )
        fig_local = px.bar(
            local_exp_df,
            x="Risk_Impact",
            y="Factor",
            orientation="h",
            text="Risk_Impact",
            color=local_exp_df["Risk_Impact"] > 0,
            color_discrete_map={True: "#b91c1c", False: "#15803d"},
            labels={"Risk_Impact": "Impact on Support Need", "Factor": "Academic Factor"},
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
        st.markdown("**Overall Academic Factors Across All Students**")
        st.caption("Relative weight of each academic metric across the full student cohort.")
        fig_imp = px.bar(
            feat_imp_df,
            x="Influence",
            y="Factor",
            orientation="h",
            text=feat_imp_df["Influence"].round(3),
            color_discrete_sequence=["#1e3a8a"],
            labels={"Influence": "Relative Weight", "Factor": "Academic Factor"},
        )
        fig_imp.update_layout(
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            font=dict(color="#000000"),
            height=360,
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    st.subheader("Transparency & Responsible Use Guidelines")
    st.markdown(
        """
        - **Clear Explanations:** Students and advisors can see exactly which academic metrics contributed to a support alert.
        - **Primary Drivers:** Recent assessment trends, attendance percentages, and exam scores carry the strongest influence.
        - **Context Over Correlation:** Statistical patterns highlight where academic support is helpful—they do not assume root causes.
        - **Privacy Protection:** Student identities remain strictly anonymized using unique student IDs.
        - **Data Completeness:** Incomplete records are flagged for manual advisor review rather than automated scoring.
    """
    )

with tab3:
    st.subheader("Academic Advisor Assistant")
    st.markdown(
        "Ask questions about **recommendations for students needing support**, **cohort study schedules**, or query an individual student ID (e.g., `S001`, `S015`)."
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {
                "role": "assistant",
                "content": "Hello! I am your **Academic Advisor Assistant**. Ask me for recommendations for students needing support, a cohort-wide study plan, or a personalized plan for any Student ID (e.g., `S004`).",
            }
        ]
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask about student recommendations or study plans...")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        bot_reply = process_chatbot_query(user_input, raw_df, agg_df)
        st.session_state.chat_history.append({"role": "assistant", "content": bot_reply})
        with st.chat_message("assistant"):
            st.markdown(bot_reply)

with tab4:
    st.subheader("⚙️ System Technicalities & Functionalities Architecture")
    st.markdown(
        "This tab outlines the backend data structures, machine learning engine workflows, feature engineering formulas, and decision rule logic powering the system."
    )

    st.markdown(
        """
        <div class="tech-box">
            <h4>1. Dataset Ingestion & Schema Structure</h4>
            <p><strong>Source File:</strong> <code>Subject_Analytics_Dataset_500_Students-v2.xlsx</code> (Multi-subject records across 500 unique students).</p>
            <p><strong>Attributes Tracked:</strong> Student ID, Department, Subject Name, Attendance %, Quiz Average, Assignment Average, Mid-Semester Marks, Comprehensive Examination Marks, Previous Term Average, Recent Assessment Average, and Missed Assessments.</p>
        </div>

        <div class="tech-box">
            <h4>2. Dynamic Feature Engineering & Data Reliability</h4>
            <p><strong>Composite Score Calculation:</strong><br>
            <code>Composite Score = (0.10 × Quiz) + (0.15 × Assignment) + (0.25 × Mid-Sem) + (0.35 × Comprehensive) + (0.15 × Previous Avg)</code></p>
            <p><strong>Performance Trajectory (Trend Delta):</strong><br>
            Computed via <code>Recent Average - Previous Average</code>. Categorized dynamically as <em>Improving</em> (>+2), <em>Declining</em> (<-3), or <em>Stable</em>.</p>
            <p><strong>Data Reliability Verification:</strong><br>
            Automated validation checks ensure attendance percentages fall between 0–100% and no mandatory assessment records are missing. Incomplete records trigger an <em>Insufficient Evidence / Needs Review</em> flag.</p>
        </div>

        <div class="tech-box">
            <h4>3. Machine Learning & Predictive Engine Architecture</h4>
            <p><strong>Target Variable Definition:</strong> Binary risk classification (<code>0</code> = Stable/On Track, <code>1</code> = Requires Academic Attention) derived from composite thresholds (<58%), attendance (<72%), or missed milestones (≥3).</p>
            <p><strong>Baseline Model:</strong> <code>LogisticRegression</code> (max_iter=1000) for linear baseline probability benchmarking.</p>
            <p><strong>Primary Model:</strong> <code>RandomForestClassifier</code> (n_estimators=120) capturing non-linear interactions across attendance, continuous evaluation scores, and assessment trajectories.</p>
            <p><strong>Evaluation Metrics:</strong> Precision, Recall, and F1-Score computed on a stratified 75/25 train-test split.</p>
        </div>

        <div class="tech-box">
            <h4>4. Recommendation & Chatbot NLP Engine Logic</h4>
            <p><strong>Priority Scoring Formula:</strong><br>
            <code>Priority Score = (100 - Composite Score) + (15 if Attendance < 75% else 0)</code></p>
            <p><strong>Rule-Based Advisory Engine:</strong> Matches student performance patterns against institutional criteria to generate tailored remediation schedules and subject priority orders.</p>
            <p><strong>Conversational NLP Interface:</strong> Utilizes regex pattern matching to parse student IDs (e.g., <code>S001</code>) and keyword queries (e.g., "study plan", "poor students"), extracting live Pandas dataframe records instantly.</p>
        </div>
    """,
        unsafe_allow_html=True,
    )
