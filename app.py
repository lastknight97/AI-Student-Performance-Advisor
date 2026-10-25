import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# ============================================================
# PAGE
# ============================================================
st.set_page_config(
    page_title="BITS Pilani | AI Student Performance Advisor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# ROBUST DATA PATH RESOLUTION
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Primary path check
DATA_PATH = os.path.join(BASE_DIR, "data", "Subject_Analytics_Dataset_200_Students.xlsx")

# Fallback path check if placed directly in root
if not os.path.exists(DATA_PATH):
    alt_path = os.path.join(BASE_DIR, "Subject_Analytics_Dataset_200_Students.xlsx")
    if os.path.exists(alt_path):
        DATA_PATH = alt_path

REQUIRED_COLUMNS = [
    "Student_ID",
    "Subject",
    "Attendance",
    "Assignment",
    "Quiz_Score",
    "Midterm_Marks",
    "Final_Marks",
]

FEATURE_COLUMNS = [
    "Attendance",
    "Assignment",
    "Quiz_Score",
    "Midterm_Marks",
    "Final_Marks",
    "missed_assessments",
]

# ============================================================
# MODERN LIGHT ACADEMIC THEME
# ============================================================
st.markdown(
    """
    <style>
    :root {
        --bg: #f4f7fb;
        --card: #ffffff;
        --line: #d7dee8;
        --ink: #000000;
        --muted: #475569;
        --blue: #2563eb;
        --green: #16a34a;
        --amber: #d97706;
        --red: #dc2626;
    }
    .stApp {
        background: var(--bg);
        color: var(--ink);
    }
    .main .block-container {
        max-width: 1450px;
        padding: 1.1rem 2rem 3rem;
    }
    /* Typography */
    h1, h2, h3, h4, h5, h6, p, span, label, li, td, th {
        color: var(--ink) !important;
    }
    h1 {
        font-size: clamp(2rem, 4vw, 3.6rem) !important;
        line-height: 1.03 !important;
        letter-spacing: -1.8px;
        margin-bottom: .25rem !important;
    }
    h2 {
        font-size: 1.65rem !important;
        letter-spacing: -.4px;
    }
    h3 {
        font-size: 1.15rem !important;
    }
    .stCaption, [data-testid="stCaptionContainer"] {
        color: var(--muted) !important;
    }
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid var(--line);
    }
    section[data-testid="stSidebar"] * {
        color: var(--ink) !important;
    }
    section[data-testid="stSidebar"] .stMarkdown h2 {
        font-size: 1.15rem !important;
        margin-bottom: .25rem;
    }
    /* Selectbox - closed state */
    div[data-baseweb="select"] > div {
        background: #ffffff !important;
        border: 1px solid #aeb9c8 !important;
        border-radius: 10px !important;
        min-height: 44px !important;
        color: #000000 !important;
    }
    div[data-baseweb="select"] * {
        color: #000000 !important;
    }
    /* Selectbox popup */
    div[data-baseweb="popover"], div[data-baseweb="menu"], div[role="listbox"] {
        background: #ffffff !important;
        color: #000000 !important;
        border: 1px solid #cbd5e1 !important;
        box-shadow: 0 12px 35px rgba(15, 23, 42, .16) !important;
        border-radius: 10px !important;
    }
    div[role="option"] {
        background: #ffffff !important;
        color: #000000 !important;
    }
    div[role="option"]:hover, div[role="option"][aria-selected="true"] {
        background: #eaf2ff !important;
        color: #000000 !important;
    }
    /* Cards */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--card) !important;
        border: 1px solid var(--line) !important;
        border-radius: 16px !important;
        box-shadow: 0 5px 18px rgba(15, 23, 42, .06) !important;
    }
    /* Metrics */
    div[data-testid="stMetric"] {
        background: #ffffff !important;
        border: 1px solid var(--line) !important;
        border-radius: 15px !important;
        padding: 16px 18px !important;
        min-height: 118px !important;
        box-shadow: 0 5px 18px rgba(15, 23, 42, .06) !important;
    }
    div[data-testid="stMetricLabel"] {
        color: #334155 !important;
        font-size: .82rem !important;
        font-weight: 750 !important;
    }
    div[data-testid="stMetricValue"] {
        color: #000000 !important;
        font-size: 1.85rem !important;
        font-weight: 850 !important;
    }
    div[data-testid="stMetricDelta"] {
        color: #000000 !important;
    }
    /* Buttons */
    .stButton > button {
        background: #ffffff !important;
        color: #000000 !important;
        border: 1px solid #aeb9c8 !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
    }
    .stButton > button:hover {
        border-color: #2563eb !important;
        color: #000000 !important;
    }
    /* Tabs */
    button[data-baseweb="tab"] {
        color: #000000 !important;
        font-weight: 750 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #000000 !important;
        border-bottom: 3px solid #2563eb !important;
    }
    /* Tables */
    div[data-testid="stDataFrame"] {
        border: 1px solid var(--line);
        border-radius: 12px;
        overflow: hidden;
    }
    /* Alert text */
    div[data-testid="stAlert"] * {
        color: #000000 !important;
    }
    /* Remove unnecessary Streamlit chrome */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# DATA LOADING
# ============================================================
@st.cache_data(show_spinner="Reading the supplied BITS Pilani dataset...")
def load_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset file could not be located at resolved path: '{path}'. "
            "Please ensure 'Subject_Analytics_Dataset_200_Students.xlsx' is placed in a 'data' folder next to app.py."
        )
    
    try:
        df = pd.read_excel(path, engine="openpyxl")
    except Exception as exc:
        raise RuntimeError(
            f"Could not read the Excel dataset via openpyxl. Make sure openpyxl is installed and the file is not corrupted or open in Excel: {exc}"
        )

    if df.empty:
        raise ValueError("The supplied Excel file contains no records.")

    # Flexible aliases for the supplied workbook.
    aliases = {
        "Student ID": "Student_ID",
        "student_id": "Student_ID",
        "StudentID": "Student_ID",
        "Subject Name": "Subject",
        "subject": "Subject",
        "Attendance %": "Attendance",
        "Attendance Rate": "Attendance",
        "attendance": "Attendance",
        "Assignment Avg": "Assignment",
        "Assignment Average": "Assignment",
        "assignment_avg": "Assignment",
        "Assignment_Score": "Assignment",
        "Quiz %": "Quiz_Score",
        "Quiz Score": "Quiz_Score",
        "quiz_avg": "Quiz_Score",
        "Midsem": "Midterm_Marks",
        "Midsem Marks": "Midterm_Marks",
        "midsem_marks": "Midterm_Marks",
        "Comprehensive": "Final_Marks",
        "Comprehensive Marks": "Final_Marks",
        "Final Marks": "Final_Marks",
        "final_marks": "Final_Marks",
    }
    df = df.rename(columns={c: aliases.get(c, c) for c in df.columns})

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
            + ".\n\nAvailable columns found in file: "
            + ", ".join(str(c) for c in df.columns)
        )

    df = df.copy()
    name_aliases = {
        "Student Name": "Student_Name",
        "student_name": "Student_Name",
        "Name": "Student_Name",
        "name": "Student_Name",
    }
    df = df.rename(columns={c: name_aliases.get(c, c) for c in df.columns})

    df["Student_ID"] = df["Student_ID"].astype(str).str.strip()
    df["Subject"] = df["Subject"].astype(str).str.strip()

    if "Student_Name" in df.columns:
        df["Student_Name"] = df["Student_Name"].fillna("").astype(str).str.strip()

    numeric = [
        "Attendance",
        "Assignment",
        "Quiz_Score",
        "Midterm_Marks",
        "Final_Marks",
    ]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Student_ID", "Subject"] + numeric)
    if df.empty:
        raise ValueError("No usable academic records remain after numeric validation.")

    aggregation = {col: "mean" for col in numeric}
    if "Student_Name" in df.columns:
        aggregation["Student_Name"] = "first"

    df = df.groupby(["Student_ID", "Subject"], as_index=False).agg(aggregation)

    df["previous_avg"] = df["Midterm_Marks"]
    df["recent_avg"] = df["Final_Marks"]

    df["performance_trend"] = np.select(
        [
            df["Final_Marks"] > df["Midterm_Marks"],
            df["Final_Marks"] < df["Midterm_Marks"],
        ],
        ["Improving", "Declining"],
        default="Stable",
    )

    original_missed = None
    for candidate in ["missed_assessments", "Missed Assessments", "Missed_Assessments"]:
        if candidate in df.columns:
            original_missed = candidate
            break

    if original_missed:
        df["missed_assessments"] = (
            pd.to_numeric(df[original_missed], errors="coerce")
            .fillna(0)
            .clip(0, 5)
            .astype(int)
        )
    else:
        df["missed_assessments"] = np.where(df["Attendance"] < 75, 2, 0)

    composite = (
        df["Attendance"] * 0.25
        + df["Final_Marks"] * 0.35
        + df["Midterm_Marks"] * 0.25
        + df["Quiz_Score"] * 0.15
    )
    df["needs_support_target"] = (composite < 58).astype(int)

    df["Grade"] = np.select(
        [
            df["Final_Marks"] >= 80,
            df["Final_Marks"] >= 70,
            df["Final_Marks"] >= 60,
            df["Final_Marks"] >= 50,
        ],
        ["A", "B", "C", "D"],
        default="F",
    )

    return df

# ============================================================
# MACHINE LEARNING
# ============================================================
@st.cache_resource(show_spinner="Training academic support models...")
def train_models(df: pd.DataFrame):
    X = df[FEATURE_COLUMNS].astype(float)
    y = df["needs_support_target"].astype(int)

    if y.nunique() < 2:
        raise ValueError("Model training requires both academic-support classes in the supplied dataset.")

    counts = y.value_counts()
    stratify = y if counts.min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=stratify
    )

    logistic = LogisticRegression(random_state=42, max_iter=1000)
    forest = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")

    logistic.fit(X_train, y_train)
    forest.fit(X_train, y_train)

    def metrics_for(model):
        pred = model.predict(X_test)
        return {
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1-Score": f1_score(y_test, pred, zero_division=0),
            "Confusion Matrix": confusion_matrix(y_test, pred),
        }

    return {
        "logistic": logistic,
        "forest": forest,
        "logistic_metrics": metrics_for(logistic),
        "forest_metrics": metrics_for(forest),
    }

# ============================================================
# PLOT STYLE
# ============================================================
def style_plot(fig, title=None):
    fig.update_layout(
        title={"text": title or "", "font": {"color": "#000000", "size": 18}, "x": 0},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font={"color": "#000000", "size": 13},
        margin={"l": 45, "r": 30, "t": 60, "b": 45},
        legend={"font": {"color": "#000000"}, "bgcolor": "#ffffff"},
    )
    fig.update_xaxes(color="#000000", tickfont={"color": "#000000"}, title_font={"color": "#000000"}, gridcolor="#d7dee8", zerolinecolor="#64748b")
    fig.update_yaxes(color="#000000", tickfont={"color": "#000000"}, title_font={"color": "#000000"}, gridcolor="#d7dee8", zerolinecolor="#64748b")
    return fig

def get_probability(model, row):
    values = row[FEATURE_COLUMNS].astype(float).to_numpy().reshape(1, -1)
    probabilities = model.predict_proba(values)[0]
    classes = list(model.classes_)
    return float(probabilities[classes.index(1)] if 1 in classes else 0.0)

# ============================================================
# LOAD WITH DETAILED EXCEPTION LOGGING
# ============================================================
try:
    data = load_dataset(DATA_PATH)
except Exception as exc:
    st.error("⚠️ The supplied Excel dataset could not be loaded.")
    st.markdown(f"**Checked File Path:** `{DATA_PATH}`")
    st.exception(exc)
    st.stop()

try:
    models = train_models(data)
except Exception as exc:
    st.error("⚠️ The academic support model could not be trained.")
    st.exception(exc)
    st.stop()

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## 🎓 BITS Pilani")
    st.markdown("### AI Student Performance Advisor")
    st.caption("Academic Learning Support & Performance Visualization")

    st.divider()
    st.markdown("#### Dataset")
    st.success("Supplied Excel dataset loaded")
    st.caption(DATA_PATH)
    st.divider()

    student_directory = data[["Student_ID"]].drop_duplicates().copy()
    if "Student_Name" in data.columns:
        names = data[["Student_ID", "Student_Name"]].drop_duplicates("Student_ID").copy()
        student_directory = student_directory.merge(names, on="Student_ID", how="left")
        student_directory["Student_Name"] = student_directory["Student_Name"].fillna("").astype(str).str.strip()
    else:
        student_directory["Student_Name"] = ""

    student_directory = student_directory.sort_values("Student_ID")

    def student_label(row):
        name = row["Student_Name"]
        sid = row["Student_ID"]
        return f"{sid} — {name}" if name else sid

    student_directory["Display"] = student_directory.apply(student_label, axis=1)
    display_to_id = dict(zip(student_directory["Display"], student_directory["Student_ID"]))

    selected_display = st.selectbox(
        "Student",
        student_directory["Display"].tolist(),
        index=0,
        key="student_selector",
    )
    selected_student = display_to_id[selected_display]

    model_choice = st.radio(
        "Academic Support Model",
        ["Random Forest (Main)", "Logistic Regression (Baseline)"],
        key="model_selector",
    )

    st.divider()
    st.markdown("#### Dataset Coverage")
    st.metric("Students", f"{data['Student_ID'].nunique():,}")
    st.metric("Academic Records", f"{len(data):,}")
    st.metric("Subjects", f"{data['Subject'].nunique():,}")

# ============================================================
# SELECTED STUDENT
# ============================================================
student_df = data[data["Student_ID"] == selected_student].copy()
if student_df.empty:
    st.error("No academic records found for this student.")
    st.stop()

if "Student_Name" in student_df.columns:
    student_name = str(student_df["Student_Name"].iloc[0]).strip()
    if not student_name or student_name.lower() == "nan":
        student_name = selected_student
else:
    student_name = selected_student

student_features = student_df[FEATURE_COLUMNS].mean().to_frame().T

selected_model = models["forest"] if model_choice.startswith("Random") else models["logistic"]
risk_probability = get_probability(selected_model, student_features.iloc[0])
risk_class = int(selected_model.predict(student_features[FEATURE_COLUMNS].astype(float))[0])

average_mark = float(student_df["Final_Marks"].mean())
average_attendance = float(student_df["Attendance"].mean())
high_risk_subjects = int(student_df["needs_support_target"].sum())

# ============================================================
# HEADER / HERO
# ============================================================
hero_left, hero_right = st.columns([4.3, 1.7], gap="large")
with hero_left:
    st.markdown("# Student Performance Dashboard")
    st.markdown("**BITS Pilani · AI-Based Student Performance Advisor**")
    st.caption("Explore academic performance, attendance, subject trends, support indicators, and personalized study guidance.")
with hero_right:
    with st.container(border=True):
        st.markdown("### Selected Student")
        st.markdown(f"**{student_name}**")
        st.caption(f"Student ID · {selected_student}")
        st.progress(min(max(risk_probability, 0.0), 1.0), text=f"Support probability · {risk_probability:.1%}")
        st.write("")

# ============================================================
# STUDENT PROFILE / STATUS
# ============================================================
profile_left, profile_right = st.columns([2.2, 1.2], gap="large")
with profile_left:
    with st.container(border=True):
        st.markdown("### 👤 Student Snapshot")
        a, b, c = st.columns(3)
        with a:
            st.markdown("**Student**")
            st.markdown(f"### {student_name}")
            st.caption(selected_student)
        with b:
            st.markdown("**Subjects Represented**")
            st.markdown(f"### {student_df['Subject'].nunique()}")
            st.caption("Subject-level records")
        with c:
            st.markdown("**Average Mark**")
            st.markdown(f"### {average_mark:.1f}/100")
            st.caption("Final / comprehensive marks")
with profile_right:
    with st.container(border=True):
        st.markdown("### Academic Status")
        if risk_class:
            st.error(f"⚠️ **Learning Support Needed**\n\nModel probability: **{risk_probability:.1%}**")
        else:
            st.success(f"✅ **Good Academic Standing**\n\nModel probability: **{risk_probability:.1%}**")
        st.write("")

# ============================================================
# TABS
# ============================================================
tab_overview, tab_subjects, tab_ml, tab_recs, tab_qa = st.tabs([
    "📊 Overview", "📚 Subject Performance", "🤖 ML Diagnostics", "📝 Study Recommendations", "💬 Academic Q&A"
])

# ============================================================
# TAB 1
# ============================================================
with tab_overview:
    st.markdown("## Academic Overview")
    k1, k2 = st.columns(2, gap="medium")
    with k1:
        st.metric("Average Student Mark", f"{average_mark:.1f}/100")
    with k2:
        st.metric("Average Attendance", f"{average_attendance:.1f}%")
    k3, k4 = st.columns(2, gap="medium")
    with k3:
        st.metric("Academic Risk Probability", f"{risk_probability:.1%}")
    with k4:
        st.metric("High-Risk Subjects", str(high_risk_subjects))
    st.write("")

    if risk_class:
        st.error(f"⚠️ **Immediate Academic Attention Recommended** — the selected {model_choice} estimates a {risk_probability:.1%} probability of academic support need.")
    else:
        st.success(f"✅ **Good Academic Standing** — the selected {model_choice} estimates a {risk_probability:.1%} probability of academic support need.")

    chart_left, chart_right = st.columns([1.5, 1], gap="large")
    with chart_left:
        with st.container(border=True):
            st.markdown("### 📈 Subject Marks")
            subject_marks = student_df[["Subject", "Final_Marks"]].groupby("Subject", as_index=False).mean().sort_values("Final_Marks", ascending=True)
            fig = px.bar(subject_marks, x="Final_Marks", y="Subject", orientation="h", text="Final_Marks", labels={"Final_Marks": "Mark / 100", "Subject": ""})
            fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
            style_plot(fig, "Average Final / Comprehensive Mark")
            st.plotly_chart(fig, use_container_width=True)

    with chart_right:
        with st.container(border=True):
            st.markdown("### 🟢 Attendance Pattern")
            attendance = pd.DataFrame({
                "Status": ["Regular ≥75%", "Low <75%"],
                "Subjects": [int((student_df["Attendance"] >= 75).sum()), int((student_df["Attendance"] < 75).sum())],
            })
            fig = px.pie(attendance, names="Status", values="Subjects", hole=.55)
            fig.update_traces(textinfo="label+percent", textfont={"color": "#000000"})
            style_plot(fig, "Attendance Class Ratio")
            st.plotly_chart(fig, use_container_width=True)

    with st.container(border=True):
        st.markdown("### 📋 Academic Record")
        cols = ["Subject", "Attendance", "Assignment", "Quiz_Score", "Midterm_Marks", "Final_Marks", "performance_trend", "Grade"]
        display = student_df[cols].rename(columns={
            "Attendance": "Attendance %", "Assignment": "Assignment %", "Quiz_Score": "Quiz %",
            "Midterm_Marks": "Midsem", "Final_Marks": "Comprehensive", "performance_trend": "Trend"
        })
        st.dataframe(display.round(1), use_container_width=True, hide_index=True)

# ============================================================
# TAB 2
# ============================================================
with tab_subjects:
    st.markdown("## Subject-Wise Performance & Trends")
    marks = student_df[["Subject", "Midterm_Marks", "recent_avg", "previous_avg"]].melt(id_vars="Subject", var_name="Measure", value_name="Marks")
    marks["Measure"] = marks["Measure"].replace({"Midterm_Marks": "Midsem", "recent_avg": "Recent Average", "previous_avg": "Previous Average"})
    
    with st.container(border=True):
        fig = px.bar(marks, x="Subject", y="Marks", color="Measure", barmode="group", text="Marks", labels={"Marks": "Marks / 100", "Subject": ""})
        fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        style_plot(fig, "Midsem vs Previous vs Recent Average")
        st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns([1.2, 1], gap="large")
    with left:
        with st.container(border=True):
            st.markdown("### 🎯 Ranking & Priority Matrix")
            priority = student_df[["Subject", "recent_avg", "Attendance", "performance_trend", "missed_assessments"]].copy()
            priority["Priority Score"] = (
                (100 - priority["recent_avg"])
                + np.maximum(75 - priority["Attendance"], 0)
                + (priority["missed_assessments"] * 5)
                + np.where(priority["performance_trend"] == "Declining", 10, 0)
            )
            priority = priority.sort_values("Priority Score", ascending=False).reset_index(drop=True)
            priority.insert(0, "Priority", np.arange(1, len(priority) + 1))
            priority_display = priority[["Priority", "Subject", "recent_avg", "Attendance", "performance_trend", "missed_assessments"]].rename(columns={
                "recent_avg": "Recent Avg %", "Attendance": "Attendance %", "performance_trend": "Trend", "missed_assessments": "Missed"
            })
            st.dataframe(priority_display.round(1), use_container_width=True, hide_index=True)

    with right:
        with st.container(border=True):
            st.markdown("### 📉 Attendance vs Performance")
            fig = px.scatter(student_df, x="Attendance", y="recent_avg", size="missed_assessments", color="performance_trend", text="Subject", labels={"Attendance": "Attendance %", "recent_avg": "Recent Average %", "performance_trend": "Trend"})
            fig.add_hline(y=50, line_dash="dash", annotation_text="50 mark reference", annotation_font={"color": "#000000"})
            fig.add_vline(x=75, line_dash="dash", annotation_text="75% attendance reference", annotation_font={"color": "#000000"})
            style_plot(fig, "Attendance and Recent Performance")
            st.plotly_chart(fig, use_container_width=True)

# ============================================================
# TAB 3
# ============================================================
with tab_ml:
    st.markdown("## ML Model Diagnostics & Insights")
    comparison = pd.DataFrame({
        "Model": ["Logistic Regression", "Random Forest"],
        "Accuracy": [models["logistic_metrics"]["Accuracy"], models["forest_metrics"]["Accuracy"]],
        "Precision": [models["logistic_metrics"]["Precision"], models["forest_metrics"]["Precision"]],
        "Recall": [models["logistic_metrics"]["Recall"], models["forest_metrics"]["Recall"]],
        "F1-Score": [models["logistic_metrics"]["F1-Score"], models["forest_metrics"]["F1-Score"]],
    })
    with st.container(border=True):
        st.markdown("### Model Comparison")
        st.dataframe(comparison.style.format({"Accuracy": "{:.1%}", "Precision": "{:.1%}", "Recall": "{:.1%}", "F1-Score": "{:.1%}"}), use_container_width=True, hide_index=True)

    left, right = st.columns([1.25, 1], gap="large")
    with left:
        with st.container(border=True):
            st.markdown("### Feature Importance — Random Forest")
            importance = pd.DataFrame({"Feature": FEATURE_COLUMNS, "Importance": models["forest"].feature_importances_}).sort_values("Importance", ascending=True)
            importance["Feature"] = importance["Feature"].replace({
                "Attendance": "Attendance", "Assignment": "Assignment Average", "Quiz_Score": "Quiz Average",
                "Midterm_Marks": "Midsem Marks", "Final_Marks": "Comprehensive Marks", "missed_assessments": "Missed Assessments"
            })
            fig = px.bar(importance, x="Importance", y="Feature", orientation="h", text="Importance", labels={"Importance": "Importance", "Feature": ""})
            fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
            style_plot(fig, "Indicators Used by the Random Forest")
            st.plotly_chart(fig, use_container_width=True)

    with right:
        with st.container(border=True):
            st.markdown("### Selected Student Model Output")
            st.metric("Predicted Status", ("Learning Support Needed" if risk_class else "No Immediate Risk"))
            st.metric("Support Need Probability", f"{risk_probability:.1%}")
            st.metric("Selected Model", model_choice)
            matrix = models["forest_metrics"]["Confusion Matrix"] if model_choice.startswith("Random") else models["logistic_metrics"]["Confusion Matrix"]
            st.markdown("### Confusion Matrix")
            cm_df = pd.DataFrame(matrix, index=["Actual: No Risk", "Actual: Risk"], columns=["Predicted: No Risk", "Predicted: Risk"])
            st.dataframe(cm_df, use_container_width=True)

# ============================================================
# TAB 4
# ============================================================
with tab_recs:
    st.markdown("## Personalized Study Recommendations")
    st.info("Recommendations are generated from the selected student's actual subject-level records.")
    recommendation_count = 0
    for _, row in student_df.sort_values("recent_avg", ascending=True).iterrows():
        subject = str(row["Subject"])
        recent = float(row["recent_avg"])
        attendance = float(row["Attendance"])
        missed = int(row["missed_assessments"])
        trend = str(row["performance_trend"])

        if recent < 50 and trend == "Declining":
            recommendation_count += 1
            st.error(f"**Priority Action:** {subject} requires immediate attention due to declining scores ({recent:.1f}%).")
        if attendance < 75:
            recommendation_count += 1
            st.warning(f"**Attendance Warning:** Attendance in {subject} is below threshold ({attendance:.1f}%). Attendance improvement recommended.")
        if missed > 0:
            recommendation_count += 1
            st.warning(f"**Assessment Alert:** {missed} missed assessment(s) detected in {subject}. Coordinate with instructor.")
        if recent >= 75 and trend == "Improving":
            recommendation_count += 1
            st.success(f"**Positive Progress:** Excellent performance in {subject}. Maintain current study strategy.")

    if recommendation_count == 0:
        st.success("No specific intervention rule was triggered by the selected student's current records.")

# ============================================================
# TAB 5
# ============================================================
with tab_qa:
    st.markdown("## Academic Q&A")
    st.caption("Ask about the selected student's available academic records.")
    question = st.text_input("Ask a question", placeholder="Which subject should I focus on?", key="academic_question")
    if question:
        q = question.lower().strip()
        if "which subject" in q and ("focus" in q or "improve" in q or "study" in q):
            declining = student_df[student_df["performance_trend"] == "Declining"]
            if not declining.empty:
                target = declining.sort_values("recent_avg", ascending=True).iloc[0]
                st.info(f"**Recommended focus:** {target['Subject']}. Lowest recent average among declining subjects at {target['recent_avg']:.1f}%.")
            else:
                target = student_df.sort_values("recent_avg", ascending=True).iloc[0]
                st.info(f"**Recommended focus:** {target['Subject']}, lowest recent average at {target['recent_avg']:.1f}%.")
        elif "attendance" in q:
            st.info(f"The selected student's average attendance is {average_attendance:.1f}% across available subjects.")
        elif "risk" in q:
            st.info(f"The selected {model_choice} estimates a {risk_probability:.1%} probability of academic support need.")
        else:
            st.info(f"For {selected_student}, average final mark is {average_mark:.1f}/100, attendance is {average_attendance:.1f}%, and {high_risk_subjects} subject(s) meet the support-target rule.")

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.caption("BITS Pilani · AI-Based Student Performance Advisor · All displayed values derived from dataset.")
