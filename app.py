import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

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
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="BITS Pilani | AI Student Performance Advisor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# LIGHT ACADEMIC THEME
# ============================================================
st.markdown(
    """
    <style>
    /* ---------- Global page ---------- */
    .stApp {
        background: #f8fafc;
        color: #000000;
    }

    .main .block-container {
        max-width: 1500px;
        padding-top: 1.2rem;
        padding-bottom: 2.5rem;
    }

    /* ---------- All ordinary text ---------- */
    html, body, [class*="css"], .stApp, .stMarkdown,
    p, span, label, div, li, td, th {
        color: #000000;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #000000 !important;
    }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #cbd5e1;
    }

    section[data-testid="stSidebar"] * {
        color: #000000 !important;
    }

    /* ---------- Metric cards ---------- */
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 14px;
        padding: 16px 18px;
        box-shadow: 0 3px 12px rgba(15, 23, 42, 0.07);
        min-height: 125px;
    }

    div[data-testid="stMetricLabel"] {
        color: #334155 !important;
        font-weight: 700;
    }

    div[data-testid="stMetricValue"] {
        color: #000000 !important;
        font-weight: 800;
    }

    div[data-testid="stMetricDelta"] {
        color: #000000 !important;
    }

    /* ---------- Native bordered containers ---------- */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 14px;
        box-shadow: 0 3px 12px rgba(15, 23, 42, 0.06);
    }

    /* ---------- Buttons ---------- */
    .stButton > button {
        border: 1px solid #94a3b8;
        border-radius: 9px;
        background: #ffffff;
        color: #000000 !important;
        font-weight: 700;
    }

    /* ---------- Inputs ---------- */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        background: #ffffff;
        color: #000000;
        border-color: #94a3b8;
    }

    input {
        color: #000000 !important;
        background: #ffffff !important;
    }

    /* ---------- Dataframes ---------- */
    div[data-testid="stDataFrame"] {
        border: 1px solid #cbd5e1;
        border-radius: 10px;
        overflow: hidden;
    }

    /* ---------- Tabs ---------- */
    button[data-baseweb="tab"] {
        color: #000000 !important;
        font-weight: 700;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #000000 !important;
        border-bottom: 3px solid #000000;
    }

    /* ---------- Alert boxes ---------- */
    div[data-testid="stAlert"] * {
        color: #000000 !important;
    }

    /* ---------- File uploader ---------- */
    section[data-testid="stFileUploaderDropzone"] {
        background: #ffffff;
        border: 1px dashed #64748b;
    }

    section[data-testid="stFileUploaderDropzone"] * {
        color: #000000 !important;
    }

    /* ---------- Hide Streamlit decoration ---------- */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# ROBUST DATA PATH RESOLUTION
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "Subject_Analytics_Dataset_200_Students.xlsx")

# Fallback check if file is placed in root instead of data folder
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


@st.cache_data(show_spinner="Loading the BITS Pilani student dataset...")
def load_dataset(path: str) -> pd.DataFrame:
    """Load only the supplied Excel dataset and normalize column names."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at expected resolved path: '{path}'. "
            "Please ensure a 'data' folder exists with 'Subject_Analytics_Dataset_200_Students.xlsx' inside it."
        )

    try:
        df = pd.read_excel(path, engine="openpyxl")
    except Exception as exc:
        raise RuntimeError(f"Unable to read the Excel dataset using openpyxl: {exc}")

    if df.empty:
        raise ValueError("The Excel dataset is empty.")

    # Map common variants without inventing any records.
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
        "Assignment": "Assignment",
        "assignment_avg": "Assignment",
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
            "The supplied dataset is missing required columns: "
            + ", ".join(missing)
            + ".\n\nAvailable columns: "
            + ", ".join(map(str, df.columns))
        )

    # Keep actual records only.
    df = df.copy()
    df["Student_ID"] = df["Student_ID"].astype(str).str.strip()
    df["Subject"] = df["Subject"].astype(str).str.strip()

    numeric_columns = [
        "Attendance",
        "Assignment",
        "Quiz_Score",
        "Midterm_Marks",
        "Final_Marks",
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Student_ID", "Subject"] + numeric_columns)

    # Aggregate subject rows only when duplicate records for the same
    # student + subject exist. This preserves the subject-level design.
    df = (
        df.groupby(["Student_ID", "Subject"], as_index=False)[numeric_columns]
        .mean()
    )

    # Required feature engineering from the supplied records.
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

    if "missed_assessments" not in df.columns:
        df["missed_assessments"] = np.where(df["Attendance"] < 75, 2, 0)

    df["missed_assessments"] = (
        pd.to_numeric(df["missed_assessments"], errors="coerce")
        .fillna(0)
        .clip(lower=0, upper=5)
        .astype(int)
    )

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
    """Train and evaluate Logistic Regression and Random Forest."""
    X = df[FEATURE_COLUMNS].astype(float)
    y = df["needs_support_target"].astype(int)

    if y.nunique() < 2:
        raise ValueError(
            "The dataset contains only one support class. "
            "At least two classes are required for model training."
        )

    class_counts = y.value_counts()
    stratify = y if class_counts.min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=stratify,
    )

    logistic = LogisticRegression(random_state=42, max_iter=1000)
    forest = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
    )

    logistic.fit(X_train, y_train)
    forest.fit(X_train, y_train)

    def evaluate(model):
        pred = model.predict(X_test)
        return {
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(
                y_test, pred, zero_division=0
            ),
            "Recall": recall_score(
                y_test, pred, zero_division=0
            ),
            "F1-Score": f1_score(
                y_test, pred, zero_division=0
            ),
            "Confusion Matrix": confusion_matrix(y_test, pred),
        }

    return {
        "logistic": logistic,
        "forest": forest,
        "logistic_metrics": evaluate(logistic),
        "forest_metrics": evaluate(forest),
        "test_size": len(y_test),
    }


# ============================================================
# PLOT HELPERS
# ============================================================
def apply_plot_theme(fig, title=None):
    fig.update_layout(
        title=dict(
            text=title or "",
            font=dict(color="#000000", size=18),
            x=0,
        ),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#000000", size=13),
        margin=dict(l=50, r=30, t=55, b=50),
        legend=dict(
            font=dict(color="#000000"),
            bgcolor="#ffffff",
            bordercolor="#cbd5e1",
            borderwidth=1,
        ),
    )

    fig.update_xaxes(
        color="#000000",
        tickfont=dict(color="#000000"),
        title_font=dict(color="#000000"),
        gridcolor="#cbd5e1",
        zerolinecolor="#64748b",
    )
    fig.update_yaxes(
        color="#000000",
        tickfont=dict(color="#000000"),
        title_font=dict(color="#000000"),
        gridcolor="#cbd5e1",
        zerolinecolor="#64748b",
    )
    return fig


def safe_probability(model, row):
    values = row[FEATURE_COLUMNS].astype(float).to_numpy().reshape(1, -1)
    probabilities = model.predict_proba(values)[0]
    classes = list(model.classes_)
    return float(
        probabilities[classes.index(1)]
        if 1 in classes
        else 0.0
    )


# ============================================================
# LOAD DATA & MODELS (WITH DETAILED ERROR DISPLAY)
# ============================================================
try:
    data = load_dataset(DATA_PATH)
except Exception as exc:
    st.error("⚠️ Unable to load the supplied dataset.")
    st.markdown(f"**Resolved Path Checked:** `{DATA_PATH}`")
    st.exception(exc)
    st.stop()

if data.empty:
    st.error("No usable records were found in the supplied Excel dataset.")
    st.stop()

try:
    models = train_models(data)
except Exception as exc:
    st.error("The academic support model could not be trained.")
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

    st.markdown("### Dataset")
    st.success("Using the supplied Excel dataset")
    st.caption(DATA_PATH)

    students = (
        data[["Student_ID"]]
        .drop_duplicates()
        .sort_values("Student_ID")["Student_ID"]
        .tolist()
    )

    selected_student = st.selectbox(
        "Select Student",
        students,
        index=0,
    )

    model_choice = st.radio(
        "Academic Support Model",
        ["Random Forest (Main)", "Logistic Regression (Baseline)"],
    )

    selected_model = (
        models["forest"]
        if model_choice.startswith("Random")
        else models["logistic"]
    )

    st.divider()
    st.markdown("### Dataset Summary")
    st.write(f"**Unique students:** {data['Student_ID'].nunique():,}")
    st.write(f"**Subjects:** {data['Subject'].nunique():,}")
    st.write(f"**Academic records:** {len(data):,}")

# ============================================================
# SELECTED STUDENT DATA
# ============================================================
student_df = data[data["Student_ID"] == selected_student].copy()

if student_df.empty:
    st.error("No academic records are available for the selected student.")
    st.stop()

student_name = (
    student_df["Student_Name"].iloc[0]
    if "Student_Name" in student_df.columns
    else selected_student
)

student_row = (
    student_df[FEATURE_COLUMNS]
    .mean()
    .to_frame()
    .T
)

student_row["needs_support_target"] = int(
    student_df["needs_support_target"].mean() >= 0.5
)

risk_probability = safe_probability(selected_model, student_row.iloc[0])
risk_class = int(selected_model.predict(
    student_row[FEATURE_COLUMNS].astype(float)
)[0])

average_mark = float(student_df["Final_Marks"].mean())
average_attendance = float(student_df["Attendance"].mean())
high_risk_subjects = int(student_df["needs_support_target"].sum())

status_is_risk = risk_class == 1

# ============================================================
# HEADER
# ============================================================
header_left, header_right = st.columns([4, 1])

with header_left:
    st.title("Student Performance Dashboard")
    st.markdown(
        "**BITS Pilani · AI-Based Student Performance Advisor**"
    )
    st.caption(
        "Academic performance, attendance, learning trends, "
        "support indicators, and personalized study guidance."
    )

with header_right:
    st.metric(
        "Selected Student",
        selected_student,
        student_name if student_name != selected_student else None,
    )

st.divider()

# ============================================================
# PROFILE STRIP
# ============================================================
p1, p2, p3, p4 = st.columns([1.6, 1.1, 1.1, 1.4])

with p1:
    with st.container(border=True):
        st.markdown("### 👤 Student")
        st.markdown(f"**{student_name}**")
        st.caption(f"Student ID: {selected_student}")

with p2:
    with st.container(border=True):
        st.markdown("### 🏫 Subjects")
        st.markdown(f"**{student_df['Subject'].nunique()}**")
        st.caption("Subjects represented")

with p3:
    with st.container(border=True):
        st.markdown("### 📚 Average Mark")
        st.markdown(f"**{average_mark:.1f}/100**")
        st.caption("Final / comprehensive marks")

with p4:
    with st.container(border=True):
        st.markdown("### Academic Status")
        if status_is_risk:
            st.error(
                f"⚠️ Learning Support Needed\n\n"
                f"ML probability: **{risk_probability:.1%}**"
            )
        else:
            st.success(
                f"✅ Good Academic Standing\n\n"
                f"ML probability: **{risk_probability:.1%}**"
            )

# ============================================================
# TABS
# ============================================================
tabs = st.tabs(
    [
        "📊 Overview",
        "📚 Subject Performance",
        "🤖 ML Diagnostics",
        "📝 Study Recommendations",
        "💬 Academic Q&A",
    ]
)

# ============================================================
# TAB 1 — OVERVIEW
# ============================================================
with tabs[0]:
    st.subheader("Academic Overview")

    k1, k2, k3, k4 = st.columns(4)

    with k1:
        st.metric("Average Student Mark", f"{average_mark:.1f}/100")

    with k2:
        st.metric("Average Attendance", f"{average_attendance:.1f}%")

    with k3:
        st.metric("Academic Risk Probability", f"{risk_probability:.1%}")

    with k4:
        st.metric("High-Risk Subjects", str(high_risk_subjects))

    st.write("")

    if status_is_risk:
        st.error(
            f"⚠️ **Immediate Academic Attention Recommended** — "
            f"the selected model estimates a {risk_probability:.1%} "
            f"probability of academic support need for {selected_student}."
        )
    else:
        st.success(
            f"✅ **Good Academic Standing** — "
            f"the selected model estimates a {risk_probability:.1%} "
            f"probability of academic support need."
        )

    left, right = st.columns([1.45, 1])

    with left:
        with st.container(border=True):
            st.markdown("### Average Marks by Subject")
            subject_plot = (
                student_df.groupby("Subject", as_index=False)["Final_Marks"]
                .mean()
                .sort_values("Final_Marks", ascending=True)
            )

            fig = px.bar(
                subject_plot,
                x="Final_Marks",
                y="Subject",
                orientation="h",
                text="Final_Marks",
                labels={
                    "Final_Marks": "Final / Comprehensive Mark",
                    "Subject": "Subject",
                },
            )
            fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
            apply_plot_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

    with right:
        with st.container(border=True):
            st.markdown("### Attendance Status")
            attendance_status = pd.DataFrame(
                {
                    "Status": [
                        "Regular Attendance (≥75%)",
                        "Low Attendance (<75%)",
                    ],
                    "Subjects": [
                        int((student_df["Attendance"] >= 75).sum()),
                        int((student_df["Attendance"] < 75).sum()),
                    ],
                }
            )

            fig = px.pie(
                attendance_status,
                names="Status",
                values="Subjects",
                hole=0.45,
            )
            fig.update_traces(
                textinfo="label+percent",
                textfont=dict(color="#000000"),
            )
            apply_plot_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Academic Record Summary")
    summary_cols = [
        "Subject",
        "Attendance",
        "Assignment",
        "Quiz_Score",
        "Midterm_Marks",
        "Final_Marks",
        "performance_trend",
        "Grade",
    ]
    st.dataframe(
        student_df[summary_cols]
        .rename(
            columns={
                "Attendance": "Attendance %",
                "Assignment": "Assignment %",
                "Quiz_Score": "Quiz %",
                "Midterm_Marks": "Midsem",
                "Final_Marks": "Comprehensive",
                "performance_trend": "Trend",
            }
        )
        .round(1),
        use_container_width=True,
        hide_index=True,
    )

# ============================================================
# TAB 2 — SUBJECT PERFORMANCE
# ============================================================
with tabs[1]:
    st.subheader("Subject-Wise Performance & Trends")

    marks_long = student_df[
        ["Subject", "Midterm_Marks", "recent_avg", "previous_avg"]
    ].melt(
        id_vars="Subject",
        var_name="Measure",
        value_name="Marks",
    )

    marks_long["Measure"] = marks_long["Measure"].replace(
        {
            "Midterm_Marks": "Midsem",
            "recent_avg": "Recent Average",
            "previous_avg": "Previous Average",
        }
    )

    fig = px.bar(
        marks_long,
        x="Subject",
        y="Marks",
        color="Measure",
        barmode="group",
        text="Marks",
        labels={"Marks": "Marks / 100"},
    )
    fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    apply_plot_theme(fig, "Subject Marks Breakdown")
    st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns([1.15, 1])

    with left:
        st.markdown("### Ranking & Priority Matrix")

        priority = student_df[
            [
                "Subject",
                "recent_avg",
                "Attendance",
                "performance_trend",
                "missed_assessments",
                "needs_support_target",
            ]
        ].copy()

        priority["Priority Score"] = (
            (100 - priority["recent_avg"])
            + np.maximum(75 - priority["Attendance"], 0)
            + priority["missed_assessments"] * 5
            + np.where(priority["performance_trend"] == "Declining", 10, 0)
        )

        priority = priority.sort_values(
            "Priority Score", ascending=False
        ).reset_index(drop=True)
        priority.insert(0, "Priority", np.arange(1, len(priority) + 1))

        display_priority = priority[
            [
                "Priority",
                "Subject",
                "recent_avg",
                "Attendance",
                "performance_trend",
                "missed_assessments",
            ]
        ].rename(
            columns={
                "recent_avg": "Recent Avg %",
                "Attendance": "Attendance %",
                "performance_trend": "Trend",
                "missed_assessments": "Missed Assessments",
            }
        )

        st.dataframe(
            display_priority.round(1),
            use_container_width=True,
            hide_index=True,
        )

    with right:
        st.markdown("### Attendance vs Performance")

        fig = px.scatter(
            student_df,
            x="Attendance",
            y="recent_avg",
            size="missed_assessments",
            color="performance_trend",
            text="Subject",
            labels={
                "Attendance": "Attendance %",
                "recent_avg": "Recent Average %",
                "performance_trend": "Performance Trend",
            },
        )
        fig.add_hline(
            y=50,
            line_dash="dash",
            annotation_text="50 mark reference",
            annotation_font=dict(color="#000000"),
        )
        fig.add_vline(
            x=75,
            line_dash="dash",
            annotation_text="75% attendance reference",
            annotation_font=dict(color="#000000"),
        )
        apply_plot_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# TAB 3 — ML DIAGNOSTICS
# ============================================================
with tabs[2]:
    st.subheader("ML Model Diagnostics & Insights")

    comparison = pd.DataFrame(
        {
            "Model": [
                "Logistic Regression",
                "Random Forest",
            ],
            "Accuracy": [
                models["logistic_metrics"]["Accuracy"],
                models["forest_metrics"]["Accuracy"],
            ],
            "Precision": [
                models["logistic_metrics"]["Precision"],
                models["forest_metrics"]["Precision"],
            ],
            "Recall": [
                models["logistic_metrics"]["Recall"],
                models["forest_metrics"]["Recall"],
            ],
            "F1-Score": [
                models["logistic_metrics"]["F1-Score"],
                models["forest_metrics"]["F1-Score"],
            ],
        }
    )

    st.markdown("### Model Comparison")
    st.dataframe(
        comparison.style.format(
            {
                "Accuracy": "{:.1%}",
                "Precision": "{:.1%}",
                "Recall": "{:.1%}",
                "F1-Score": "{:.1%}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown("### Random Forest Feature Importance")

        importance = pd.DataFrame(
            {
                "Feature": FEATURE_COLUMNS,
                "Importance": models["forest"].feature_importances_,
            }
        ).sort_values("Importance", ascending=True)

        importance["Feature"] = importance["Feature"].replace(
            {
                "Attendance": "Attendance",
                "Assignment": "Assignment Average",
                "Quiz_Score": "Quiz Average",
                "Midterm_Marks": "Midsem Marks",
                "Final_Marks": "Comprehensive Marks",
                "missed_assessments": "Missed Assessments",
            }
        )

        fig = px.bar(
            importance,
            x="Importance",
            y="Feature",
            orientation="h",
            text="Importance",
            labels={"Importance": "Feature Importance"},
        )
        fig.update_traces(
            texttemplate="%{text:.3f}",
            textposition="outside",
        )
        apply_plot_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Selected Student Model Output")

        st.metric(
            "Predicted Academic Risk",
            "Requires Academic Attention"
            if risk_class
            else "No Immediate Risk",
        )

        st.metric(
            "Support Need Probability",
            f"{risk_probability:.1%}",
        )

        st.metric(
            "Model",
            model_choice,
        )

        cm = (
            models["forest_metrics"]["Confusion Matrix"]
            if model_choice.startswith("Random")
            else models["logistic_metrics"]["Confusion Matrix"]
        )

        st.markdown("### Confusion Matrix")
        cm_df = pd.DataFrame(
            cm,
            index=["Actual: No Risk", "Actual: Risk"],
            columns=["Predicted: No Risk", "Predicted: Risk"],
        )
        st.dataframe(cm_df, use_container_width=True)

# ============================================================
# TAB 4 — RECOMMENDATIONS
# ============================================================
with tabs[3]:
    st.subheader("Personalized Study Recommendations")
    st.caption(
        "Every recommendation below is generated from the selected "
        "student's actual subject records."
    )

    recommendation_count = 0

    for _, row in student_df.sort_values(
        "recent_avg", ascending=True
    ).iterrows():

        subject = row["Subject"]
        recent = float(row["recent_avg"])
        attendance = float(row["Attendance"])
        missed = int(row["missed_assessments"])
        trend = row["performance_trend"]

        if recent < 50 and trend == "Declining":
            recommendation_count += 1
            st.error(
                f"**Priority Action:** {subject} requires immediate "
                f"attention due to declining scores ({recent:.1f}%)."
            )

        if attendance < 75:
            recommendation_count += 1
            st.warning(
                f"**Attendance Warning:** Attendance in {subject} is "
                f"below threshold ({attendance:.1f}%). "
                f"Attendance improvement recommended."
            )

        if missed > 0:
            recommendation_count += 1
            st.warning(
                f"**Assessment Alert:** {missed} missed assessment(s) "
                f"detected in {subject}. Coordinate with the instructor "
                f"for retakes or supplementary submission."
            )

        if recent >= 75 and trend == "Improving":
            recommendation_count += 1
            st.success(
                f"**Positive Progress:** Excellent performance in "
                f"{subject}. Maintain the current study strategy."
            )

    if recommendation_count == 0:
        st.success(
            "No specific intervention rule was triggered by the "
            "selected student's current academic records. "
            "Continue monitoring subject performance, attendance, "
            "and assessment completion."
        )

# ============================================================
# TAB 5 — ACADEMIC Q&A
# ============================================================
with tabs[4]:
    st.subheader("Academic Q&A")
    st.caption(
        "Ask questions about the selected student's available "
        "academic records."
    )

    question = st.text_input(
        "Ask an academic question",
        placeholder="Which subject should I focus on?",
    )

    if question:
        q = question.lower().strip()

        if (
            "which subject" in q
            and ("focus" in q or "improve" in q or "study" in q)
        ):
            declining = student_df[
                student_df["performance_trend"] == "Declining"
            ]

            if not declining.empty:
                target = declining.sort_values(
                    "recent_avg", ascending=True
                ).iloc[0]

                st.info(
                    f"**Recommended focus:** {target['Subject']}. "
                    f"It has the lowest recent average among the "
                    f"declining subjects at {target['recent_avg']:.1f}%."
                )
            else:
                target = student_df.sort_values(
                    "recent_avg", ascending=True
                ).iloc[0]

                st.info(
                    f"**Recommended focus:** {target['Subject']}, "
                    f"which has the lowest recent average at "
                    f"{target['recent_avg']:.1f}%."
                )

        elif (
            "why" in q
            and ("decreased" in q or "declined" in q or "performance" in q)
        ):
            declining = student_df[
                student_df["performance_trend"] == "Declining"
            ]

            if declining.empty:
                st.info(
                    "The selected student's subject records do not "
                    "show a declining performance trend."
                )
            else:
                st.markdown("**Observed factors in declining subjects:**")

                for _, row in declining.iterrows():
                    reasons = []

                    if row["Attendance"] < 75:
                        reasons.append(
                            f"attendance is {row['Attendance']:.1f}%"
                        )

                    if row["missed_assessments"] > 0:
                        reasons.append(
                            f"{int(row['missed_assessments'])} "
                            "missed assessment(s)"
                        )

                    drop = row["previous_avg"] - row["recent_avg"]
                    if drop > 0:
                        reasons.append(
                            f"recent average is {drop:.1f} points "
                            "below the previous average"
                        )

                    if reasons:
                        st.write(
                            f"**{row['Subject']}:** "
                            + "; ".join(reasons)
                            + "."
                        )
                    else:
                        st.write(
                            f"**{row['Subject']}:** declining trend is "
                            "visible in the available marks, but no "
                            "additional attendance or missed-assessment "
                            "factor is present in the record."
                        )

        elif "attendance" in q:
            st.info(
                f"The selected student's average attendance is "
                f"{average_attendance:.1f}% across the available subjects."
            )

        elif "risk" in q:
            st.info(
                f"The selected {model_choice} estimates a "
                f"{risk_probability:.1%} probability of academic "
                f"support need."
            )

        else:
            st.info(
                f"For {selected_student}, the current average final mark "
                f"is {average_mark:.1f}/100, average attendance is "
                f"{average_attendance:.1f}%, and {high_risk_subjects} "
                f"subject(s) currently meet the support-target rule."
            )

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.caption(
    "BITS Pilani · AI-Based Student Performance Advisor · "
    "All displayed academic values are calculated from the supplied "
    "Subject_Analytics_Dataset_200_Students.xlsx dataset."
)
