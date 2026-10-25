# ============================================================
# BITS PILANI
# AI-BASED STUDENT PERFORMANCE ADVISOR
#
# app.py
#
# DATA SOURCE:
# data/Subject_Analytics_Dataset_200_Students.xlsx
#
# IMPORTANT:
# No synthetic data is generated.
# The uploaded/reference dashboard image is used only for
# visual/layout inspiration.
# ============================================================

import os
import warnings

import numpy as np
import pandas as pd
import streamlit as st
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

warnings.filterwarnings("ignore")


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="BITS Pilani | Student Performance Advisor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# APPLICATION CONSTANTS
# ============================================================

DATA_PATH = "data/Subject_Analytics_Dataset_200_Students.xlsx"

FEATURE_COLUMNS = [
    "attendance",
    "assignment_avg",
    "quiz_avg",
    "midsem_marks",
    "previous_avg",
    "recent_avg",
    "missed_assessments",
]

TARGET_COLUMN = "risk"

RISK_LABELS = {
    0: "No Immediate Risk",
    1: "Requires Academic Attention",
}


# ============================================================
# HIGH-CONTRAST ACADEMIC DESIGN
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       GLOBAL
       ======================================================== */

    .stApp {
        background: #f4f7fb;
    }

    .main {
        background: #f4f7fb;
    }

    [data-testid="stAppViewContainer"] {
        background: #f4f7fb;
    }

    * {
        font-family: "Inter", "Segoe UI", Arial, sans-serif;
    }

    h1, h2, h3, h4, h5, h6,
    p, label, span, div {
        color: #000000;
    }

    /* ========================================================
       SIDEBAR
       ======================================================== */

    section[data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #cbd5e1;
    }

    section[data-testid="stSidebar"] * {
        color: #000000 !important;
    }

    .sidebar-brand {
        padding: 8px 4px 20px 4px;
        border-bottom: 1px solid #e2e8f0;
        margin-bottom: 20px;
    }

    .sidebar-brand-title {
        font-size: 20px;
        font-weight: 900;
        color: #000000;
    }

    .sidebar-brand-subtitle {
        font-size: 12px;
        color: #475569;
        margin-top: 4px;
    }

    /* ========================================================
       HEADER
       ======================================================== */

    .dashboard-header {
        background: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 14px;
        padding: 20px 24px;
        margin-bottom: 16px;
        box-shadow: 0 3px 12px rgba(15, 23, 42, 0.07);
    }

    .dashboard-title {
        font-size: 28px;
        font-weight: 900;
        color: #000000 !important;
        margin: 0;
    }

    .dashboard-subtitle {
        font-size: 13px;
        color: #475569 !important;
        margin-top: 5px;
    }

    .data-status {
        text-align: right;
        font-size: 12px;
        color: #475569 !important;
        padding-top: 5px;
    }

    /* ========================================================
       KPI CARDS
       ======================================================== */

    .kpi-card {
        background: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 12px;
        min-height: 120px;
        padding: 17px;
        box-shadow: 0 3px 12px rgba(15, 23, 42, 0.07);
    }

    .kpi-title {
        font-size: 11px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #475569 !important;
    }

    .kpi-value {
        font-size: 26px;
        font-weight: 900;
        color: #000000 !important;
        margin-top: 8px;
    }

    .kpi-description {
        font-size: 11px;
        color: #64748b !important;
        margin-top: 4px;
    }

    /* ========================================================
       SECTION / CARD
       ======================================================== */

    .section-card {
        background: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 12px;
        padding: 17px;
        box-shadow: 0 3px 12px rgba(15, 23, 42, 0.06);
        margin-bottom: 15px;
    }

    .section-title {
        font-size: 15px;
        font-weight: 900;
        color: #000000 !important;
        margin-bottom: 4px;
    }

    .section-description {
        font-size: 11px;
        color: #64748b !important;
        margin-bottom: 10px;
    }

    /* ========================================================
       STATUS
       ======================================================== */

    .status-good {
        background: #dcfce7;
        border: 1px solid #86efac;
        border-radius: 9px;
        padding: 12px 16px;
        font-weight: 800;
        color: #14532d !important;
        margin-bottom: 15px;
    }

    .status-warning {
        background: #fef3c7;
        border: 1px solid #fbbf24;
        border-radius: 9px;
        padding: 12px 16px;
        font-weight: 800;
        color: #78350f !important;
        margin-bottom: 15px;
    }

    .status-danger {
        background: #fee2e2;
        border: 1px solid #fca5a5;
        border-radius: 9px;
        padding: 12px 16px;
        font-weight: 800;
        color: #7f1d1d !important;
        margin-bottom: 15px;
    }

    /* ========================================================
       PROFILE
       ======================================================== */

    .profile-card {
        background: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 3px 12px rgba(15, 23, 42, 0.06);
    }

    .profile-name {
        font-size: 21px;
        font-weight: 900;
    }

    .profile-id {
        font-size: 12px;
        color: #64748b !important;
        margin-top: 3px;
    }

    .profile-stat {
        border-left: 3px solid #0f172a;
        padding-left: 10px;
        margin-top: 12px;
    }

    .profile-stat-label {
        font-size: 10px;
        color: #64748b !important;
        text-transform: uppercase;
        font-weight: 700;
    }

    .profile-stat-value {
        font-size: 17px;
        font-weight: 900;
    }

    /* ========================================================
       RECOMMENDATION
       ======================================================== */

    .recommendation {
        background: #ffffff;
        border: 1px solid #cbd5e1;
        border-left: 4px solid #0f172a;
        border-radius: 9px;
        padding: 14px;
        margin-bottom: 9px;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.05);
    }

    .recommendation-heading {
        font-size: 12px;
        font-weight: 900;
        color: #000000 !important;
    }

    .recommendation-text {
        font-size: 12px;
        color: #334155 !important;
        margin-top: 5px;
        line-height: 1.5;
    }

    /* ========================================================
       CHAT
       ======================================================== */

    .answer-card {
        background: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 10px;
        padding: 15px;
        margin-top: 8px;
        color: #000000 !important;
    }

    /* ========================================================
       STREAMLIT ELEMENTS
       ======================================================== */

    div[data-baseweb="select"] {
        background: #ffffff !important;
    }

    div[data-baseweb="select"] * {
        color: #000000 !important;
    }

    .stSelectbox label,
    .stMultiSelect label,
    .stRadio label,
    .stTextInput label {
        color: #000000 !important;
        font-weight: 700 !important;
    }

    button {
        color: #000000 !important;
    }

    [data-testid="stMetricValue"] {
        color: #000000 !important;
    }

    [data-testid="stMetricLabel"] {
        color: #475569 !important;
    }

    /* ========================================================
       TABLE
       ======================================================== */

    [data-testid="stDataFrame"] {
        border: 1px solid #cbd5e1;
        border-radius: 8px;
        overflow: hidden;
    }

    /* ========================================================
       FOOTER
       ======================================================== */

    .footer {
        text-align: center;
        padding: 20px 0 5px 0;
        color: #64748b !important;
        font-size: 11px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_dataset(path: str) -> pd.DataFrame:
    """Load the real Excel dataset only."""

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    df = pd.read_excel(
        path,
        engine="openpyxl",
    )

    if df.empty:
        raise ValueError(
            "The Excel dataset contains no records."
        )

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )

    return df


# ============================================================
# COLUMN MAPPING
# ============================================================

def detect_column(
    df,
    candidates,
    required=True,
):
    """Find a column using common academic naming variants."""

    for candidate in candidates:

        candidate = (
            candidate
            .lower()
            .strip()
            .replace(" ", "_")
            .replace("-", "_")
        )

        if candidate in df.columns:
            return candidate

    if required:
        raise ValueError(
            "Could not find a required column. "
            f"Accepted names: {candidates}"
        )

    return None


def prepare_dataset(raw: pd.DataFrame) -> pd.DataFrame:
    """Convert the real Excel data into the application schema."""

    df = raw.copy()

    # --------------------------------------------------------
    # Student ID
    # --------------------------------------------------------

    student_col = detect_column(
        df,
        [
            "student_id",
            "studentid",
            "student",
            "id",
        ],
    )

    df["student_id"] = (
        df[student_col]
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------------------
    # Student Name
    # --------------------------------------------------------

    name_col = detect_column(
        df,
        [
            "student_name",
            "studentname",
            "name",
            "student_full_name",
        ],
        required=False,
    )

    if name_col:
        df["student_name"] = (
            df[name_col]
            .astype(str)
            .str.strip()
        )
    else:
        df["student_name"] = df["student_id"]

    # --------------------------------------------------------
    # Department
    # --------------------------------------------------------

    department_col = detect_column(
        df,
        [
            "department",
            "dept",
            "branch",
            "program",
        ],
        required=False,
    )

    if department_col:
        df["department"] = (
            df[department_col]
            .astype(str)
            .str.strip()
        )
    else:
        df["department"] = "Not Available"

    # --------------------------------------------------------
    # Semester
    # --------------------------------------------------------

    semester_col = detect_column(
        df,
        [
            "semester",
            "sem",
            "term",
        ],
        required=False,
    )

    if semester_col:
        df["semester"] = (
            df[semester_col]
            .astype(str)
            .str.strip()
        )
    else:
        df["semester"] = "Not Available"

    # --------------------------------------------------------
    # Subject
    # --------------------------------------------------------

    subject_col = detect_column(
        df,
        [
            "subject",
            "course",
            "module",
            "subject_name",
            "course_name",
        ],
    )

    df["subject"] = (
        df[subject_col]
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------------------
    # Academic fields
    # --------------------------------------------------------

    attendance_col = detect_column(
        df,
        [
            "attendance",
            "attendance_percentage",
            "attendance_percent",
            "attendance_%",
        ],
    )

    assignment_col = detect_column(
        df,
        [
            "assignment_avg",
            "assignment_average",
            "assignment_marks",
            "assignment_score",
            "assignment",
        ],
    )

    quiz_col = detect_column(
        df,
        [
            "quiz_avg",
            "quiz_average",
            "quiz_marks",
            "quiz_score",
            "quiz",
        ],
    )

    midsem_col = detect_column(
        df,
        [
            "midsem_marks",
            "midterm_marks",
            "midterm",
            "mid_sem_marks",
            "mid_sem",
        ],
    )

    final_col = detect_column(
        df,
        [
            "comprehensive_marks",
            "final_marks",
            "final_mark",
            "comprehensive",
            "final",
        ],
    )

    df["attendance"] = pd.to_numeric(
        df[attendance_col],
        errors="coerce",
    )

    df["assignment_avg"] = pd.to_numeric(
        df[assignment_col],
        errors="coerce",
    )

    df["quiz_avg"] = pd.to_numeric(
        df[quiz_col],
        errors="coerce",
    )

    df["midsem_marks"] = pd.to_numeric(
        df[midsem_col],
        errors="coerce",
    )

    df["comprehensive_marks"] = pd.to_numeric(
        df[final_col],
        errors="coerce",
    )

    # --------------------------------------------------------
    # Previous Average
    # --------------------------------------------------------

    previous_col = detect_column(
        df,
        [
            "previous_avg",
            "previous_average",
            "previous_marks",
            "previous_score",
        ],
        required=False,
    )

    if previous_col:

        df["previous_avg"] = pd.to_numeric(
            df[previous_col],
            errors="coerce",
        )

    else:

        # Use the available midsem score as the historical
        # academic comparison when the source dataset does
        # not explicitly provide previous_avg.
        df["previous_avg"] = df[
            "midsem_marks"
        ]

    # --------------------------------------------------------
    # Recent Average
    # --------------------------------------------------------

    recent_col = detect_column(
        df,
        [
            "recent_avg",
            "recent_average",
            "recent_marks",
            "recent_score",
        ],
        required=False,
    )

    if recent_col:

        df["recent_avg"] = pd.to_numeric(
            df[recent_col],
            errors="coerce",
        )

    else:

        df["recent_avg"] = df[
            "comprehensive_marks"
        ]

    # --------------------------------------------------------
    # Performance Trend
    # --------------------------------------------------------

    trend_col = detect_column(
        df,
        [
            "performance_trend",
            "trend",
            "academic_trend",
        ],
        required=False,
    )

    if trend_col:

        df["performance_trend"] = (
            df[trend_col]
            .astype(str)
            .str.strip()
            .str.title()
        )

        df["performance_trend"] = (
            df["performance_trend"]
            .replace(
                {
                    "Declined": "Declining",
                    "Decreasing": "Declining",
                    "Increased": "Improving",
                    "Increasing": "Improving",
                    "Improving/stable": "Stable",
                    "Improving / Stable": "Stable",
                }
            )
        )

    else:

        change = (
            df["recent_avg"]
            - df["previous_avg"]
        )

        df["performance_trend"] = np.select(
            [
                change >= 5,
                change <= -5,
            ],
            [
                "Improving",
                "Declining",
            ],
            default="Stable",
        )

    # --------------------------------------------------------
    # Missed Assessments
    # --------------------------------------------------------

    missed_col = detect_column(
        df,
        [
            "missed_assessments",
            "missed_assessment",
            "missed_tests",
            "absent_assessments",
        ],
        required=False,
    )

    if missed_col:

        df["missed_assessments"] = pd.to_numeric(
            df[missed_col],
            errors="coerce",
        ).fillna(0)

    else:

        # Derivation is used only if this field is absent from
        # the real dataset.
        df["missed_assessments"] = np.where(
            df["attendance"] < 75,
            2,
            0,
        )

    df["missed_assessments"] = (
        df["missed_assessments"]
        .clip(0, 5)
        .round()
        .astype(int)
    )

    # --------------------------------------------------------
    # Academic Risk Rule
    # --------------------------------------------------------

    df["risk"] = (
        (
            (df["recent_avg"] < 50)
            | (df["attendance"] < 70)
            | (df["missed_assessments"] >= 2)
            | (
                (df["performance_trend"] == "Declining")
                & (df["recent_avg"] < 60)
            )
        )
        .astype(int)
    )

    # --------------------------------------------------------
    # Grade
    # --------------------------------------------------------

    df["grade"] = np.select(
        [
            df["comprehensive_marks"] >= 80,
            df["comprehensive_marks"] >= 70,
            df["comprehensive_marks"] >= 60,
            df["comprehensive_marks"] >= 50,
        ],
        [
            "A",
            "B",
            "C",
            "D",
        ],
        default="F",
    )

    # --------------------------------------------------------
    # Clean data
    # --------------------------------------------------------

    required_numeric = [
        "attendance",
        "assignment_avg",
        "quiz_avg",
        "midsem_marks",
        "previous_avg",
        "recent_avg",
        "comprehensive_marks",
    ]

    for col in required_numeric:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    df = df.dropna(
        subset=[
            "student_id",
            "subject",
            "attendance",
            "assignment_avg",
            "quiz_avg",
            "midsem_marks",
            "recent_avg",
        ]
    ).copy()

    return df


# ============================================================
# ML TRAINING
# ============================================================

@st.cache_resource
def train_models(data: pd.DataFrame):

    X = data[
        FEATURE_COLUMNS
    ].copy()

    y = data[
        TARGET_COLUMN
    ].astype(int)

    X = X.apply(
        pd.to_numeric,
        errors="coerce",
    )

    X = X.fillna(
        X.median()
    )

    if y.nunique() < 2:
        raise ValueError(
            "The academic risk rule produced only one target "
            "class. Both risk classes are required for ML "
            "classification."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # --------------------------------------------------------
    # Logistic Regression
    # --------------------------------------------------------

    logistic = LogisticRegression(
        random_state=42,
        max_iter=1000,
    )

    logistic.fit(
        X_train,
        y_train,
    )

    logistic_pred = logistic.predict(
        X_test
    )

    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    forest = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    )

    forest.fit(
        X_train,
        y_train,
    )

    forest_pred = forest.predict(
        X_test
    )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    def evaluate(actual, predicted):

        return {
            "Accuracy": accuracy_score(
                actual,
                predicted,
            ),
            "Precision": precision_score(
                actual,
                predicted,
                zero_division=0,
            ),
            "Recall": recall_score(
                actual,
                predicted,
                zero_division=0,
            ),
            "F1-Score": f1_score(
                actual,
                predicted,
                zero_division=0,
            ),
            "Confusion Matrix": confusion_matrix(
                actual,
                predicted,
                labels=[0, 1],
            ),
        }

    metrics = {
        "Logistic Regression": evaluate(
            y_test,
            logistic_pred,
        ),
        "Random Forest": evaluate(
            y_test,
            forest_pred,
        ),
    }

    importance = pd.DataFrame(
        {
            "Feature": FEATURE_COLUMNS,
            "Importance": forest.feature_importances_,
        }
    ).sort_values(
        "Importance",
        ascending=False,
    )

    return (
        logistic,
        forest,
        metrics,
        importance,
    )


# ============================================================
# STUDENT-LEVEL AGGREGATION
# ============================================================

def student_summary(data):

    summary = (
        data.groupby(
            "student_id",
            as_index=False,
        )
        .agg(
            {
                "student_name": "first",
                "department": "first",
                "semester": "first",
                "attendance": "mean",
                "assignment_avg": "mean",
                "quiz_avg": "mean",
                "midsem_marks": "mean",
                "previous_avg": "mean",
                "recent_avg": "mean",
                "comprehensive_marks": "mean",
                "missed_assessments": "sum",
                "risk": "sum",
            }
        )
    )

    summary["grade"] = np.select(
        [
            summary["comprehensive_marks"] >= 80,
            summary["comprehensive_marks"] >= 70,
            summary["comprehensive_marks"] >= 60,
            summary["comprehensive_marks"] >= 50,
        ],
        [
            "A",
            "B",
            "C",
            "D",
        ],
        default="F",
    )

    return summary


# ============================================================
# MODEL RISK PREDICTION
# ============================================================

def predict_student_risk(
    student_df,
    model,
):

    X = student_df[
        FEATURE_COLUMNS
    ].copy()

    X = X.apply(
        pd.to_numeric,
        errors="coerce",
    )

    X = X.fillna(
        X.median()
    )

    probability = model.predict_proba(
        X
    )[:, 1]

    return (
        int(probability.mean() >= 0.50),
        float(probability.mean()),
    )


# ============================================================
# PRIORITY CALCULATION
# ============================================================

def priority_score(row):

    score = 0

    if row["recent_avg"] < 50:
        score += 4
    elif row["recent_avg"] < 60:
        score += 2

    if row["attendance"] < 70:
        score += 3
    elif row["attendance"] < 75:
        score += 1

    if row["missed_assessments"] >= 2:
        score += 3
    elif row["missed_assessments"] > 0:
        score += 1

    if row["performance_trend"] == "Declining":
        score += 2

    if row["midsem_marks"] < 50:
        score += 2

    return score


# ============================================================
# RECOMMENDATIONS
# ============================================================

def generate_recommendations(student_df):

    recommendations = []

    for _, row in student_df.iterrows():

        subject = row["subject"]
        recent = row["recent_avg"]
        attendance = row["attendance"]
        missed = row["missed_assessments"]
        trend = row["performance_trend"]

        # Rule 1
        if recent < 50 and trend == "Declining":

            recommendations.append(
                (
                    "Priority Action",
                    f"{subject} requires immediate attention "
                    f"due to declining scores ({recent:.1f}%).",
                )
            )

        # Rule 2
        if attendance < 75:

            recommendations.append(
                (
                    "Attendance Warning",
                    f"Attendance in {subject} is below threshold "
                    f"({attendance:.1f}%). Attendance improvement "
                    f"recommended.",
                )
            )

        # Rule 3
        if missed > 0:

            recommendations.append(
                (
                    "Assessment Alert",
                    f"{int(missed)} missed assessment(s) detected "
                    f"in {subject}. Coordinate with instructor for "
                    f"retakes or supplementary submission.",
                )
            )

        # Rule 4
        if recent >= 75 and trend == "Improving":

            recommendations.append(
                (
                    "Positive Progress",
                    f"Excellent performance in {subject}. "
                    f"Maintain current study strategy.",
                )
            )

    return recommendations


# ============================================================
# QUESTION ENGINE
# ============================================================

def answer_question(
    question,
    student_df,
):

    q = question.lower()

    # Focus
    if (
        "focus" in q
        or "which subject" in q
        or "priority" in q
    ):

        declining = student_df[
            student_df["performance_trend"]
            == "Declining"
        ]

        if not declining.empty:

            row = declining.loc[
                declining["recent_avg"].idxmin()
            ]

            return (
                f"Your first focus should be {row['subject']}. "
                f"It has the lowest recent average among your "
                f"declining subjects at {row['recent_avg']:.1f}%."
            )

        row = student_df.loc[
            student_df["recent_avg"].idxmin()
        ]

        return (
            f"Based on recent average, {row['subject']} currently "
            f"has the lowest score at {row['recent_avg']:.1f}%."
        )

    # Why decreased
    if (
        "decreased" in q
        or "decrease" in q
        or "declined" in q
        or "dropped" in q
        or "why" in q
    ):

        declining = student_df[
            student_df["performance_trend"]
            == "Declining"
        ]

        if declining.empty:

            return (
                "The available records do not currently show "
                "a Declining performance trend."
            )

        explanations = []

        for _, row in declining.iterrows():

            difference = (
                row["recent_avg"]
                - row["previous_avg"]
            )

            explanations.append(
                f"{row['subject']} changed by "
                f"{difference:+.1f} percentage points; "
                f"attendance is {row['attendance']:.1f}% and "
                f"{int(row['missed_assessments'])} assessment(s) "
                f"were missed."
            )

        return " ".join(explanations)

    # Attendance
    if "attendance" in q:

        average = student_df[
            "attendance"
        ].mean()

        low = student_df[
            student_df["attendance"] < 75
        ]

        if low.empty:

            return (
                f"Average attendance is {average:.1f}%. "
                "No subject is below the 75% threshold."
            )

        subjects = ", ".join(
            low["subject"]
        )

        return (
            f"Average attendance is {average:.1f}%. "
            f"Subjects below 75%: {subjects}."
        )

    # Performance
    if (
        "performance" in q
        or "marks" in q
        or "score" in q
    ):

        average = student_df[
            "recent_avg"
        ].mean()

        return (
            f"Your average recent score across the available "
            f"subjects is {average:.1f}%."
        )

    # Declining
    if "declining" in q:

        declining = student_df[
            student_df["performance_trend"]
            == "Declining"
        ]

        if declining.empty:

            return (
                "No subject is currently classified as Declining."
            )

        return (
            "Declining subjects: "
            + ", ".join(
                declining["subject"]
            )
        )

    # General
    average = student_df[
        "recent_avg"
    ].mean()

    attendance = student_df[
        "attendance"
    ].mean()

    risks = int(
        student_df["risk"].sum()
    )

    return (
        f"Your current academic summary is: average recent "
        f"score {average:.1f}%, average attendance "
        f"{attendance:.1f}%, with {risks} subject(s) meeting "
        f"the defined academic-risk criteria."
    )


# ============================================================
# LOAD DATA
# ============================================================

try:

    raw_data = load_dataset(
        DATA_PATH
    )

except FileNotFoundError:

    st.error(
        "Required dataset was not found."
    )

    st.code(
        DATA_PATH
    )

    st.stop()

except Exception as error:

    st.error(
        "Unable to read the Excel dataset."
    )

    st.exception(error)

    st.stop()


# ============================================================
# PREPARE DATA
# ============================================================

try:

    data = prepare_dataset(
        raw_data
    )

except Exception as error:

    st.error(
        "The Excel file could not be mapped to the "
        "academic dashboard fields."
    )

    st.exception(error)

    st.stop()


if data.empty:

    st.error(
        "No valid academic records were available "
        "after data preparation."
    )

    st.stop()


# ============================================================
# TRAIN MODELS
# ============================================================

try:

    (
        logistic_model,
        forest_model,
        model_metrics,
        feature_importance,
    ) = train_models(data)

except Exception as error:

    st.error(
        "Machine-learning model training could not be completed."
    )

    st.exception(error)

    st.stop()


# ============================================================
# STUDENT SUMMARY
# ============================================================

summary = student_summary(
    data
)

total_students = data[
    "student_id"
].nunique()

total_records = len(data)

total_subjects = data[
    "subject"
].nunique()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-brand">

            <div class="sidebar-brand-title">
                🎓 BITS Pilani
            </div>

            <div class="sidebar-brand-subtitle">
                AI-Based Student Performance Advisor
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "### Dashboard Controls"
    )

    st.markdown(
        "#### Student"
    )

    student_options = sorted(
        data["student_id"].unique()
    )

    selected_student = st.selectbox(
        "Select Student",
        student_options,
    )

    st.markdown(
        "#### Prediction Model"
    )

    model_choice = st.radio(
        "Select Model",
        [
            "Random Forest (Main)",
            "Logistic Regression (Baseline)",
        ],
    )

    if model_choice == "Random Forest (Main)":

        active_model = forest_model

    else:

        active_model = logistic_model

    st.markdown("---")

    st.markdown(
        "### Dataset Information"
    )

    st.write(
        f"**Unique Students:** {total_students}"
    )

    st.write(
        f"**Subject Records:** {total_records}"
    )

    st.write(
        f"**Subjects:** {total_subjects}"
    )

    st.caption(
        f"Source: {DATA_PATH}"
    )


# ============================================================
# SELECTED STUDENT
# ============================================================

selected_student_data = data[
    data["student_id"]
    == selected_student
].copy()

selected_summary = summary[
    summary["student_id"]
    == selected_student
].iloc[0]


predicted_class, risk_probability = predict_student_risk(
    selected_student_data,
    active_model,
)

risk_label = RISK_LABELS[
    predicted_class
]


# ============================================================
# HEADER
# ============================================================

header_left, header_right = st.columns(
    [4, 1]
)

with header_left:

    st.markdown(
        """
        <div class="dashboard-header">

            <div class="dashboard-title">
                Student Performance Dashboard
            </div>

            <div class="dashboard-subtitle">
                BITS Pilani · AI-Based Academic Learning Support
                and Performance Visualization System
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

with header_right:

    st.markdown(
        f"""
        <div class="dashboard-header">

            <div class="data-status">
                <strong>Live Dataset</strong><br>
                {total_students} unique students<br>
                {total_records} academic records
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# STUDENT PROFILE STRIP
# ============================================================

profile_col1, profile_col2, profile_col3, profile_col4 = st.columns(
    [2.2, 1.3, 1.3, 1.5]
)

with profile_col1:

    st.markdown(
        f"""
        <div class="profile-card">

            <div class="profile-name">
                👤 {selected_summary["student_name"]}
            </div>

            <div class="profile-id">
                Student ID: {selected_student}
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

with profile_col2:

    st.markdown(
        f"""
        <div class="profile-card">

            <div class="profile-stat-label">
                Department
            </div>

            <div class="profile-stat-value">
                {selected_summary["department"]}
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

with profile_col3:

    st.markdown(
        f"""
        <div class="profile-card">

            <div class="profile-stat-label">
                Semester
            </div>

            <div class="profile-stat-value">
                {selected_summary["semester"]}
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

with profile_col4:

    status_class = (
        "status-good"
        if predicted_class == 0
        else "status-warning"
    )

    status_icon = (
        "✅"
        if predicted_class == 0
        else "⚠️"
    )

    st.markdown(
        f"""
        <div class="{status_class}">
            {status_icon} {risk_label}
            <br>
            <small>
                ML probability: {risk_probability * 100:.1f}%
            </small>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# MAIN DASHBOARD TABS
# ============================================================

tab_overview, tab_subjects, tab_ml, tab_recommendations, tab_qa = st.tabs(
    [
        "Overview",
        "Subject Performance",
        "ML Diagnostics",
        "Study Recommendations",
        "Academic Q&A",
    ]
)


# ============================================================
# TAB 1 — OVERVIEW
# ============================================================

with tab_overview:

    # --------------------------------------------------------
    # KPI ROW
    # --------------------------------------------------------

    k1, k2, k3, k4 = st.columns(4)

    with k1:

        st.markdown(
            f"""
            <div class="kpi-card">

                <div class="kpi-title">
                    Average Student Mark
                </div>

                <div class="kpi-value">
                    {selected_student_data["recent_avg"].mean():.1f}
                </div>

                <div class="kpi-description">
                    Average recent score / 100
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    with k2:

        st.markdown(
            f"""
            <div class="kpi-card">

                <div class="kpi-title">
                    Attendance
                </div>

                <div class="kpi-value">
                    {selected_student_data["attendance"].mean():.1f}%
                </div>

                <div class="kpi-description">
                    Average attendance
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    with k3:

        st.markdown(
            f"""
            <div class="kpi-card">

                <div class="kpi-title">
                    Academic Risk Score
                </div>

                <div class="kpi-value">
                    {risk_probability * 100:.1f}%
                </div>

                <div class="kpi-description">
                    Predicted support probability
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    with k4:

        st.markdown(
            f"""
            <div class="kpi-card">

                <div class="kpi-title">
                    High-Risk Subjects
                </div>

                <div class="kpi-value">
                    {int(selected_student_data["risk"].sum())}
                </div>

                <div class="kpi-description">
                    Subjects requiring attention
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("")

    # --------------------------------------------------------
    # ACADEMIC STATUS
    # --------------------------------------------------------

    if predicted_class == 1:

        st.markdown(
            f"""
            <div class="status-danger">

                ⚠️ <strong>Learning Support Needed</strong>

                <br>

                The selected model estimates a
                <strong>{risk_probability * 100:.1f}%</strong>
                probability of academic risk.

                Review the subject-level results below.

            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        st.markdown(
            f"""
            <div class="status-good">

                ✅ <strong>Good Academic Standing</strong>

                <br>

                The selected model estimates a
                <strong>{risk_probability * 100:.1f}%</strong>
                probability of academic risk.

                Continue monitoring performance.

            </div>
            """,
            unsafe_allow_html=True,
        )

    # --------------------------------------------------------
    # MAIN TABLE
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="section-card">

            <div class="section-title">
                Academic Details of Student
            </div>

            <div class="section-description">
                Subject-level academic records from the
                selected student's actual dataset.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    overview = selected_student_data[
        [
            "subject",
            "attendance",
            "assignment_avg",
            "quiz_avg",
            "midsem_marks",
            "comprehensive_marks",
            "recent_avg",
            "performance_trend",
            "risk",
        ]
    ].copy()

    overview["risk"] = overview[
        "risk"
    ].map(RISK_LABELS)

    overview.columns = [
        "Subject",
        "Attendance %",
        "Assignment %",
        "Quiz %",
        "Midsem",
        "Comprehensive",
        "Recent Avg %",
        "Trend",
        "Academic Status",
    ]

    st.dataframe(
        overview.style.format(
            {
                "Attendance %": "{:.1f}",
                "Assignment %": "{:.1f}",
                "Quiz %": "{:.1f}",
                "Midsem": "{:.1f}",
                "Comprehensive": "{:.1f}",
                "Recent Avg %": "{:.1f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    # --------------------------------------------------------
    # THREE-COLUMN VISUAL AREA
    # --------------------------------------------------------

    st.markdown("")

    chart1, chart2, chart3 = st.columns(3)

    # --------------------------------------------------------
    # Chart 1 — Subject Recent Marks
    # --------------------------------------------------------

    with chart1:

        st.markdown(
            """
            <div class="section-title">
                Recent Performance by Subject
            </div>
            """,
            unsafe_allow_html=True,
        )

        recent_chart = selected_student_data.copy()

        fig = px.bar(
            recent_chart,
            x="subject",
            y="recent_avg",
            text="recent_avg",
            labels={
                "subject": "Subject",
                "recent_avg": "Recent Average (%)",
            },
        )

        fig.update_traces(
            texttemplate="%{text:.1f}",
            textposition="outside",
        )

        fig.update_yaxes(
            range=[0, 110],
            gridcolor="#cbd5e1",
            tickfont=dict(color="#000000"),
        )

        fig.update_xaxes(
            tickangle=-35,
            tickfont=dict(color="#000000"),
        )

        fig.update_layout(
            height=330,
            margin=dict(
                l=20,
                r=10,
                t=20,
                b=80,
            ),
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            font=dict(
                color="#000000"
            ),
            showlegend=False,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    # --------------------------------------------------------
    # Chart 2 — Attendance
    # --------------------------------------------------------

    with chart2:

        st.markdown(
            """
            <div class="section-title">
                Attendance by Subject
            </div>
            """,
            unsafe_allow_html=True,
        )

        attendance_chart = selected_student_data.copy()

        attendance_chart["Status"] = np.where(
            attendance_chart["attendance"] >= 75,
            "Regular Attendance",
            "Below 75%",
        )

        fig = px.bar(
            attendance_chart,
            x="subject",
            y="attendance",
            text="attendance",
            labels={
                "subject": "Subject",
                "attendance": "Attendance (%)",
            },
            color="Status",
        )

        fig.update_traces(
            texttemplate="%{text:.1f}%",
            textposition="outside",
        )

        fig.add_hline(
            y=75,
            line_dash="dash",
            annotation_text="75% Threshold",
            annotation_font_color="#000000",
        )

        fig.update_yaxes(
            range=[0, 110],
            gridcolor="#cbd5e1",
            tickfont=dict(color="#000000"),
        )

        fig.update_xaxes(
            tickangle=-35,
            tickfont=dict(color="#000000"),
        )

        fig.update_layout(
            height=330,
            margin=dict(
                l=20,
                r=10,
                t=20,
                b=80,
            ),
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            font=dict(
                color="#000000"
            ),
            legend=dict(
                font=dict(
                    color="#000000"
                )
            ),
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    # --------------------------------------------------------
    # Chart 3 — Grade Distribution
    # --------------------------------------------------------

    with chart3:

        st.markdown(
            """
            <div class="section-title">
                Student Grade Distribution
            </div>
            """,
            unsafe_allow_html=True,
        )

        grades = (
            selected_student_data[
                "grade"
            ]
            .value_counts()
            .reindex(
                ["A", "B", "C", "D", "F"],
                fill_value=0,
            )
            .reset_index()
        )

        grades.columns = [
            "Grade",
            "Subjects",
        ]

        fig = px.bar(
            grades,
            x="Grade",
            y="Subjects",
            text="Subjects",
            labels={
                "Grade": "Grade",
                "Subjects": "Number of Subjects",
            },
        )

        fig.update_traces(
            textposition="outside",
        )

        fig.update_yaxes(
            dtick=1,
            gridcolor="#cbd5e1",
            tickfont=dict(color="#000000"),
        )

        fig.update_xaxes(
            tickfont=dict(color="#000000"),
        )

        fig.update_layout(
            height=330,
            margin=dict(
                l=20,
                r=10,
                t=20,
                b=50,
            ),
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            font=dict(
                color="#000000"
            ),
            showlegend=False,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )


# ============================================================
# TAB 2 — SUBJECT PERFORMANCE
# ============================================================

with tab_subjects:

    st.subheader(
        "Subject-Wise Performance & Trends"
    )

    # --------------------------------------------------------
    # Marks comparison
    # --------------------------------------------------------

    chart_data = selected_student_data[
        [
            "subject",
            "previous_avg",
            "midsem_marks",
            "recent_avg",
        ]
    ].melt(
        id_vars="subject",
        var_name="Measure",
        value_name="Marks",
    )

    chart_data["Measure"] = chart_data[
        "Measure"
    ].replace(
        {
            "previous_avg": "Previous Average",
            "midsem_marks": "Midsem",
            "recent_avg": "Recent Average",
        }
    )

    fig = px.bar(
        chart_data,
        x="subject",
        y="Marks",
        color="Measure",
        barmode="group",
        text="Marks",
        labels={
            "subject": "Subject",
            "Marks": "Marks (%)",
            "Measure": "Measure",
        },
    )

    fig.update_traces(
        texttemplate="%{text:.1f}",
        textposition="outside",
    )

    fig.update_yaxes(
        range=[0, 110],
        gridcolor="#cbd5e1",
    )

    fig.update_layout(
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#000000"),
        height=450,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    # --------------------------------------------------------
    # Priority matrix
    # --------------------------------------------------------

    st.subheader(
        "Ranking & Priority Matrix"
    )

    priority = selected_student_data.copy()

    priority["Priority Score"] = (
        priority.apply(
            priority_score,
            axis=1,
        )
    )

    priority = priority.sort_values(
        [
            "Priority Score",
            "recent_avg",
        ],
        ascending=[
            False,
            True,
        ],
    )

    priority["Priority Rank"] = np.arange(
        1,
        len(priority) + 1,
    )

    priority_display = priority[
        [
            "Priority Rank",
            "subject",
            "Priority Score",
            "recent_avg",
            "attendance",
            "performance_trend",
            "missed_assessments",
        ]
    ].copy()

    priority_display.columns = [
        "Rank",
        "Subject",
        "Priority Score",
        "Recent Average %",
        "Attendance %",
        "Performance Trend",
        "Missed Assessments",
    ]

    st.dataframe(
        priority_display.style.format(
            {
                "Recent Average %": "{:.1f}",
                "Attendance %": "{:.1f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    # --------------------------------------------------------
    # Attendance / Performance relationship
    # --------------------------------------------------------

    st.subheader(
        "Attendance vs. Academic Performance"
    )

    fig = px.scatter(
        selected_student_data,
        x="attendance",
        y="recent_avg",
        text="subject",
        size="missed_assessments",
        color="performance_trend",
        labels={
            "attendance": "Attendance (%)",
            "recent_avg": "Recent Average (%)",
            "performance_trend": "Performance Trend",
        },
    )

    fig.add_vline(
        x=75,
        line_dash="dash",
        annotation_text="75% Attendance",
        annotation_font_color="#000000",
    )

    fig.add_hline(
        y=50,
        line_dash="dash",
        annotation_text="50% Score",
        annotation_font_color="#000000",
    )

    fig.update_traces(
        textposition="top center"
    )

    fig.update_layout(
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#000000"),
        height=450,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


# ============================================================
# TAB 3 — ML DIAGNOSTICS
# ============================================================

with tab_ml:

    st.subheader(
        "Machine-Learning Model Diagnostics"
    )

    # --------------------------------------------------------
    # Comparison table
    # --------------------------------------------------------

    model_rows = []

    for name in [
        "Logistic Regression",
        "Random Forest",
    ]:

        metric = model_metrics[name]

        model_rows.append(
            {
                "Model": name,
                "Accuracy": metric["Accuracy"],
                "Precision": metric["Precision"],
                "Recall": metric["Recall"],
                "F1-Score": metric["F1-Score"],
            }
        )

    comparison = pd.DataFrame(
        model_rows
    )

    st.dataframe(
        comparison.style.format(
            {
                "Accuracy": "{:.3f}",
                "Precision": "{:.3f}",
                "Recall": "{:.3f}",
                "F1-Score": "{:.3f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    # --------------------------------------------------------
    # Feature Importance
    # --------------------------------------------------------

    st.subheader(
        "Academic Indicators Influencing Risk Prediction"
    )

    readable = {
        "attendance": "Attendance",
        "assignment_avg": "Assignment Average",
        "quiz_avg": "Quiz Average",
        "midsem_marks": "Midsem Marks",
        "previous_avg": "Previous Average",
        "recent_avg": "Recent Average",
        "missed_assessments": "Missed Assessments",
    }

    importance = feature_importance.copy()

    importance["Feature"] = importance[
        "Feature"
    ].map(readable)

    importance = importance.sort_values(
        "Importance",
        ascending=True,
    )

    fig = px.bar(
        importance,
        x="Importance",
        y="Feature",
        orientation="h",
        text="Importance",
        labels={
            "Importance": "Importance",
            "Feature": "Academic Indicator",
        },
    )

    fig.update_traces(
        texttemplate="%{text:.3f}",
        textposition="outside",
    )

    fig.update_layout(
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#000000"),
        height=450,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    # --------------------------------------------------------
    # Confusion matrices
    # --------------------------------------------------------

    cm1, cm2 = st.columns(2)

    with cm1:

        st.markdown(
            "**Logistic Regression — Confusion Matrix**"
        )

        matrix = model_metrics[
            "Logistic Regression"
        ]["Confusion Matrix"]

        fig = go.Figure(
            data=go.Heatmap(
                z=matrix,
                x=[
                    "Predicted No Risk",
                    "Predicted Risk",
                ],
                y=[
                    "Actual No Risk",
                    "Actual Risk",
                ],
                text=matrix,
                texttemplate="%{text}",
                colorscale="Blues",
            )
        )

        fig.update_layout(
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            font=dict(color="#000000"),
            height=330,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    with cm2:

        st.markdown(
            "**Random Forest — Confusion Matrix**"
        )

        matrix = model_metrics[
            "Random Forest"
        ]["Confusion Matrix"]

        fig = go.Figure(
            data=go.Heatmap(
                z=matrix,
                x=[
                    "Predicted No Risk",
                    "Predicted Risk",
                ],
                y=[
                    "Actual No Risk",
                    "Actual Risk",
                ],
                text=matrix,
                texttemplate="%{text}",
                colorscale="Blues",
            )
        )

        fig.update_layout(
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            font=dict(color="#000000"),
            height=330,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    # --------------------------------------------------------
    # Current student model output
    # --------------------------------------------------------

    st.subheader(
        f"Current Student Prediction — {selected_student}"
    )

    prediction_col1, prediction_col2 = st.columns(2)

    with prediction_col1:

        st.metric(
            "Predicted Academic Status",
            risk_label,
        )

    with prediction_col2:

        st.metric(
            "Risk Probability",
            f"{risk_probability * 100:.1f}%",
        )


# ============================================================
# TAB 4 — RECOMMENDATIONS
# ============================================================

with tab_recommendations:

    st.subheader(
        "Personalized Study Improvement Plan"
    )

    st.caption(
        "Recommendations are generated only from the selected "
        "student's academic records and the configured rules."
    )

    recommendations = generate_recommendations(
        selected_student_data
    )

    if not recommendations:

        st.markdown(
            """
            <div class="recommendation">

                <div class="recommendation-heading">
                    ✅ No Specific Intervention Rule Triggered
                </div>

                <div class="recommendation-text">
                    The available academic records do not
                    currently trigger any of the configured
                    recommendation rules.
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        for title, text in recommendations:

            st.markdown(
                f"""
                <div class="recommendation">

                    <div class="recommendation-heading">
                        {title}
                    </div>

                    <div class="recommendation-text">
                        {text}
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

    # --------------------------------------------------------
    # Subject-level study summary
    # --------------------------------------------------------

    st.subheader(
        "Study Focus Summary"
    )

    study = selected_student_data.copy()

    study["Priority Score"] = study.apply(
        priority_score,
        axis=1,
    )

    study = study.sort_values(
        "Priority Score",
        ascending=False,
    )

    for _, row in study.iterrows():

        if row["Priority Score"] >= 5:

            status = "High Focus"

        elif row["Priority Score"] >= 2:

            status = "Moderate Focus"

        else:

            status = "Maintain"

        st.markdown(
            f"""
            <div class="recommendation">

                <div class="recommendation-heading">
                    {row["subject"]} — {status}
                </div>

                <div class="recommendation-text">
                    Recent average:
                    <strong>{row["recent_avg"]:.1f}%</strong>
                    &nbsp; | &nbsp;
                    Attendance:
                    <strong>{row["attendance"]:.1f}%</strong>
                    &nbsp; | &nbsp;
                    Trend:
                    <strong>{row["performance_trend"]}</strong>
                    &nbsp; | &nbsp;
                    Missed assessments:
                    <strong>{int(row["missed_assessments"])}</strong>
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# TAB 5 — CONVERSATIONAL Q&A
# ============================================================

with tab_qa:

    st.subheader(
        "💬 Academic Q&A"
    )

    st.write(
        f"Ask about the academic records of {selected_student}."
    )

    suggested_questions = [
        "Which subject should I focus on?",
        "Why has my performance decreased?",
        "What is my attendance situation?",
        "What is my recent academic performance?",
        "Which subjects are declining?",
    ]

    selected_question = st.selectbox(
        "Choose a question",
        [
            "Select a question..."
        ] + suggested_questions,
    )

    custom_question = st.text_input(
        "Or type your own question",
        placeholder=(
            "Example: Which subject needs the most attention?"
        ),
    )

    if st.button(
        "Get Academic Answer",
        type="primary",
    ):

        if custom_question.strip():

            question = custom_question.strip()

        elif selected_question != "Select a question...":

            question = selected_question

        else:

            st.warning(
                "Please select or enter a question."
            )

            question = None

        if question:

            answer = answer_question(
                question,
                selected_student_data,
            )

            st.chat_message(
                "user"
            ).write(
                question
            )

            with st.chat_message(
                "assistant"
            ):

                st.markdown(
                    f"""
                    <div class="answer-card">
                        {answer}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    f"""
    <div class="footer">

        <strong>
            BITS Pilani — AI-Based Student Performance Advisor
        </strong>

        <br>

        Real academic dataset:
        {DATA_PATH}

        <br>

        {total_students} unique students
        ·
        {total_records} subject records
        ·
        {total_subjects} subjects

    </div>
    """,
    unsafe_allow_html=True,
)
