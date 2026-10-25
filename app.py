# app.py
# ============================================================
# BITS Pilani
# AI-Based Student Performance Advisor and Visualization System
# ============================================================

import os
import warnings
from typing import Dict, List, Tuple

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
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="BITS Pilani | AI Student Performance Advisor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CONSTANTS
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

SUBJECTS = [
    "Mathematics",
    "Statistics",
    "Data Structures",
    "Machine Learning",
    "Database Systems",
]

RISK_LABELS = {
    0: "No Immediate Risk",
    1: "Requires Academic Attention",
}


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #0b1220;
        color: #ffffff;
    }

    html, body, [class*="css"] {
        color: #ffffff !important;
    }

    section[data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #334155;
    }

    section[data-testid="stSidebar"] * {
        color: #ffffff !important;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-weight: 800 !important;
    }

    p, span, label {
        color: #ffffff;
    }

    .main-title {
        font-size: 2.35rem;
        font-weight: 900;
        color: #ffffff !important;
        margin-bottom: 0.15rem;
    }

    .subtitle {
        color: #cbd5e1 !important;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    .metric-card {
        background: #111827;
        border: 1px solid #475569;
        border-radius: 14px;
        padding: 20px;
        min-height: 135px;
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.30);
    }

    .metric-label {
        color: #cbd5e1 !important;
        font-size: 0.88rem;
        font-weight: 700;
    }

    .metric-value {
        color: #ffffff !important;
        font-size: 2rem;
        font-weight: 900;
        margin-top: 8px;
    }

    .metric-help {
        color: #94a3b8 !important;
        font-size: 0.8rem;
        margin-top: 5px;
    }

    .risk-high {
        background: #7f1d1d;
        border: 2px solid #fca5a5;
        border-radius: 12px;
        padding: 18px 22px;
        margin: 15px 0;
    }

    .risk-low {
        background: #14532d;
        border: 2px solid #86efac;
        border-radius: 12px;
        padding: 18px 22px;
        margin: 15px 0;
    }

    .recommendation-card {
        background: #111827;
        border: 1px solid #475569;
        border-left: 5px solid #38bdf8;
        border-radius: 10px;
        padding: 15px 18px;
        margin: 9px 0;
    }

    .recommendation-title {
        color: #ffffff !important;
        font-weight: 800;
    }

    .recommendation-body {
        color: #e2e8f0 !important;
        margin-top: 4px;
    }

    .qa-answer {
        background: #1e293b;
        border: 1px solid #475569;
        border-radius: 10px;
        padding: 14px;
        color: #ffffff !important;
        margin-top: 8px;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid #475569;
        border-radius: 8px;
    }

    div[data-baseweb="select"] {
        background-color: #ffffff !important;
    }

    div[data-baseweb="select"] * {
        color: #000000 !important;
    }

    .stButton > button {
        color: #000000 !important;
        background-color: #ffffff !important;
        border: 1px solid #94a3b8;
        font-weight: 800;
        border-radius: 8px;
    }

    button[data-baseweb="tab"] {
        color: #cbd5e1 !important;
        font-weight: 700;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #ffffff !important;
    }

    hr {
        border-color: #334155 !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_excel_dataset(path: str) -> pd.DataFrame:
    """
    Load the real BITS Pilani student dataset.

    No synthetic data is generated.
    """

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at: {path}"
        )

    try:
        df = pd.read_excel(
            path,
            engine="openpyxl",
        )
    except Exception as exc:
        raise RuntimeError(
            f"Unable to read Excel dataset: {exc}"
        ) from exc

    if df.empty:
        raise ValueError(
            "The Excel dataset is empty."
        )

    # Normalize column names.
    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )

    return df


# ============================================================
# COLUMN DETECTION
# ============================================================

def find_column(
    df: pd.DataFrame,
    possible_names: List[str],
    required: bool = True,
) -> str:

    for name in possible_names:

        normalized = (
            name.lower()
            .strip()
            .replace(" ", "_")
            .replace("-", "_")
        )

        if normalized in df.columns:
            return normalized

    if required:
        raise ValueError(
            "Required column not found. Expected one of: "
            + ", ".join(possible_names)
        )

    return ""


def standardize_dataset(
    raw_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert the actual Excel dataset into the internal
    academic analytics schema.

    The original dataset remains unchanged on disk.
    """

    df = raw_df.copy()

    # --------------------------------------------------------
    # Detect Student ID
    # --------------------------------------------------------

    student_col = find_column(
        df,
        [
            "student_id",
            "studentid",
            "student id",
            "id",
        ],
    )

    df["student_id"] = (
        df[student_col]
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------------------
    # Detect Subject
    # --------------------------------------------------------

    subject_col = find_column(
        df,
        [
            "subject",
            "course",
            "module",
            "subject_name",
        ],
    )

    df["subject"] = (
        df[subject_col]
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------------------
    # Detect academic columns
    # --------------------------------------------------------

    attendance_col = find_column(
        df,
        [
            "attendance",
            "attendance_percentage",
            "attendance_percent",
            "attendance_%",
        ],
    )

    assignment_col = find_column(
        df,
        [
            "assignment_avg",
            "assignment_average",
            "assignment_marks",
            "assignment_score",
            "assignment",
        ],
    )

    quiz_col = find_column(
        df,
        [
            "quiz_avg",
            "quiz_average",
            "quiz_score",
            "quiz_marks",
            "quiz",
        ],
    )

    midsem_col = find_column(
        df,
        [
            "midsem_marks",
            "midterm_marks",
            "midterm",
            "mid_sem_marks",
            "mid_sem",
        ],
    )

    final_col = find_column(
        df,
        [
            "comprehensive_marks",
            "final_marks",
            "final_mark",
            "comprehensive",
            "final",
        ],
    )

    # --------------------------------------------------------
    # Convert numeric fields
    # --------------------------------------------------------

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
    # Previous and recent averages
    # --------------------------------------------------------

    previous_col = find_column(
        df,
        [
            "previous_avg",
            "previous_average",
            "previous_marks",
            "previous_score",
        ],
        required=False,
    )

    recent_col = find_column(
        df,
        [
            "recent_avg",
            "recent_average",
            "recent_marks",
            "recent_score",
        ],
        required=False,
    )

    if previous_col:

        df["previous_avg"] = pd.to_numeric(
            df[previous_col],
            errors="coerce",
        )

    else:

        # If the source dataset does not have a previous-average
        # field, use Midterm Marks as the closest historical
        # academic measure.
        df["previous_avg"] = df[
            "midsem_marks"
        ]

    if recent_col:

        df["recent_avg"] = pd.to_numeric(
            df[recent_col],
            errors="coerce",
        )

    else:

        # The final/comprehensive mark is the latest available
        # assessment measure.
        df["recent_avg"] = df[
            "comprehensive_marks"
        ]

    # --------------------------------------------------------
    # Performance Trend
    # --------------------------------------------------------

    trend_col = find_column(
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

        # Normalize common variants.
        df["performance_trend"] = (
            df["performance_trend"]
            .replace(
                {
                    "Improving/stable": "Improving",
                    "Improving / Stable": "Improving",
                    "Needs Attention": "Declining",
                    "Declined": "Declining",
                }
            )
        )

    else:

        difference = (
            df["recent_avg"]
            - df["previous_avg"]
        )

        df["performance_trend"] = np.select(
            [
                difference >= 5,
                difference <= -5,
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

    missed_col = find_column(
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

        # Dataset does not contain this field.
        # Attendance is used only to derive the requested
        # academic support indicator.
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
    # Primary-Key Aggregation
    # --------------------------------------------------------
    #
    # The Excel dataset may contain one row per student/subject.
    # Student-level analytics are calculated from subject records,
    # while subject-level records remain available for detailed
    # views.
    #
    # We do NOT reduce the subject dataset to one row per student
    # here because the dashboard requires subject-level analysis.

    # --------------------------------------------------------
    # Calculate Risk
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
    # Clean numeric values
    # --------------------------------------------------------

    numeric_columns = [
        "attendance",
        "assignment_avg",
        "quiz_avg",
        "midsem_marks",
        "comprehensive_marks",
        "previous_avg",
        "recent_avg",
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # Remove rows where core academic information is unusable.
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
# MODEL TRAINING
# ============================================================

@st.cache_resource
def train_models(
    data: pd.DataFrame,
):
    """
    Train Logistic Regression and Random Forest models
    using the real Excel dataset.
    """

    X = data[FEATURE_COLUMNS].copy()

    y = data[TARGET_COLUMN].astype(int)

    X = X.apply(
        pd.to_numeric,
        errors="coerce",
    )

    X = X.fillna(
        X.median()
    )

    if y.nunique() < 2:

        raise ValueError(
            "The calculated academic risk target contains only "
            "one class. Both risk classes are required for "
            "classification training."
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

    logistic_model = LogisticRegression(
        random_state=42,
        max_iter=1000,
    )

    logistic_model.fit(
        X_train,
        y_train,
    )

    logistic_pred = logistic_model.predict(
        X_test
    )

    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    random_forest_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    )

    random_forest_model.fit(
        X_train,
        y_train,
    )

    rf_pred = random_forest_model.predict(
        X_test
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    def metrics_for_model(
        actual,
        predicted,
    ):

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
        "Logistic Regression": metrics_for_model(
            y_test,
            logistic_pred,
        ),
        "Random Forest": metrics_for_model(
            y_test,
            rf_pred,
        ),
    }

    feature_importance = pd.DataFrame(
        {
            "Feature": FEATURE_COLUMNS,
            "Importance": random_forest_model.feature_importances_,
        }
    ).sort_values(
        "Importance",
        ascending=False,
    )

    return (
        logistic_model,
        random_forest_model,
        metrics,
        feature_importance,
    )


# ============================================================
# STUDENT ANALYTICS
# ============================================================

def get_student_data(
    data: pd.DataFrame,
    student_id: str,
) -> pd.DataFrame:

    return data[
        data["student_id"] == student_id
    ].copy()


def get_student_risk_probability(
    student_df: pd.DataFrame,
    model,
) -> float:

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

    probabilities = model.predict_proba(
        X
    )[:, 1]

    return float(
        probabilities.mean()
    )


def calculate_priority(
    row: pd.Series,
) -> int:

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
# RECOMMENDATION ENGINE
# ============================================================

def generate_recommendations(
    student_df: pd.DataFrame,
) -> List[Dict[str, str]]:

    recommendations = []

    for _, row in student_df.iterrows():

        subject = row["subject"]
        recent = row["recent_avg"]
        attendance = row["attendance"]
        missed = row["missed_assessments"]
        trend = row["performance_trend"]

        # Rule 1
        if (
            recent < 50
            and trend == "Declining"
        ):

            recommendations.append(
                {
                    "type": "Priority Action",
                    "text": (
                        f"Priority Action: {subject} requires "
                        f"immediate attention due to declining "
                        f"scores ({recent:.1f}%)."
                    ),
                }
            )

        # Rule 2
        if attendance < 75:

            recommendations.append(
                {
                    "type": "Attendance Warning",
                    "text": (
                        f"Attendance Warning: Attendance in "
                        f"{subject} is below threshold "
                        f"({attendance:.1f}%). Attendance "
                        f"improvement recommended."
                    ),
                }
            )

        # Rule 3
        if missed > 0:

            recommendations.append(
                {
                    "type": "Assessment Alert",
                    "text": (
                        f"Assessment Alert: {int(missed)} missed "
                        f"assessment(s) detected. Coordinate "
                        f"with instructor for retakes or "
                        f"supplementary submission."
                    ),
                }
            )

        # Rule 4
        if (
            recent >= 75
            and trend == "Improving"
        ):

            recommendations.append(
                {
                    "type": "Positive Progress",
                    "text": (
                        f"Positive Progress: Excellent performance "
                        f"in {subject}. Maintain current study strategy."
                    ),
                }
            )

    return recommendations


# ============================================================
# QUESTION ANSWERS
# ============================================================

def answer_question(
    question: str,
    student_df: pd.DataFrame,
) -> str:

    q = question.lower().strip()

    # --------------------------------------------------------
    # Focus question
    # --------------------------------------------------------

    if (
        "focus" in q
        or "which subject" in q
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
                f"You should focus first on {row['subject']}. "
                f"It has the lowest recent average among your "
                f"declining subjects at {row['recent_avg']:.1f}%."
            )

        row = student_df.loc[
            student_df["recent_avg"].idxmin()
        ]

        return (
            f"No subject is currently classified as Declining. "
            f"Based on recent average, {row['subject']} has "
            f"the lowest score at {row['recent_avg']:.1f}%."
        )

    # --------------------------------------------------------
    # Performance decrease
    # --------------------------------------------------------

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
                "any subject classified as Declining."
            )

        responses = []

        for _, row in declining.iterrows():

            change = (
                row["recent_avg"]
                - row["previous_avg"]
            )

            response = (
                f"{row['subject']}: recent average "
                f"{row['recent_avg']:.1f}% compared with "
                f"previous average {row['previous_avg']:.1f}% "
                f"(change {change:+.1f} percentage points), "
                f"attendance {row['attendance']:.1f}%, and "
                f"{int(row['missed_assessments'])} missed "
                f"assessment(s)."
            )

            responses.append(response)

        return " ".join(responses)

    # --------------------------------------------------------
    # Attendance
    # --------------------------------------------------------

    if "attendance" in q:

        average = student_df[
            "attendance"
        ].mean()

        low_attendance = student_df[
            student_df["attendance"] < 75
        ]

        if low_attendance.empty:

            return (
                f"Your average attendance across the available "
                f"subjects is {average:.1f}%. No subject is below "
                f"the 75% attendance threshold."
            )

        subjects = ", ".join(
            low_attendance["subject"]
            .tolist()
        )

        return (
            f"Your average attendance is {average:.1f}%. "
            f"The following subjects are below 75%: {subjects}."
        )

    # --------------------------------------------------------
    # Recent performance
    # --------------------------------------------------------

    if (
        "performance" in q
        or "marks" in q
        or "score" in q
    ):

        average = student_df[
            "recent_avg"
        ].mean()

        return (
            f"Your average recent academic score across the "
            f"available subjects is {average:.1f}%."
        )

    # --------------------------------------------------------
    # Declining subjects
    # --------------------------------------------------------

    if "declining" in q:

        declining = student_df[
            student_df["performance_trend"]
            == "Declining"
        ]

        if declining.empty:

            return (
                "There are no subjects currently classified "
                "as Declining."
            )

        subjects = ", ".join(
            declining["subject"].tolist()
        )

        return (
            f"The subjects currently classified as Declining "
            f"are: {subjects}."
        )

    # --------------------------------------------------------
    # General fallback
    # --------------------------------------------------------

    average_recent = student_df[
        "recent_avg"
    ].mean()

    average_attendance = student_df[
        "attendance"
    ].mean()

    risk_subjects = int(
        student_df["risk"].sum()
    )

    return (
        f"Based on the available academic records, your average "
        f"recent score is {average_recent:.1f}%, average "
        f"attendance is {average_attendance:.1f}%, and "
        f"{risk_subjects} subject(s) currently meet the defined "
        f"academic-risk rule."
    )


# ============================================================
# LOAD REAL DATASET
# ============================================================

try:

    raw_data = load_excel_dataset(
        DATA_PATH
    )

except FileNotFoundError as exc:

    st.error(
        "❌ Required dataset was not found."
    )

    st.info(
        f"Place the Excel file at:\n\n"
        f"`{DATA_PATH}`"
    )

    st.stop()

except Exception as exc:

    st.error(
        "❌ The dataset could not be loaded."
    )

    st.exception(exc)

    st.stop()


# ============================================================
# STANDARDIZE DATASET
# ============================================================

try:

    data = standardize_dataset(
        raw_data
    )

except Exception as exc:

    st.error(
        "❌ The Excel dataset could not be mapped "
        "to the required academic fields."
    )

    st.info(
        "Please check the column names in "
        "`Subject_Analytics_Dataset_200_Students.xlsx`."
    )

    st.exception(exc)

    st.stop()


if data.empty:

    st.error(
        "No usable academic records remain after data cleaning."
    )

    st.stop()


# ============================================================
# DATASET SUMMARY
# ============================================================

total_students = data[
    "student_id"
].nunique()

total_subject_records = len(data)

total_subjects = data[
    "subject"
].nunique()


# ============================================================
# TRAIN ML MODELS
# ============================================================

try:

    (
        logistic_model,
        random_forest_model,
        model_metrics,
        feature_importance,
    ) = train_models(data)

except Exception as exc:

    st.error(
        "❌ Machine-learning training failed."
    )

    st.exception(exc)

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    """
    <div style="
        font-size:1.35rem;
        font-weight:900;
        color:#ffffff;
    ">
        🎓 BITS Pilani
    </div>

    <div style="
        font-size:0.95rem;
        color:#cbd5e1;
        margin-top:5px;
    ">
        AI Student Performance Advisor
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    "### Dataset"
)

st.sidebar.success(
    "Using the real Excel dataset"
)

st.sidebar.caption(
    DATA_PATH
)

st.sidebar.markdown(
    f"""
    **Students:** {total_students}  
    **Subject Records:** {total_subject_records}  
    **Subjects:** {total_subjects}
    """
)

st.sidebar.markdown("---")

# ------------------------------------------------------------
# Student Selector
# ------------------------------------------------------------

student_ids = sorted(
    data["student_id"]
    .unique()
    .tolist()
)

selected_student = st.sidebar.selectbox(
    "Select Student",
    student_ids,
)

# ------------------------------------------------------------
# Model Selector
# ------------------------------------------------------------

selected_model_name = st.sidebar.radio(
    "Prediction Model",
    [
        "Random Forest (Main)",
        "Logistic Regression (Baseline)",
    ],
)

if (
    selected_model_name
    == "Random Forest (Main)"
):

    selected_model = random_forest_model

else:

    selected_model = logistic_model


# ============================================================
# SELECTED STUDENT
# ============================================================

student_df = data[
    data["student_id"]
    == selected_student
].copy()

if student_df.empty:

    st.error(
        "No academic records were found for the selected student."
    )

    st.stop()


# ============================================================
# STUDENT RISK PREDICTION
# ============================================================

risk_probability = (
    get_student_risk_probability(
        student_df,
        selected_model,
    )
)

risk_class = int(
    risk_probability >= 0.50
)

risk_label = RISK_LABELS[
    risk_class
]


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="main-title">
        🎓 BITS Pilani — AI-Based Student Performance Advisor
    </div>

    <div class="subtitle">
        Academic performance analysis, learning support,
        machine-learning risk classification, and
        personalized study guidance.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📊 Overview & Academic Risk",
        "📚 Subject Performance & Trends",
        "🤖 ML Diagnostics & Insights",
        "📝 Personalized Recommendations",
        "💬 Conversational Q&A",
    ]
)


# ============================================================
# TAB 1 — OVERVIEW
# ============================================================

with tab1:

    st.subheader(
        f"Academic Risk Status — {selected_student}"
    )

    overall_average = student_df[
        "recent_avg"
    ].mean()

    overall_attendance = student_df[
        "attendance"
    ].mean()

    high_risk_subjects = int(
        student_df["risk"].sum()
    )

    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">
                    Overall Average Marks
                </div>
                <div class="metric-value">
                    {overall_average:.1f}%
                </div>
                <div class="metric-help">
                    Average recent score
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">
                    Overall Attendance
                </div>
                <div class="metric-value">
                    {overall_attendance:.1f}%
                </div>
                <div class="metric-help">
                    Average across subjects
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">
                    Academic Risk Score
                </div>
                <div class="metric-value">
                    {risk_probability * 100:.1f}%
                </div>
                <div class="metric-help">
                    ML predicted risk probability
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c4:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">
                    High-Risk Subjects
                </div>
                <div class="metric-value">
                    {high_risk_subjects}
                </div>
                <div class="metric-help">
                    Subjects meeting risk criteria
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --------------------------------------------------------
    # RISK ALERT
    # --------------------------------------------------------

    if risk_class == 1:

        st.markdown(
            f"""
            <div class="risk-high">
                <h3 style="color:#ffffff;">
                    ⚠️ Requires Academic Attention
                </h3>

                <div style="color:#ffffff;">
                    The {selected_model_name} model predicts a
                    {risk_probability * 100:.1f}% probability of
                    academic risk for {selected_student}.
                    Review the subject-level indicators and
                    personalized recommendations.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        st.markdown(
            f"""
            <div class="risk-low">
                <h3 style="color:#ffffff;">
                    ✅ No Immediate Risk
                </h3>

                <div style="color:#ffffff;">
                    The {selected_model_name} model predicts a
                    {risk_probability * 100:.1f}% probability of
                    academic risk for {selected_student}.
                    Continue monitoring academic performance.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --------------------------------------------------------
    # SUBJECT STATUS
    # --------------------------------------------------------

    st.subheader(
        "Subject-Level Academic Status"
    )

    overview_table = student_df[
        [
            "subject",
            "recent_avg",
            "attendance",
            "performance_trend",
            "missed_assessments",
            "risk",
        ]
    ].copy()

    overview_table["risk"] = (
        overview_table["risk"]
        .map(RISK_LABELS)
    )

    overview_table.columns = [
        "Subject",
        "Recent Average (%)",
        "Attendance (%)",
        "Performance Trend",
        "Missed Assessments",
        "Academic Risk",
    ]

    st.dataframe(
        overview_table.style.format(
            {
                "Recent Average (%)": "{:.1f}",
                "Attendance (%)": "{:.1f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# TAB 2 — SUBJECT PERFORMANCE
# ============================================================

with tab2:

    st.subheader(
        "Subject-Wise Performance & Trends"
    )

    # --------------------------------------------------------
    # MARKS BREAKDOWN
    # --------------------------------------------------------

    chart_data = student_df[
        [
            "subject",
            "midsem_marks",
            "recent_avg",
            "previous_avg",
        ]
    ].melt(
        id_vars="subject",
        var_name="Measure",
        value_name="Marks",
    )

    chart_data["Measure"] = (
        chart_data["Measure"]
        .replace(
            {
                "midsem_marks": "Midsem",
                "recent_avg": "Recent Average",
                "previous_avg": "Previous Average",
            }
        )
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
            "Measure": "Academic Measure",
        },
    )

    fig.update_traces(
        texttemplate="%{text:.1f}",
        textposition="outside",
    )

    fig.update_yaxes(
        range=[0, 110],
        gridcolor="#475569",
        tickfont=dict(
            color="#ffffff"
        ),
        title_font=dict(
            color="#ffffff"
        ),
    )

    fig.update_xaxes(
        tickfont=dict(
            color="#ffffff"
        ),
        title_font=dict(
            color="#ffffff"
        ),
    )

    fig.update_layout(
        title="Subject Marks Breakdown",
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        font=dict(
            color="#ffffff"
        ),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    # --------------------------------------------------------
    # PRIORITY MATRIX
    # --------------------------------------------------------

    st.subheader(
        "Ranking & Priority Matrix"
    )

    priority = student_df.copy()

    priority["Priority Score"] = (
        priority.apply(
            calculate_priority,
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
            "previous_avg",
            "attendance",
            "performance_trend",
            "missed_assessments",
        ]
    ].copy()

    priority_display.columns = [
        "Priority Rank",
        "Subject",
        "Priority Score",
        "Recent Average (%)",
        "Previous Average (%)",
        "Attendance (%)",
        "Performance Trend",
        "Missed Assessments",
    ]

    st.dataframe(
        priority_display.style.format(
            {
                "Recent Average (%)": "{:.1f}",
                "Previous Average (%)": "{:.1f}",
                "Attendance (%)": "{:.1f}",
                "Priority Score": "{:.0f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    # --------------------------------------------------------
    # ATTENDANCE VS PERFORMANCE
    # --------------------------------------------------------

    st.subheader(
        "Attendance vs. Performance"
    )

    fig_scatter = px.scatter(
        student_df,
        x="attendance",
        y="recent_avg",
        color="performance_trend",
        size="missed_assessments",
        hover_name="subject",
        text="subject",
        labels={
            "attendance": "Attendance (%)",
            "recent_avg": "Recent Average (%)",
            "performance_trend": "Performance Trend",
        },
    )

    fig_scatter.update_traces(
        textposition="top center"
    )

    fig_scatter.add_vline(
        x=75,
        line_dash="dash",
        annotation_text="75% Attendance Threshold",
        annotation_font_color="#ffffff",
    )

    fig_scatter.add_hline(
        y=50,
        line_dash="dash",
        annotation_text="50% Recent Average",
        annotation_font_color="#ffffff",
    )

    fig_scatter.update_layout(
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        font=dict(
            color="#ffffff"
        ),
    )

    st.plotly_chart(
        fig_scatter,
        use_container_width=True,
    )


# ============================================================
# TAB 3 — ML DIAGNOSTICS
# ============================================================

with tab3:

    st.subheader(
        "ML Model Diagnostics & Insights"
    )

    # --------------------------------------------------------
    # MODEL COMPARISON
    # --------------------------------------------------------

    rows = []

    for model_name in [
        "Logistic Regression",
        "Random Forest",
    ]:

        m = model_metrics[
            model_name
        ]

        rows.append(
            {
                "Model": model_name,
                "Accuracy": m["Accuracy"],
                "Precision": m["Precision"],
                "Recall": m["Recall"],
                "F1-Score": m["F1-Score"],
            }
        )

    comparison = pd.DataFrame(rows)

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
    # CONFUSION MATRICES
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            "#### Logistic Regression"
        )

        cm = model_metrics[
            "Logistic Regression"
        ]["Confusion Matrix"]

        fig_cm = go.Figure(
            data=go.Heatmap(
                z=cm,
                x=[
                    "Predicted No Risk",
                    "Predicted Risk",
                ],
                y=[
                    "Actual No Risk",
                    "Actual Risk",
                ],
                text=cm,
                texttemplate="%{text}",
                colorscale="Blues",
            )
        )

        fig_cm.update_layout(
            paper_bgcolor="#111827",
            plot_bgcolor="#111827",
            font=dict(
                color="#ffffff"
            ),
        )

        st.plotly_chart(
            fig_cm,
            use_container_width=True,
        )

    with col2:

        st.markdown(
            "#### Random Forest"
        )

        cm = model_metrics[
            "Random Forest"
        ]["Confusion Matrix"]

        fig_cm = go.Figure(
            data=go.Heatmap(
                z=cm,
                x=[
                    "Predicted No Risk",
                    "Predicted Risk",
                ],
                y=[
                    "Actual No Risk",
                    "Actual Risk",
                ],
                text=cm,
                texttemplate="%{text}",
                colorscale="Blues",
            )
        )

        fig_cm.update_layout(
            paper_bgcolor="#111827",
            plot_bgcolor="#111827",
            font=dict(
                color="#ffffff"
            ),
        )

        st.plotly_chart(
            fig_cm,
            use_container_width=True,
        )

    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    st.subheader(
        "Feature Importance — Random Forest"
    )

    readable_names = {
        "attendance": "Attendance",
        "assignment_avg": "Assignment Average",
        "quiz_avg": "Quiz Average",
        "midsem_marks": "Midsem Marks",
        "previous_avg": "Previous Average",
        "recent_avg": "Recent Average",
        "missed_assessments": "Missed Assessments",
    }

    importance = feature_importance.copy()

    importance["Feature"] = (
        importance["Feature"]
        .map(readable_names)
    )

    importance = importance.sort_values(
        "Importance",
        ascending=True,
    )

    fig_importance = px.bar(
        importance,
        x="Importance",
        y="Feature",
        orientation="h",
        text="Importance",
        labels={
            "Importance": "Feature Importance",
            "Feature": "Academic Indicator",
        },
    )

    fig_importance.update_traces(
        texttemplate="%{text:.3f}",
        textposition="outside",
    )

    fig_importance.update_layout(
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        font=dict(
            color="#ffffff"
        ),
    )

    st.plotly_chart(
        fig_importance,
        use_container_width=True,
    )

    # --------------------------------------------------------
    # MODEL OUTPUT
    # --------------------------------------------------------

    st.subheader(
        f"Model Risk Output — {selected_student}"
    )

    output1, output2 = st.columns(2)

    with output1:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">
                    Prediction Model
                </div>
                <div class="metric-value"
                     style="font-size:1.4rem;">
                    {selected_model_name}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with output2:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">
                    Predicted Academic Risk
                </div>
                <div class="metric-value"
                     style="font-size:1.4rem;">
                    {risk_label}
                </div>
                <div class="metric-help">
                    Probability: {risk_probability * 100:.1f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# TAB 4 — RECOMMENDATIONS
# ============================================================

with tab4:

    st.subheader(
        "Personalized Study Recommendations"
    )

    st.caption(
        "Every recommendation is generated from the selected "
        "student's actual records."
    )

    recommendations = generate_recommendations(
        student_df
    )

    if not recommendations:

        st.markdown(
            """
            <div class="recommendation-card">
                <div class="recommendation-title">
                    ✅ No Specific Intervention Rule Triggered
                </div>

                <div class="recommendation-body">
                    None of the configured academic support
                    rules are currently triggered by the
                    student's available records.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        for recommendation in recommendations:

            st.markdown(
                f"""
                <div class="recommendation-card">
                    <div class="recommendation-title">
                        {recommendation["type"]}
                    </div>

                    <div class="recommendation-body">
                        {recommendation["text"]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # --------------------------------------------------------
    # ACTUAL ACADEMIC RECORDS
    # --------------------------------------------------------

    st.subheader(
        "Student Academic Record"
    )

    record = student_df[
        [
            "subject",
            "attendance",
            "assignment_avg",
            "quiz_avg",
            "midsem_marks",
            "comprehensive_marks",
            "previous_avg",
            "recent_avg",
            "performance_trend",
            "missed_assessments",
            "risk",
        ]
    ].copy()

    record["risk"] = (
        record["risk"]
        .map(RISK_LABELS)
    )

    record.columns = [
        "Subject",
        "Attendance (%)",
        "Assignment Average (%)",
        "Quiz Average (%)",
        "Midsem Marks",
        "Comprehensive Marks",
        "Previous Average (%)",
        "Recent Average (%)",
        "Performance Trend",
        "Missed Assessments",
        "Academic Risk",
    ]

    st.dataframe(
        record.style.format(
            {
                "Attendance (%)": "{:.1f}",
                "Assignment Average (%)": "{:.1f}",
                "Quiz Average (%)": "{:.1f}",
                "Midsem Marks": "{:.1f}",
                "Comprehensive Marks": "{:.1f}",
                "Previous Average (%)": "{:.1f}",
                "Recent Average (%)": "{:.1f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# TAB 5 — CONVERSATIONAL Q&A
# ============================================================

with tab5:

    st.subheader(
        "💬 Conversational Academic Q&A"
    )

    st.write(
        "Ask questions about the selected student's "
        "actual academic records."
    )

    question_options = [
        "Select a question...",
        "Which subject should I focus on?",
        "Why has my performance decreased?",
        "What is my attendance situation?",
        "What is my recent academic performance?",
        "Which subjects are declining?",
    ]

    selected_question = st.selectbox(
        "Suggested Question",
        question_options,
    )

    custom_question = st.text_input(
        "Or enter your own question",
        placeholder=(
            "Example: Which subjects need attention?"
        ),
    )

    if st.button(
        "Analyze Question"
    ):

        if custom_question.strip():

            question = custom_question.strip()

        elif selected_question != "Select a question...":

            question = selected_question

        else:

            st.warning(
                "Please select or enter a question."
            )

            question = ""

        if question:

            with st.chat_message("user"):
                st.write(question)

            answer = answer_question(
                question,
                student_df,
            )

            with st.chat_message("assistant"):

                st.markdown(
                    f"""
                    <div class="qa-answer">
                        {answer}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    f"""
    <div style="
        text-align:center;
        padding:12px;
        color:#94a3b8 !important;
    ">
        <strong style="color:#ffffff;">
            BITS Pilani — AI-Based Student Performance Advisor
        </strong>
        <br>
        <span style="color:#94a3b8;">
            Source: {DATA_PATH}
            &nbsp;|&nbsp;
            {total_students} unique students
            &nbsp;|&nbsp;
            {total_subject_records} subject records
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)
