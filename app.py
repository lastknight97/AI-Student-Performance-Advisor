import os
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

warnings.filterwarnings("ignore")


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Student Learning Support & Performance Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# ACADEMIC THEME
# ============================================================

st.markdown(
    """
    <style>
        * {
            font-family: Arial, Helvetica, sans-serif;
        }

        html, body, [class*="css"] {
            color: #000000 !important;
        }

        .stApp {
            background-color: #f8fafc;
        }

        section[data-testid="stSidebar"] {
            background-color: #ffffff;
            border-right: 1px solid #cbd5e1;
        }

        section[data-testid="stSidebar"] * {
            color: #000000 !important;
        }

        h1, h2, h3, h4, h5, h6,
        p, label, span, div {
            color: #000000;
        }

        .dashboard-title {
            font-size: 2.1rem;
            font-weight: 800;
            color: #000000 !important;
            margin-bottom: 0.2rem;
        }

        .dashboard-subtitle {
            font-size: 1rem;
            color: #1e293b !important;
            margin-bottom: 1.5rem;
        }

        .academic-card {
            background: #ffffff;
            border: 1px solid #cbd5e1;
            border-radius: 12px;
            padding: 20px;
            min-height: 125px;
            box-shadow: 0 3px 10px rgba(15, 23, 42, 0.08);
            margin-bottom: 15px;
        }

        .kpi-title {
            color: #1e293b !important;
            font-size: 0.9rem;
            font-weight: 700;
            margin-bottom: 8px;
        }

        .kpi-value {
            color: #000000 !important;
            font-size: 2rem;
            font-weight: 800;
        }

        .kpi-description {
            color: #1e293b !important;
            font-size: 0.82rem;
            margin-top: 4px;
        }

        .status-good {
            display: inline-block;
            background: #dcfce7;
            border: 1px solid #166534;
            color: #000000 !important;
            border-radius: 999px;
            padding: 7px 13px;
            font-weight: 800;
        }

        .status-support {
            display: inline-block;
            background: #fef3c7;
            border: 1px solid #92400e;
            color: #000000 !important;
            border-radius: 999px;
            padding: 7px 13px;
            font-weight: 800;
        }

        .rank-card {
            background: #ffffff;
            border: 1px solid #cbd5e1;
            border-radius: 12px;
            padding: 18px;
            text-align: center;
            box-shadow: 0 3px 10px rgba(15, 23, 42, 0.08);
        }

        .rank-number {
            font-size: 1.7rem;
            font-weight: 900;
            color: #000000 !important;
        }

        .rank-name {
            font-size: 1.1rem;
            font-weight: 800;
            color: #000000 !important;
        }

        .rank-id {
            color: #1e293b !important;
            font-size: 0.85rem;
        }

        .rank-mark {
            font-size: 1.4rem;
            font-weight: 900;
            color: #000000 !important;
            margin-top: 8px;
        }

        .study-step {
            background: #ffffff;
            border-left: 5px solid #000000;
            border-top: 1px solid #cbd5e1;
            border-right: 1px solid #cbd5e1;
            border-bottom: 1px solid #cbd5e1;
            border-radius: 8px;
            padding: 13px 16px;
            margin: 8px 0;
            color: #000000 !important;
        }

        .section-heading {
            color: #000000 !important;
            font-size: 1.35rem;
            font-weight: 800;
            margin-top: 1rem;
            margin-bottom: 0.8rem;
        }

        [data-testid="stMetricValue"] {
            color: #000000 !important;
        }

        [data-testid="stMetricLabel"] {
            color: #000000 !important;
        }

        .stSelectbox label,
        .stRadio label {
            color: #000000 !important;
            font-weight: 700 !important;
        }

        .stDataFrame {
            border: 1px solid #cbd5e1;
        }

        footer {
            visibility: hidden;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATA LOADING
# ============================================================

DATA_PATH = "data/Subject_Analytics_Dataset_200_Students.xlsx"


@st.cache_data
def load_data(path):
    if not os.path.exists(path):
        return None

    raw = pd.read_excel(path)

    # Clean column names
    raw.columns = (
        raw.columns.astype(str)
        .str.strip()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )

    return raw


raw_df = load_data(DATA_PATH)

if raw_df is None:
    st.error(
        f"Dataset not found at `{DATA_PATH}`. "
        "Please place Subject_Analytics_Dataset_200_Students.xlsx "
        "inside the `data` folder."
    )
    st.stop()


# ============================================================
# COLUMN VALIDATION
# ============================================================

required_columns = [
    "Student_ID",
    "Final_Marks",
    "Midterm_Marks",
    "Attendance",
    "Quiz_Score",
]

missing_columns = [
    col for col in required_columns
    if col not in raw_df.columns
]

if missing_columns:
    st.error(
        "The dataset is missing the following required columns: "
        + ", ".join(missing_columns)
    )
    st.stop()


# ============================================================
# DATA CLEANING & STUDENT-LEVEL AGGREGATION
# ============================================================

numeric_columns = [
    "Final_Marks",
    "Midterm_Marks",
    "Attendance",
    "Quiz_Score",
]

# Assignment column can have slightly different names
assignment_candidates = [
    "Assignment",
    "Assignment_Score",
    "Assignment_Marks",
    "Assignment_Score_Marks",
]

assignment_column = next(
    (col for col in assignment_candidates if col in raw_df.columns),
    None,
)

if assignment_column:
    numeric_columns.append(assignment_column)

for col in numeric_columns:
    raw_df[col] = pd.to_numeric(raw_df[col], errors="coerce")


# Identify demographic/text fields
text_columns = raw_df.select_dtypes(include=["object"]).columns.tolist()

# Student_ID should remain the primary key, not a demographic field.
text_columns = [
    col for col in text_columns
    if col != "Student_ID"
]


# Mean for numeric academic measures.
# First non-null value for text/demographic fields.
aggregation = {}

for col in raw_df.columns:
    if col == "Student_ID":
        continue

    if col in numeric_columns:
        aggregation[col] = "mean"
    elif col in text_columns:
        aggregation[col] = "first"
    else:
        aggregation[col] = "first"


df = raw_df.groupby("Student_ID", as_index=False).agg(aggregation)


# ============================================================
# OPTIONAL COLUMN DISCOVERY
# ============================================================

def find_column(dataframe, candidates):
    for candidate in candidates:
        if candidate in dataframe.columns:
            return candidate
    return None


name_column = find_column(
    df,
    [
        "Student_Name",
        "Name",
        "StudentName",
        "Student",
    ],
)

department_column = find_column(
    df,
    [
        "Department",
        "Department_Name",
        "Program",
        "Course",
    ],
)

semester_column = find_column(
    df,
    [
        "Semester",
        "Semester_Name",
        "Term",
    ],
)


# Fallback values if optional fields are absent
if name_column is None:
    df["Student_Name_Display"] = df["Student_ID"].astype(str)
    name_column = "Student_Name_Display"

if department_column is None:
    df["Department_Display"] = "General"
    department_column = "Department_Display"

if semester_column is None:
    df["Semester_Display"] = "Not specified"
    semester_column = "Semester_Display"


# ============================================================
# FEATURE ENGINEERING
# ============================================================

df["performance_trend"] = np.where(
    df["Final_Marks"] >= df["Midterm_Marks"],
    "Improving/Stable",
    "Needs Attention",
)

df["missed_assessments"] = np.where(
    df["Attendance"] < 75,
    2,
    0,
)

df["Composite_Score"] = (
    (df["Attendance"] * 0.25)
    + (df["Final_Marks"] * 0.35)
    + (df["Midterm_Marks"] * 0.25)
    + (df["Quiz_Score"] * 0.15)
)

df["needs_support_target"] = (
    df["Composite_Score"] < 58
).astype(int)


# ============================================================
# GRADE MAPPING
# ============================================================

def calculate_grade(mark):
    if pd.isna(mark):
        return "F"
    if mark >= 80:
        return "A"
    if mark >= 70:
        return "B"
    if mark >= 60:
        return "C"
    if mark >= 50:
        return "D"
    return "F"


df["Grade"] = df["Final_Marks"].apply(calculate_grade)


# ============================================================
# MACHINE LEARNING MODEL
# ============================================================

feature_columns = [
    "Attendance",
    "Final_Marks",
    "Midterm_Marks",
    "Quiz_Score",
]

X = df[feature_columns].copy()
y = df["needs_support_target"].copy()

# Fill missing numeric values
X = X.fillna(X.median(numeric_only=True))

model = None
model_report = None
feature_importance = None

# A classification model requires at least two classes.
if y.nunique() >= 2 and len(df) >= 10:

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=250,
        random_state=42,
        class_weight="balanced",
        min_samples_leaf=2,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    model_report = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0,
    )

    feature_importance = pd.DataFrame(
        {
            "Indicator": feature_columns,
            "Importance": model.feature_importances_,
        }
    ).sort_values(
        "Importance",
        ascending=True,
    )

    # Probability of needing academic support
    if 1 in model.classes_:
        support_index = model.predict_proba(X)[:, list(model.classes_).index(1)]
    else:
        support_index = np.zeros(len(df))

else:
    # Safe fallback if the dataset contains only one target class.
    support_index = df["needs_support_target"].astype(float).values

df["Support_Need_Index"] = support_index * 100


# ============================================================
# LEARNING STATUS
# ============================================================

df["Learning_Status"] = np.where(
    df["needs_support_target"] == 1,
    "Learning Support Needed",
    "Good Academic Standing",
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def academic_status_badge(status):
    if status == "Good Academic Standing":
        return (
            '<span class="status-good">'
            "✅ Good Academic Standing"
            "</span>"
        )

    return (
        '<span class="status-support">'
        "⚠️ Learning Support Needed"
        "</span>"
    )


def create_kpi(title, value, description=""):
    st.markdown(
        f"""
        <div class="academic-card">
            <div class="kpi-title">{title}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-description">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def apply_chart_format(fig, title=None):
    if title:
        fig.update_layout(
            title={
                "text": title,
                "font": {
                    "color": "#000000",
                    "size": 18,
                },
            }
        )

    fig.update_layout(
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font={
            "color": "#000000",
            "size": 13,
        },
        margin=dict(l=50, r=30, t=60, b=50),
        legend=dict(
            font=dict(color="#000000"),
        ),
    )

    fig.update_xaxes(
        tickfont=dict(color="#000000"),
        title_font=dict(color="#000000"),
        showgrid=True,
        gridcolor="#cbd5e1",
        zerolinecolor="#000000",
    )

    fig.update_yaxes(
        tickfont=dict(color="#000000"),
        title_font=dict(color="#000000"),
        showgrid=True,
        gridcolor="#cbd5e1",
        zerolinecolor="#000000",
    )

    return fig


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    """
    <h1 style="color:#000000;">🎓 Academic Dashboard</h1>
    <p style="color:#000000;">
        Student Learning Support & Performance
    </p>
    """,
    unsafe_allow_html=True,
)

view = st.sidebar.radio(
    "Select Dashboard View",
    [
        "📊 Student Overview Dashboard",
        "👤 Student Profile Study",
        "🏫 Class Learning Progress",
        "🤖 AI Model Verification",
    ],
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    f"""
    <p style="color:#000000;">
        <b>Unique Students:</b> {df["Student_ID"].nunique()}
    </p>
    <p style="color:#000000;">
        <b>Student Records:</b> {len(raw_df)}
    </p>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# VIEW 1 — STUDENT OVERVIEW DASHBOARD
# ============================================================

if view == "📊 Student Overview Dashboard":

    st.markdown(
        '<div class="dashboard-title">📊 Student Overview Dashboard</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="dashboard-subtitle">'
        "A summary of student academic achievement, attendance, "
        "and learning support requirements."
        "</div>",
        unsafe_allow_html=True,
    )

    total_students = df["Student_ID"].nunique()

    average_mark = df["Final_Marks"].mean()

    pass_rate = (
        (df["Final_Marks"] >= 50).mean() * 100
    )

    support_count = int(
        df["needs_support_target"].sum()
    )

    # ---------------- KPI CARDS ----------------

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        create_kpi(
            "Total Enrolled Students",
            f"{total_students:,}",
            "Unique Student IDs",
        )

    with c2:
        create_kpi(
            "Average Student Mark",
            f"{average_mark:.1f} / 100",
            "Average final mark",
        )

    with c3:
        create_kpi(
            "Overall Pass Rate",
            f"{pass_rate:.1f}%",
            "Students scoring 50 or above",
        )

    with c4:
        create_kpi(
            "Students Needing Support",
            f"{support_count:,}",
            "Based on the academic support criterion",
        )

    # ---------------- TOP PERFORMERS ----------------

    st.markdown(
        '<div class="section-heading">🏆 Top Academic Performers</div>',
        unsafe_allow_html=True,
    )

    top_students = (
        df.sort_values("Final_Marks", ascending=False)
        .head(3)
    )

    rank_columns = st.columns(3)

    for rank, (_, student) in enumerate(
        top_students.iterrows(),
        start=1,
    ):
        with rank_columns[rank - 1]:
            st.markdown(
                f"""
                <div class="rank-card">
                    <div class="rank-number">Rank {rank}</div>
                    <div class="rank-name">
                        {student[name_column]}
                    </div>
                    <div class="rank-id">
                        Student ID: {student["Student_ID"]}
                    </div>
                    <div class="rank-mark">
                        {student["Final_Marks"]:.1f} / 100
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ---------------- CHARTS ----------------

    st.markdown(
        '<div class="section-heading">📈 Academic Performance Overview</div>',
        unsafe_allow_html=True,
    )

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:

        department_marks = (
            df.groupby(department_column)["Final_Marks"]
            .mean()
            .reset_index()
            .sort_values("Final_Marks", ascending=False)
        )

        fig_department = px.bar(
            department_marks,
            x=department_column,
            y="Final_Marks",
            text="Final_Marks",
            labels={
                department_column: "Department",
                "Final_Marks": "Average Final Mark",
            },
        )

        fig_department.update_traces(
            texttemplate="%{text:.1f}",
            textposition="outside",
        )

        fig_department = apply_chart_format(
            fig_department,
            "Average Marks by Department",
        )

        st.plotly_chart(
            fig_department,
            use_container_width=True,
        )

    with chart_col2:

        attendance_labels = [
            "Regular Attendance (≥75%)",
            "Low Attendance (<75%)",
        ]

        regular = int(
            (df["Attendance"] >= 75).sum()
        )

        low = int(
            (df["Attendance"] < 75).sum()
        )

        attendance_data = pd.DataFrame(
            {
                "Attendance Category": attendance_labels,
                "Students": [regular, low],
            }
        )

        fig_attendance = px.pie(
            attendance_data,
            names="Attendance Category",
            values="Students",
            hole=0.35,
        )

        fig_attendance.update_traces(
            textinfo="label+percent+value",
            textfont=dict(color="#000000"),
        )

        fig_attendance = apply_chart_format(
            fig_attendance,
            "Attendance Class Ratio",
        )

        st.plotly_chart(
            fig_attendance,
            use_container_width=True,
        )


# ============================================================
# VIEW 2 — STUDENT PROFILE STUDY
# ============================================================

elif view == "👤 Student Profile Study":

    st.markdown(
        '<div class="dashboard-title">👤 Student Profile Study</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="dashboard-subtitle">'
        "Individual student academic performance and personalized "
        "study improvement guidance."
        "</div>",
        unsafe_allow_html=True,
    )

    df["Student_Selector"] = (
        df["Student_ID"].astype(str)
        + " — "
        + df[name_column].astype(str)
    )

    selected_student = st.selectbox(
        "Select Student",
        sorted(df["Student_Selector"].tolist()),
    )

    student = df[
        df["Student_Selector"] == selected_student
    ].iloc[0]

    # ---------------- STUDENT SUMMARY ----------------

    st.markdown(
        '<div class="section-heading">Student Academic Summary</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        create_kpi(
            "Student",
            str(student[name_column]),
            f"ID: {student['Student_ID']}",
        )

    with c2:
        create_kpi(
            "Department & Semester",
            str(student[department_column]),
            f"Semester: {student[semester_column]}",
        )

    with c3:
        create_kpi(
            "Average Mark",
            f"{student['Final_Marks']:.1f} / 100",
            f"Grade: {student['Grade']}",
        )

    with c4:
        create_kpi(
            "Attendance",
            f"{student['Attendance']:.1f}%",
            f"Support index: {student['Support_Need_Index']:.1f}%",
        )

    st.markdown(
        academic_status_badge(
            student["Learning_Status"]
        ),
        unsafe_allow_html=True,
    )

    # ---------------- STUDY GUIDANCE ----------------

    st.markdown(
        '<div class="section-heading">'
        "📚 Personalized Study Improvement Steps"
        "</div>",
        unsafe_allow_html=True,
    )

    steps = []

    if student["Quiz_Score"] < 60:
        steps.append(
            "Review quiz topics where marks were below 60%, "
            "then complete short practice questions for each topic."
        )

    if student["Attendance"] < 75:
        steps.append(
            "Improve class attendance and review lessons missed "
            "during periods of low attendance."
        )

    if student["Midterm_Marks"] < 60:
        steps.append(
            "Revisit midterm topics and create a weekly revision "
            "plan focused on concepts with lower marks."
        )

    if student["Final_Marks"] < student["Midterm_Marks"]:
        steps.append(
            "Your final mark is below your midterm mark. "
            "Compare both assessments and revise topics where "
            "performance declined."
        )

    if assignment_column:
        if student[assignment_column] < 60:
            steps.append(
                "Complete additional assignment practice and "
                "review teacher feedback before the next submission."
            )

    if not steps:
        steps.append(
            "Maintain your current study routine and continue "
            "regular revision, practice, and class participation."
        )

    for i, step in enumerate(steps, start=1):
        st.markdown(
            f"""
            <div class="study-step">
                <b>Step {i}:</b> {step}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ---------------- PERFORMANCE CHART ----------------

    st.markdown(
        '<div class="section-heading">📊 Performance Bar Overview</div>',
        unsafe_allow_html=True,
    )

    performance_data = {
        "Assessment": [
            "Attendance",
            "Quiz Score",
            "Assignment",
            "Midterm Mark",
            "Final Mark",
        ],
        "Score": [
            student["Attendance"],
            student["Quiz_Score"],
            (
                student[assignment_column]
                if assignment_column
                else np.nan
            ),
            student["Midterm_Marks"],
            student["Final_Marks"],
        ],
    }

    performance_df = pd.DataFrame(performance_data)

    # Remove unavailable assignment values
    performance_df = performance_df.dropna(
        subset=["Score"]
    )

    fig_performance = px.bar(
        performance_df,
        x="Assessment",
        y="Score",
        text="Score",
        labels={
            "Assessment": "Academic Measure",
            "Score": "Score / Percentage",
        },
    )

    fig_performance.update_traces(
        texttemplate="%{text:.1f}",
        textposition="outside",
    )

    fig_performance.update_yaxes(
        range=[0, 105]
    )

    fig_performance = apply_chart_format(
        fig_performance,
        "Student Performance by Academic Measure",
    )

    st.plotly_chart(
        fig_performance,
        use_container_width=True,
    )

    # ---------------- RECORD TABLE ----------------

    st.markdown(
        '<div class="section-heading">📋 Student Record Summary</div>',
        unsafe_allow_html=True,
    )

    record_columns = [
        "Student_ID",
        name_column,
        department_column,
        semester_column,
        "Attendance",
        "Quiz_Score",
    ]

    if assignment_column:
        record_columns.append(assignment_column)

    record_columns.extend(
        [
            "Midterm_Marks",
            "Final_Marks",
            "Grade",
            "performance_trend",
            "missed_assessments",
            "Support_Need_Index",
            "Learning_Status",
        ]
    )

    record = student[record_columns].to_frame().T

    display_names = {
        "Student_ID": "Student ID",
        name_column: "Student Name",
        department_column: "Department",
        semester_column: "Semester",
        "Attendance": "Attendance %",
        "Quiz_Score": "Quiz Score",
        assignment_column: "Assignment Score",
        "Midterm_Marks": "Midterm Mark",
        "Final_Marks": "Final Mark",
        "performance_trend": "Performance Trend",
        "missed_assessments": "Missed Assessments",
        "Support_Need_Index": "Support Need Index",
        "Learning_Status": "Learning Status",
    }

    record = record.rename(
        columns={
            k: v
            for k, v in display_names.items()
            if k is not None and k in record.columns
        }
    )

    st.dataframe(
        record,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# VIEW 3 — CLASS LEARNING PROGRESS
# ============================================================

elif view == "🏫 Class Learning Progress":

    st.markdown(
        '<div class="dashboard-title">🏫 Class Learning Progress</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="dashboard-subtitle">'
        "Class-level academic progress, departmental comparison, "
        "and grade distribution."
        "</div>",
        unsafe_allow_html=True,
    )

    active_students = int(
        df["Student_ID"].nunique()
    )

    class_average_final = df["Final_Marks"].mean()

    class_average_attendance = df["Attendance"].mean()

    c1, c2, c3 = st.columns(3)

    with c1:
        create_kpi(
            "Active Student Count",
            f"{active_students:,}",
            "Unique students in the dataset",
        )

    with c2:
        create_kpi(
            "Class Average Final Mark",
            f"{class_average_final:.1f} / 100",
            "Average final assessment mark",
        )

    with c3:
        create_kpi(
            "Class Average Attendance",
            f"{class_average_attendance:.1f}%",
            "Average student attendance",
        )

    # ---------------- DEPARTMENT COMPARISON ----------------

    st.markdown(
        '<div class="section-heading">'
        "📈 Departmental Academic Comparison"
        "</div>",
        unsafe_allow_html=True,
    )

    department_comparison = (
        df.groupby(department_column)[
            ["Midterm_Marks", "Final_Marks"]
        ]
        .mean()
        .reset_index()
    )

    department_long = department_comparison.melt(
        id_vars=[department_column],
        value_vars=[
            "Midterm_Marks",
            "Final_Marks",
        ],
        var_name="Assessment",
        value_name="Average Mark",
    )

    department_long["Assessment"] = (
        department_long["Assessment"]
        .replace(
            {
                "Midterm_Marks": "Midterm Mark",
                "Final_Marks": "Final Mark",
            }
        )
    )

    fig_comparison = px.bar(
        department_long,
        x=department_column,
        y="Average Mark",
        color="Assessment",
        barmode="group",
        text="Average Mark",
        labels={
            department_column: "Department",
            "Average Mark": "Average Mark",
        },
    )

    fig_comparison.update_traces(
        texttemplate="%{text:.1f}",
        textposition="outside",
    )

    fig_comparison = apply_chart_format(
        fig_comparison,
        "Midterm vs Final Marks by Department",
    )

    st.plotly_chart(
        fig_comparison,
        use_container_width=True,
    )

    # ---------------- GRADE DISTRIBUTION ----------------

    st.markdown(
        '<div class="section-heading">📚 Grade Distribution</div>',
        unsafe_allow_html=True,
    )

    grade_order = ["A", "B", "C", "D", "F"]

    grade_counts = (
        df["Grade"]
        .value_counts()
        .reindex(grade_order, fill_value=0)
        .reset_index()
    )

    grade_counts.columns = [
        "Grade",
        "Students",
    ]

    fig_grades = px.bar(
        grade_counts,
        x="Grade",
        y="Students",
        text="Students",
        labels={
            "Grade": "Letter Grade",
            "Students": "Number of Students",
        },
    )

    fig_grades.update_traces(
        textposition="outside",
    )

    fig_grades = apply_chart_format(
        fig_grades,
        "Student Grade Distribution",
    )

    st.plotly_chart(
        fig_grades,
        use_container_width=True,
    )


# ============================================================
# VIEW 4 — AI MODEL VERIFICATION
# ============================================================

elif view == "🤖 AI Model Verification":

    st.markdown(
        '<div class="dashboard-title">🤖 AI Model Verification</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="dashboard-subtitle">'
        "Verification of the academic support prediction model "
        "using held-out student records."
        "</div>",
        unsafe_allow_html=True,
    )

    if model is None or model_report is None:

        st.warning(
            "The prediction model could not be trained because "
            "the dataset does not contain enough target variation."
        )

    else:

        # ---------------- CLASSIFICATION PERFORMANCE ----------------

        st.markdown(
            '<div class="section-heading">'
            "📋 Classification Performance"
            "</div>",
            unsafe_allow_html=True,
        )

        report_rows = []

        for label, display_name in [
            ("0", "Good Academic Standing"),
            ("1", "Learning Support Needed"),
        ]:
            if label in model_report:
                report_rows.append(
                    {
                        "Academic Outcome": display_name,
                        "Precision": model_report[label]["precision"],
                        "Recall": model_report[label]["recall"],
                        "F1-Score": model_report[label]["f1-score"],
                    }
                )

        metrics_df = pd.DataFrame(report_rows)

        for col in [
            "Precision",
            "Recall",
            "F1-Score",
        ]:
            if col in metrics_df.columns:
                metrics_df[col] = (
                    metrics_df[col] * 100
                ).round(1)

        st.dataframe(
            metrics_df,
            use_container_width=True,
            hide_index=True,
        )

        # ---------------- MODEL SUMMARY ----------------

        st.markdown(
            '<div class="section-heading">📘 Model Training Summary</div>',
            unsafe_allow_html=True,
        )

        summary_col1, summary_col2, summary_col3 = st.columns(3)

        with summary_col1:
            create_kpi(
                "Training Method",
                "Random Forest",
                "Supervised classification",
            )

        with summary_col2:
            create_kpi(
                "Training Split",
                "80 / 20",
                "80% training, 20% testing",
            )

        with summary_col3:
            create_kpi(
                "Support Criterion",
                "< 58",
                "Composite academic score",
            )

        # ---------------- FEATURE IMPORTANCE ----------------

        st.markdown(
            '<div class="section-heading">'
            "📊 Academic Indicator Importance"
            "</div>",
            unsafe_allow_html=True,
        )

        importance_display = feature_importance.copy()

        importance_display["Indicator"] = (
            importance_display["Indicator"]
            .replace(
                {
                    "Attendance": "Attendance",
                    "Final_Marks": "Final Marks",
                    "Midterm_Marks": "Midterm Marks",
                    "Quiz_Score": "Quiz Score",
                }
            )
        )

        fig_importance = px.bar(
            importance_display,
            x="Importance",
            y="Indicator",
            orientation="h",
            text="Importance",
            labels={
                "Importance": "Importance",
                "Indicator": "Academic Indicator",
            },
        )

        fig_importance.update_traces(
            texttemplate="%{text:.3f}",
            textposition="outside",
        )

        fig_importance = apply_chart_format(
            fig_importance,
            "Feature Importance Ranking",
        )

        st.plotly_chart(
            fig_importance,
            use_container_width=True,
        )

        st.info(
            "The Support Need Index is a model-generated probability "
            "that a student belongs to the Learning Support Needed "
            "category. It should be used as an educational support "
            "indicator alongside teacher review and student context."
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; color:#000000;">
        <b>Student Learning Support & Performance Dashboard</b><br>
        Academic performance information should support teaching,
        learning, revision, and student guidance.
    </div>
    """,
    unsafe_allow_html=True,
)
