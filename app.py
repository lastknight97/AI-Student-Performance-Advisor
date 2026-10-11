import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go



# --------------------------------------------------

# PAGE CONFIG

# --------------------------------------------------



st.set_page_config(

    page_title="AI Student Performance Advisor",

    page_icon="🎓",

    layout="wide"

)



# --------------------------------------------------

# CUSTOM STYLING

# --------------------------------------------------



st.markdown("""

<style>

.main {

    background-color: #f8fafc;

}

h1,h2,h3 {

    color: #1e3a8a;

}

[data-testid="stMetric"] {

    background-color: white;

    padding: 15px;

    border-radius: 12px;

    border: 1px solid #e2e8f0;

}

</style>

""", unsafe_allow_html=True)



# --------------------------------------------------

# SAMPLE DATA

# --------------------------------------------------



student_data = {

    "Student Name": "Aarav Sharma",

    "Student ID": "S001",

    "Department": "Computer Science",

    "Semester": 5,

    "Attendance": 88,

    "Average": 82,

    "Trend": "Improving",

    "Risk": "Low",

    "Study Hours": 12,

    "Submission Rate": 96

}



subject_df = pd.DataFrame({

    "Subject": [

        "Programming",

        "Mathematics",

        "Physics",

        "Chemistry",

        "Database Systems"

    ],

    "Marks": [92, 84, 74, 69, 88],

    "Attendance": [95, 88, 76, 72, 91]

})



trend_df = pd.DataFrame({

    "Assessment": [

        "Quiz 1",

        "Quiz 2",

        "Assignment",

        "Midterm",

        "Final"

    ],

    "Score": [68, 73, 80, 84, 89]

})



feature_df = pd.DataFrame({

    "Feature": [

        "Attendance",

        "Quiz Scores",

        "Assignments",

        "Recent Performance",

        "Missed Assessments"

    ],

    "Importance": [38, 20, 18, 14, 10]

})



# --------------------------------------------------

# SIDEBAR

# --------------------------------------------------



st.sidebar.title("Navigation")



page = st.sidebar.radio(

    "Choose Page",

    [

        "Executive Overview",

        "Subject Analytics",

        "Performance Trends",

        "Academic Risk",

        "Strengths & Weaknesses",

        "Recommendations",

        "AI Advisor",

        "Explainable AI"

    ]

)



# --------------------------------------------------

# HEADER

# --------------------------------------------------



st.title("🎓 AI-Based Student Performance Advisor")

st.caption(

    "Educational Analytics • Academic Risk Assessment • Personalized Recommendations"

)



# --------------------------------------------------

# EXECUTIVE OVERVIEW

# --------------------------------------------------



if page == "Executive Overview":



    st.subheader("Executive Overview")



    st.info(

        f"""

Student Name: {student_data['Student Name']}



Student ID: {student_data['Student ID']}



Department: {student_data['Department']}



Semester: {student_data['Semester']}

"""

    )



    c1, c2, c3 = st.columns(3)



    c1.metric("Attendance", f"{student_data['Attendance']}%")

    c2.metric("Average Score", student_data["Average"])

    c3.metric("Risk Status", student_data["Risk"])



    c4, c5, c6 = st.columns(3)



    c4.metric("Performance Trend", student_data["Trend"])

    c5.metric("Study Hours", student_data["Study Hours"])

    c6.metric(

        "Submission Rate",

        f"{student_data['Submission Rate']}%"

    )



# --------------------------------------------------

# SUBJECT ANALYTICS

# --------------------------------------------------



elif page == "Subject Analytics":



    st.subheader("📚 Subject Analytics")



    fig1 = px.bar(

        subject_df,

        x="Subject",

        y="Marks",

        color="Marks",

        title="Subject-wise Performance"

    )



    st.plotly_chart(fig1, use_container_width=True)



    radar = go.Figure()



    radar.add_trace(

        go.Scatterpolar(

            r=subject_df["Marks"],

            theta=subject_df["Subject"],

            fill="toself"

        )

    )



    radar.update_layout(

        polar=dict(

            radialaxis=dict(

                visible=True,

                range=[0, 100]

            )

        ),

        showlegend=False

    )



    st.plotly_chart(

        radar,

        use_container_width=True

    )



    col1, col2 = st.columns(2)



    col1.success("🏆 Strongest Subject: Programming")

    col2.error("⚠ Weakest Subject: Chemistry")



# --------------------------------------------------

# PERFORMANCE TRENDS

# --------------------------------------------------



elif page == "Performance Trends":



    st.subheader("📈 Performance Trends")



    c1, c2, c3 = st.columns(3)



    c1.metric("Previous Average", "72")

    c2.metric("Current Average", "82")

    c3.metric("Improvement", "+10")



    fig2 = px.line(

        trend_df,

        x="Assessment",

        y="Score",

        markers=True

    )



    st.plotly_chart(fig2, use_container_width=True)



    fig3 = px.area(

        trend_df,

        x="Assessment",

        y="Score"

    )



    st.plotly_chart(fig3, use_container_width=True)



# --------------------------------------------------

# ACADEMIC RISK

# --------------------------------------------------



elif page == "Academic Risk":



    st.subheader("⚠ Academic Risk Analysis")



    risk_gauge = go.Figure(

        go.Indicator(

            mode="gauge+number",

            value=25,

            title={"text": "Risk Score"},

            gauge={

                "axis": {"range": [0, 100]}

            }

        )

    )



    st.plotly_chart(

        risk_gauge,

        use_container_width=True

    )



    st.success("Current Status: LOW RISK")



    st.markdown("""

### Contributing Factors



✅ High Attendance



✅ Good Submission Rate



⚠ Physics Performance



⚠ Chemistry Performance

""")



# --------------------------------------------------

# STRENGTHS & WEAKNESSES

# --------------------------------------------------



elif page == "Strengths & Weaknesses":



    st.subheader("💪 Strengths & Weaknesses")



    col1, col2 = st.columns(2)



    with col1:

        st.success("""

✅ Programming



✅ Mathematics



✅ Assignment Completion



✅ Class Participation

""")



    with col2:

        st.error("""

⚠ Physics



⚠ Chemistry



⚠ Exam Preparation

""")



# --------------------------------------------------

# RECOMMENDATIONS

# --------------------------------------------------



elif page == "Recommendations":



    st.subheader("💡 Personalized Recommendations")



    recommendations = [

        "Increase attendance above 85%",

        "Improve Physics performance",

        "Revise Chemistry weekly",

        "Maintain strong Programming performance",

        "Add 3 extra study hours each week"

    ]



    for item in recommendations:

        st.info(item)



# --------------------------------------------------

# AI ADVISOR

# --------------------------------------------------



elif page == "AI Advisor":



    st.subheader("🤖 AI Academic Advisor")



    st.chat_message("user").write(

        "Which subject should I focus on?"

    )



    st.chat_message("assistant").write(

        """

Physics requires additional attention because

it currently has the lowest marks among all subjects.



Suggested Actions:



• Weekly revision



• Mock quizzes



• Practice assignments



• Additional study hours

"""

    )



# --------------------------------------------------

# EXPLAINABLE AI

# --------------------------------------------------



elif page == "Explainable AI":



    st.subheader("🔍 Explainable AI")



    fig4 = px.bar(

        feature_df,

        x="Importance",

        y="Feature",

        orientation="h",

        color="Importance",

        title="Feature Importance"

    )



    st.plotly_chart(

        fig4,

        use_container_width=True

    )



    fig5 = px.pie(

        feature_df,

        values="Importance",

        names="Feature",

        hole=0.5

    )



    st.plotly_chart(

        fig5,

        use_container_width=True

    )
