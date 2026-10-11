import streamlit as st



st.set_page_config(
    page_title="AI Student Performance Advisor",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 AI-Based Student Performance Advisor")
st.write("Project Demo")

        col1, col2, col3 = st.columns(3)
        col1.metric("Attendance", "88%")
        col2.metric("Average Score", "82")
        col3.metric("Risk Status", "LOW")

st.success("Dashboard deployed successfully!")

