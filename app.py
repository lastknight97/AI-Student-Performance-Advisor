import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
 
st.set_page_config(page_title="AI Student Performance Advisor",layout="wide")
 
@st.cache_data
def load_data():
    try:
        subject_df=pd.read_excel("data/Subject_Analytics_Dataset_200_Students.xlsx")
        risk_df=pd.read_excel("data/ML_Risk_Dataset_200_Students.xlsx")
        return subject_df,risk_df
    except Exception:
        return None,None
 
subject_df,risk_df=load_data()
 
st.title("🎓 AI-Based Student Performance Advisor")
 
if subject_df is None:
    st.warning("Upload datasets into data folder to enable full functionality.")
    st.stop()
 
student_id=st.sidebar.selectbox("Student", sorted(subject_df.Student_ID.unique()))
student_subjects=subject_df[subject_df.Student_ID==student_id]
student_risk=risk_df[risk_df.Student_ID==student_id].iloc[0]
 
tabs=st.tabs(["Overview","Subject Analytics","Risk","Recommendations","Explainable AI"])
 
with tabs[0]:
    c1,c2,c3,c4=st.columns(4)
    c1.metric('Attendance',f"{student_risk['Attendance']}%")
    c2.metric('Trend',student_risk['Performance_Trend'])
    c3.metric('Study Hours',student_risk['Study_Hours_Per_Week'])
    c4.metric('Submission Rate',f"{student_risk['Submission_Rate']}%")
 
with tabs[1]:
    st.plotly_chart(px.bar(student_subjects,x='Subject',y='Final_Marks',color='Final_Marks'),use_container_width=True)
 
with tabs[2]:
    df=risk_df.copy()
    df['Performance_Trend']=df['Performance_Trend'].map({'Declining':0,'Stable':1,'Improving':2})
    X=df[['Attendance','Assignment_Avg','Quiz_Avg','Internal_Marks','Previous_Sem_Avg','Recent_Test_Avg','Missed_Assessments','Study_Hours_Per_Week','Submission_Rate','Class_Participation','Performance_Trend']]
    y=df['Academic_Risk']
    model=RandomForestClassifier(n_estimators=100,random_state=42)
    model.fit(X,y)
    row=df[df.Student_ID==student_id][X.columns]
    pred=model.predict(row)[0]
    prob=model.predict_proba(row)[0].max()*100
    st.metric('Predicted Risk','HIGH' if pred==1 else 'LOW')
    st.metric('Confidence',f'{prob:.1f}%')
 
with tabs[3]:
    weak=student_subjects.loc[student_subjects.Final_Marks.idxmin(),'Subject']
    rec=[]
    if student_risk['Attendance']<75: rec.append('Increase attendance above 80%')
    rec.append(f'Focus on {weak}')
    if student_risk['Missed_Assessments']>=3: rec.append('Complete pending assessments')
    for r in rec: st.info(r)
 
with tabs[4]:
    imp=pd.DataFrame({'Feature':X.columns,'Importance':model.feature_importances_})
    st.plotly_chart(px.bar(imp.sort_values('Importance'),x='Importance',y='Feature',orientation='h'),use_container_width=True)
