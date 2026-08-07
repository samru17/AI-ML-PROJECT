import streamlit as st
import pandas as pd
import pickle


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="centered"
)


# ==========================================
# LOAD MODEL
# ==========================================

with open("model.pkl", "rb") as file:
    model = pickle.load(file)


# ==========================================
# TITLE
# ==========================================

st.title("🎓 Student Performance Prediction System")

st.write(
    "Enter the student's information below to predict "
    "whether the student is likely to Pass or Fail."
)


st.divider()


# ==========================================
# USER INPUT
# ==========================================

study_hours = st.number_input(
    "Study Hours per Day",
    min_value=1,
    max_value=15,
    value=5
)


attendance = st.number_input(
    "Attendance Percentage",
    min_value=0,
    max_value=100,
    value=75
)


previous_marks = st.number_input(
    "Previous Exam Marks",
    min_value=0,
    max_value=100,
    value=60
)


assignment_score = st.number_input(
    "Assignment Score",
    min_value=0,
    max_value=100,
    value=60
)


sleep_hours = st.number_input(
    "Sleep Hours per Day",
    min_value=1,
    max_value=15,
    value=7
)


internet_access = st.selectbox(
    "Internet Access",
    ["Yes", "No"]
)


extracurricular = st.selectbox(
    "Extracurricular Activities",
    ["Yes", "No"]
)


# ==========================================
# PREDICTION
# ==========================================

if st.button("🔮 Predict Result"):

    # Create input DataFrame
    myinput = pd.DataFrame({
        "study_hours": [study_hours],
        "attendance": [attendance],
        "previous_marks": [previous_marks],
        "assignment_score": [assignment_score],
        "sleep_hours": [sleep_hours],
        "internet_access": [internet_access],
        "extracurricular": [extracurricular]
    })


    # Prediction
    result = model.predict(myinput)[0]


    st.divider()

    if result == "Pass":

        st.success("🎉 Prediction: Student is likely to PASS!")

    else:

        st.error("⚠️ Prediction: Student is likely to FAIL.")


    # Probability
    probability = model.predict_proba(myinput)[0]

    classes = model.classes_

    probability_df = pd.DataFrame({
        "Result": classes,
        "Probability": probability
    })

    st.subheader("Prediction Probability")

    st.dataframe(
        probability_df,
        use_container_width=True
    )
