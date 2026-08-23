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
# CUSTOM CSS
# ==========================================

# ==========================================
# BACKGROUND PHOTO
# ==========================================
# ==========================================
# BACKGROUND PHOTO
# ==========================================

st.markdown("""
<style>

.stApp {
    background-image: url("student.jpg");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* Main content box */
.block-container {
    background-color: rgba(255, 255, 255, 0.92);
    padding: 2rem;
    border-radius: 20px;
    margin-top: 30px;
    margin-bottom: 30px;
}

/* Title */
h1 {
    text-align: center;
    color: #1f2937;
}

/* Button */
.stButton > button {
    width: 100%;
    height: 50px;
    border-radius: 10px;
    font-size: 18px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True) 


# ==========================================
# LOAD MODEL
# ==========================================

try:

    with open("model.pkl", "rb") as file:
        model = pickle.load(file)

except FileNotFoundError:

    st.error("❌ model.pkl file not found.")
    st.stop()

except Exception as e:

    st.error("❌ Error while loading the model.")
    st.write(e)
    st.stop()


# ==========================================
# TITLE
# ==========================================

st.title("🎓 Student Performance Prediction System")

st.write(
    "Enter the student's information below to predict "
    "whether the student is likely to **Pass or Fail**."
)

st.divider()


# ==========================================
# USER INPUT
# ==========================================

st.subheader("📋 Student Information")


col1, col2 = st.columns(2)


with col1:

    study_hours = st.number_input(
        "📚 Study Hours per Day",
        min_value=1.0,
        max_value=15.0,
        value=5.0,
        step=0.5
    )

    attendance = st.number_input(
        "📅 Attendance Percentage",
        min_value=0.0,
        max_value=100.0,
        value=75.0,
        step=1.0
    )

    previous_marks = st.number_input(
        "📝 Previous Exam Marks",
        min_value=0.0,
        max_value=100.0,
        value=60.0,
        step=1.0
    )

    assignment_score = st.number_input(
        "📖 Assignment Score",
        min_value=0.0,
        max_value=100.0,
        value=60.0,
        step=1.0
    )


with col2:

    sleep_hours = st.number_input(
        "😴 Sleep Hours per Day",
        min_value=1.0,
        max_value=15.0,
        value=7.0,
        step=0.5
    )

    internet_access = st.selectbox(
        "🌐 Internet Access",
        ["Yes", "No"]
    )

    extracurricular = st.selectbox(
        "🏃 Extracurricular Activities",
        ["Yes", "No"]
    )


st.divider()


# ==========================================
# PREDICTION BUTTON
# ==========================================

if st.button("🔮 Predict Student Result"):

    # ======================================
    # CREATE INPUT DATAFRAME
    # ======================================

    myinput = pd.DataFrame({

        "study_hours": [study_hours],

        "attendance": [attendance],

        "previous_marks": [previous_marks],

        "assignment_score": [assignment_score],

        "sleep_hours": [sleep_hours],

        "internet_access": [internet_access],

        "extracurricular": [extracurricular]

    })


    # ======================================
    # SHOW INPUT DATA
    # ======================================

    with st.expander("🔍 View Student Input"):

        st.dataframe(
            myinput,
            use_container_width=True
        )


    # ======================================
    # MAKE PREDICTION
    # ======================================

    try:

        result = model.predict(myinput)[0]

    except Exception as e:

        st.error("❌ Prediction error occurred.")
        st.write(e)
        st.stop()


    st.divider()


    # ======================================
    # DISPLAY RESULT
    # ======================================

    st.subheader("🎯 Prediction Result")


    if str(result).lower() == "pass":

        st.success(
            "🎉 PASS — Student is likely to PASS!"
        )

    else:

        st.error(
            "⚠️ FAIL — Student is likely to FAIL."
        )


    # ======================================
    # PREDICTION PROBABILITY
    # ======================================

    if hasattr(model, "predict_proba"):

        try:

            probability = model.predict_proba(myinput)[0]

            classes = model.classes_

            probability_df = pd.DataFrame({

                "Result": classes,

                "Probability": probability

            })


            # Convert probability to percentage

            probability_df["Probability (%)"] = (
                probability_df["Probability"] * 100
            ).round(2)


            probability_df = probability_df.drop(
                columns=["Probability"]
            )


            st.subheader("📊 Prediction Probability")


            st.dataframe(
                probability_df,
                use_container_width=True,
                hide_index=True
            )


            # ==================================
            # PROGRESS BAR
            # ==================================

            for i, row in probability_df.iterrows():

                st.write(
                    f"**{row['Result']} : "
                    f"{row['Probability (%)']}%**"
                )

                st.progress(
                    int(row["Probability (%)"])
                )


        except Exception as e:

            st.warning(
                "⚠️ Probability could not be displayed."
            )

            st.write(e)


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "🎓 Student Performance Prediction System | "
    "Machine Learning Project"
)
