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
# BACKGROUND PHOTO
# ==========================================

st.markdown("""
<style>

/* Full Background Image */
.stApp {
    background-image: url("https://images.unsplash.com/photo-1509062522246-3755977927d7?auto=format&fit=crop&w=1920&q=80");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}

/* Main Content Box */
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
    background-color: #2563eb;
    color: white;
    border: none;
}

.stButton > button:hover {
    background-color: #1d4ed8;
    color: white;
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


  # ==========================================
# DISPLAY RESULT
# ==========================================

st.subheader("🎯 Prediction Result")

# Check different possible PASS outputs
if str(result).lower() in ["pass", "true", "1"]:

    st.success("🎉 PASS — Student is likely to PASS!")

    # Celebration animation
    st.balloons()

    st.markdown("""
    <div style="
        background-color: #d1fae5;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-size: 22px;
        font-weight: bold;
        color: #065f46;
        margin-top: 15px;
    ">
        🎉 Congratulations! 🎓<br>
        Great performance! Keep studying and achieving your goals! 🚀
    </div>
    """, unsafe_allow_html=True)

else:

    st.error("⚠️ FAIL — Student is likely to FAIL.")

    st.markdown("""
    <div style="
        background-color: #fee2e2;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-size: 20px;
        font-weight: bold;
        color: #991b1b;
        margin-top: 15px;
    ">
        💪 Don't worry! Keep practicing and studying.<br>
        You can improve your performance! 📚
    </div>
    """, unsafe_allow_html=True)

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
