
import streamlit as st
import pandas as pd
import joblib

# Load the saved model and label encoder.
@st.cache_resource
def load_artifacts():
    model = joblib.load("university_student_stress_model.pkl")
    label_encoder = joblib.load("stress_label_encoder.pkl")
    return model, label_encoder


model, label_encoder = load_artifacts()

st.set_page_config(
    page_title="Student Stress Prediction",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Student Stress Level Prediction")
st.write(
    "Enter the student's details to predict the stress category "
    "using the trained machine learning model."
)

st.info(
    "This is an educational machine learning project, "
    "not a medical or psychological assessment."
)

with st.form("student_form"):

    st.subheader("Academic Information")

    age = st.slider("Age", 19, 24, 21)
    study_hours = st.slider("Study Hours", 0, 9, 4)
    attendance = st.slider("Class Attendance (%)", 40, 99, 70)
    exam_frequency = st.slider("Exam Frequency", 1, 9, 5)
    assignment_load = st.slider("Assignment Load", 1, 9, 5)

    tuition = st.selectbox("Tuition", ["No", "Yes"])
    university_type = st.selectbox(
        "University Type",
        [
            "National University",
            "Private University",
            "Public University"
        ]
    )

    st.subheader("Lifestyle Information")

    sleep_hours = st.slider("Sleep Hours", 4, 9, 7)
    exercise = st.selectbox("Physical Exercise", ["No", "Yes"])
    social_media = st.slider("Social Media Use", 0, 7, 3)
    screen_time = st.slider("Screen Time", 1, 11, 6)

    st.subheader("Personal and Social Factors")

    gender = st.selectbox("Gender", ["Female", "Male"])

    income = st.selectbox(
        "Family Income Level",
        ["Low", "Medium", "High"]
    )

    peer_pressure = st.slider("Peer Pressure", 1, 9, 5)
    family_support = st.slider("Family Support", 1, 9, 5)
    anxiety = st.slider("Anxiety Level", 1, 9, 5)

    submitted = st.form_submit_button(
        "Predict Stress Level",
        type="primary",
        use_container_width=True
    )


if submitted:

    # Column names and types must match the training data.
    student_data = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Study_Hours": study_hours,
        "Class_Attendance": attendance,
        "Tuition": tuition,
        "Exam_Frequency": exam_frequency,
        "Assignment_Load": assignment_load,
        "Sleep_Hours": sleep_hours,
        "Physical_Exercise": exercise,
        "Social_Media_Use": social_media,
        "Screen_Time": screen_time,
        "Family_Income_Level": income,
        "Peer_Pressure": peer_pressure,
        "Family_Support": family_support,
        "Anxiety_Level": anxiety,
        "University_Type": university_type
    }])

    try:
        # Predict the encoded class and convert it to its label.
        prediction = model.predict(student_data)
        predicted_label = label_encoder.inverse_transform(
            prediction.astype(int)
        )[0]

        st.subheader("Prediction Result")
        st.success(f"Predicted Stress Level: {predicted_label}")

        # Display class probabilities if supported by the model.
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(student_data)[0]

            probability_df = pd.DataFrame({
                "Stress Level": label_encoder.classes_,
                "Probability (%)": probabilities * 100
            }).sort_values(
                "Probability (%)",
                ascending=False
            )

            st.subheader("Model Class Probabilities")
            st.bar_chart(
                probability_df.set_index("Stress Level")
            )

            display_df = probability_df.copy()
            display_df["Probability (%)"] = (
                display_df["Probability (%)"].round(2)
            )

            st.dataframe(display_df, hide_index=True)

        st.caption(
            "Probabilities represent the model's estimates, "
            "not a clinical measure of stress."
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")
