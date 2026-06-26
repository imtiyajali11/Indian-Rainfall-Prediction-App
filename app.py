from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st


ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = ARTIFACT_DIR / "rainfall_model.joblib"
METADATA_PATH = ARTIFACT_DIR / "metadata.json"
RESULTS_PATH = ARTIFACT_DIR / "model_results.csv"


st.set_page_config(
    page_title="Indian Rainfall Prediction",
    page_icon="🌧️",
    layout="wide"
)


st.title("🌧️ Indian Rainfall Prediction App")
st.write("Predict monthly rainfall using State, District and Month.")


if not MODEL_PATH.exists():
    st.error("Model file not found. Please run train_model.py first.")
    st.code("python3 train_model.py")
    st.stop()

if not METADATA_PATH.exists():
    st.error("Metadata file not found. Please run train_model.py first.")
    st.code("python3 train_model.py")
    st.stop()


model = joblib.load(MODEL_PATH)

with open(METADATA_PATH, "r", encoding="utf-8") as file:
    metadata = json.load(file)


def month_name(month_number):
    months = {
        1: "January",
        2: "February",
        3: "March",
        4: "April",
        5: "May",
        6: "June",
        7: "July",
        8: "August",
        9: "September",
        10: "October",
        11: "November",
        12: "December"
    }
    return months.get(month_number, str(month_number))


states = metadata["states"]
districts_by_state = metadata["districts_by_state"]

col1, col2, col3 = st.columns(3)

with col1:
    selected_state = st.selectbox("Select State", states)

with col2:
    district_options = districts_by_state.get(selected_state, [])
    selected_district = st.selectbox("Select District", district_options)

with col3:
    selected_month = st.selectbox(
        "Select Month",
        list(range(1, 13)),
        format_func=month_name
    )


input_data = pd.DataFrame({
    "state": [selected_state],
    "district": [selected_district],
    "month": [selected_month]
})


st.subheader("Input Preview")
st.dataframe(input_data, use_container_width=True)


if st.button("Predict Rainfall"):
    prediction = model.predict(input_data)[0]

    st.success("Prediction completed successfully.")
    st.metric(
        label="Predicted Monthly Rainfall",
        value=f"{prediction:.2f}"
    )

    st.write(
        f"Predicted rainfall for **{selected_district}, {selected_state}** "
        f"in **{month_name(selected_month)}** is **{prediction:.2f}**."
    )


if RESULTS_PATH.exists():
    st.subheader("Model Performance")

    results_df = pd.read_csv(RESULTS_PATH)
    st.dataframe(results_df, use_container_width=True)

    st.write("Best model used:", metadata["best_model"])