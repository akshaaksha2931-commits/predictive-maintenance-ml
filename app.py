import streamlit as st
import pandas as pd
import joblib

# ---------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------

st.set_page_config(
    page_title="Predictive Maintenance System",
    page_icon="⚙️",
    layout="wide"
)

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------

model = joblib.load("models/deployment_model.pkl")

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------

st.title("⚙️ Predictive Maintenance System")
st.markdown(
    "### Machine Failure Risk Prediction using Machine Learning"
)

st.write(
    "This system uses machine sensor parameters to estimate "
    "the probability of machine failure and support preventive maintenance."
)

st.divider()

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.header("About the Model")

st.sidebar.write(
    """
    **Algorithm:** Random Forest Classifier
    
    **Purpose:** Predict machine failure risk
    
    **Input:** Machine sensor parameters
    
    **Output:** Failure probability
    
    **Decision Threshold:** 45%
    """
)

st.sidebar.info(
    "The model was trained using historical machine sensor data."
)

# ---------------------------------------------------
# INPUT SECTION
# ---------------------------------------------------

st.header("🔧 Machine Parameters")

col1, col2, col3 = st.columns(3)

with col1:
    machine_type = st.selectbox(
        "Machine Type",
        ["H", "L", "M"]
    )

with col2:
    air_temp = st.number_input(
        "Air Temperature [K]",
        min_value=250.0,
        max_value=350.0,
        value=300.0,
        step=0.1
    )

with col3:
    process_temp = st.number_input(
        "Process Temperature [K]",
        min_value=250.0,
        max_value=350.0,
        value=310.0,
        step=0.1
    )

col4, col5, col6 = st.columns(3)

with col4:
    rotational_speed = st.number_input(
        "Rotational Speed [rpm]",
        min_value=500,
        max_value=3000,
        value=1500,
        step=10
    )

with col5:
    torque = st.number_input(
        "Torque [Nm]",
        min_value=0.0,
        max_value=100.0,
        value=40.0,
        step=0.5
    )

with col6:
    tool_wear = st.number_input(
        "Tool Wear [min]",
        min_value=0,
        max_value=300,
        value=100,
        step=1
    )

st.divider()

# ---------------------------------------------------
# PREDICTION
# ---------------------------------------------------

if st.button(
    "🔍 Predict Machine Failure",
    use_container_width=True
):

    # Create input dataframe
    data = pd.DataFrame([{
        "Type": machine_type,
        "Air temperature [K]": air_temp,
        "Process temperature [K]": process_temp,
        "Rotational speed [rpm]": rotational_speed,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear
    }])

    # Encode machine type
    data = pd.get_dummies(
        data,
        columns=["Type"],
        drop_first=True
    )

    # Match training columns
    expected_columns = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
        "Type_L",
        "Type_M"
    ]

    data = data.reindex(
        columns=expected_columns,
        fill_value=0
    )

    # Predict probability
    probability = model.predict_proba(data)[0][1]

    # Decision threshold
    threshold = 0.45

    prediction = 1 if probability >= threshold else 0

    probability_percentage = probability * 100

    # ------------------------------------------------
    # RESULTS
    # ------------------------------------------------

    st.divider()
    st.header("📊 Prediction Result")

    result_col1, result_col2 = st.columns(2)

    with result_col1:

        if prediction == 1:
            st.error("⚠️ MACHINE FAILURE RISK DETECTED")
        else:
            st.success("✅ NO MACHINE FAILURE RISK DETECTED")

    with result_col2:

        st.metric(
            "Predicted Failure Probability",
            f"{probability_percentage:.2f}%"
        )

    # Probability bar
    st.subheader("Failure Risk Level")

    st.progress(min(probability, 1.0))

    # ------------------------------------------------
    # MAINTENANCE RECOMMENDATION
    # ------------------------------------------------

    st.subheader("🛠️ Maintenance Recommendation")

    if prediction == 1:

        st.warning(
            """
            The machine shows an elevated predicted risk of failure.

            Recommended actions:
            - Inspect the machine condition
            - Check tool wear
            - Inspect torque and rotational speed
            - Perform preventive maintenance if required
            """
        )

    else:

        st.info(
            """
            The machine currently shows a lower predicted risk
            of failure.

            Continue routine monitoring and preventive maintenance.
            """
        )

    # ------------------------------------------------
    # INPUT SUMMARY
    # ------------------------------------------------

    st.subheader("📋 Sensor Input Summary")

    summary = pd.DataFrame({
        "Parameter": [
            "Machine Type",
            "Air Temperature",
            "Process Temperature",
            "Rotational Speed",
            "Torque",
            "Tool Wear"
        ],
        "Value": [
            machine_type,
            f"{air_temp:.1f} K",
            f"{process_temp:.1f} K",
            f"{rotational_speed} rpm",
            f"{torque:.1f} Nm",
            f"{tool_wear} min"
        ]
    })

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.divider()
# ---------------------------------------------------
# FEATURE IMPORTANCE
# ---------------------------------------------------

st.divider()

st.header("📈 Model Feature Importance")

st.write(
    "The chart below shows the relative importance of each "
    "sensor parameter used by the Random Forest model."
)

feature_names = [
    "Air Temperature",
    "Process Temperature",
    "Rotational Speed",
    "Torque",
    "Tool Wear",
    "Machine Type L",
    "Machine Type M"
]

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    "Importance",
    ascending=True
)

st.bar_chart(
    importance_df.set_index("Feature")
)
st.caption(
    "Predictive Maintenance System | Random Forest Machine Learning Model"
)