
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. FILE CONFIGURATION
# ==========================================

MODEL_FILE = "Model_heart.pkl"
SCALER_FILE = "Scaler_heart.pkl"
COLUMNS_FILE = "Columns_heart.pkl"


# ==========================================
# 3. LOAD AND VALIDATE ARTIFACTS
# ==========================================

@st.cache_resource
def load_artifacts():

    for filename in [MODEL_FILE, SCALER_FILE, COLUMNS_FILE]:
        if not Path(filename).exists():
            raise FileNotFoundError(
                f"Missing file: {filename}. "
                "Check that it exists in your project folder."
            )

    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    columns = joblib.load(COLUMNS_FILE)

    # Handle a model accidentally saved inside a list/tuple/dict.
    def find_estimator(obj):

        if hasattr(obj, "predict"):
            return obj

        if isinstance(obj, (list, tuple)):
            for item in obj:
                found = find_estimator(item)
                if found is not None:
                    return found

        if isinstance(obj, dict):
            for key in ["model", "estimator", "classifier"]:
                if key in obj:
                    found = find_estimator(obj[key])
                    if found is not None:
                        return found

        return None

    model = find_estimator(model)

    if model is None:
        raise TypeError(
            "The model file does not contain a trained estimator "
            "with a predict() method. Re-save your trained model."
        )

    if not isinstance(columns, (list, tuple)):
        raise TypeError(
            "Columns file must contain a list or tuple of feature names."
        )

    columns = list(columns)

    if not columns:
        raise ValueError("The saved feature column list is empty.")

    if not all(isinstance(col, str) for col in columns):
        raise TypeError("Feature names must all be strings.")

    if len(columns) != len(set(columns)):
        raise ValueError("Duplicate feature names found in Columns file.")

    if hasattr(scaler, "n_features_in_"):
        if scaler.n_features_in_ != len(columns):
            raise ValueError(
                f"Scaler expects {scaler.n_features_in_} features, "
                f"but Columns file contains {len(columns)}."
            )

    if hasattr(model, "n_features_in_"):
        if model.n_features_in_ != len(columns):
            raise ValueError(
                f"Model expects {model.n_features_in_} features, "
                f"but Columns file contains {len(columns)}."
            )

    return model, scaler, columns


try:
    model, scaler, columns = load_artifacts()
    artifacts_loaded = True

except Exception as e:
    artifacts_loaded = False
    load_error = str(e)


# ==========================================
# 4. CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.stApp {
    background-color: #0E1117;
    color: #FAFAFA;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}

.main-title {
    font-size: 38px;
    font-weight: 800;
    color: #FFFFFF;
    margin-bottom: 5px;
}

.subtitle {
    color: #A6AAB4;
    font-size: 16px;
    margin-bottom: 25px;
}

.section-title {
    font-size: 21px;
    font-weight: 700;
    color: #E5E7EB;
    margin-top: 20px;
    margin-bottom: 15px;
}

div.stButton > button {
    width: 100%;
    height: 52px;
    border-radius: 10px;
    background-color: #E63946;
    color: white;
    font-size: 17px;
    font-weight: 700;
    border: none;
}

div.stButton > button:hover {
    background-color: #C92D3B;
    color: white;
}

.result-card {
    padding: 25px;
    border-radius: 15px;
    background-color: #1B2330;
    border: 1px solid #303A49;
    margin-top: 20px;
}

.small-text {
    color: #A6AAB4;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# 5. SIDEBAR
# ==========================================

with st.sidebar:

    st.markdown("## ❤️ HeartGuard")

    st.caption("Machine Learning Risk Assessment")

    st.divider()

    st.markdown("### About this application")

    st.write(
        "This application uses a trained machine learning "
        "model to estimate heart disease risk from the "
        "provided health indicators."
    )

    st.divider()

    if artifacts_loaded:
        st.success("ML Model Loaded Successfully")
        st.success("Scaler Loaded Successfully")
        st.success("Feature Columns Loaded Successfully")
    else:
        st.error("Model initialization failed.")

    st.caption("Academic ML project | Prediction is not a diagnosis.")


# ==========================================
# 6. MAIN HEADER
# ==========================================

st.markdown(
    '<div class="main-title">Heart Disease Risk Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Enter the health indicators below to generate a model-based risk prediction.'
    '</div>',
    unsafe_allow_html=True
)

if not artifacts_loaded:

    st.error(f"Unable to initialize the application: {load_error}")

    st.info(
        "Check your model, scaler, and columns pickle files. "
        "The model filename must point to your actual trained estimator."
    )

    st.stop()


# ==========================================
# 7. INPUT FORM
# ==========================================

with st.form("heart_prediction_form"):

    st.markdown(
        '<div class="section-title">01. Personal Information</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider(
            "Age (years)",
            min_value=18,
            max_value=100,
            value=30
        )

    with col2:
        sex = st.selectbox(
            "Sex",
            ["M", "F"],
            format_func=lambda x: "Male" if x == "M" else "Female"
        )

    st.divider()

    st.markdown(
        '<div class="section-title">02. Cardiovascular Indicators</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        chest_pain = st.selectbox(
            "Chest Pain Type",
            ["ASY", "NAP", "ATA", "TA"],
            format_func=lambda x: {
                "ASY": "Asymptomatic",
                "NAP": "Non-Anginal Pain",
                "ATA": "Atypical Angina",
                "TA": "Typical Angina"
            }[x]
        )

        resting_bp = st.number_input(
            "Resting Blood Pressure (mm Hg)",
            min_value=80,
            max_value=200,
            value=120
        )

        cholesterol = st.number_input(
            "Cholesterol (mg/dl)",
            min_value=100,
            max_value=600,
            value=200
        )

        fasting_bs = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dl?",
            ["No", "Yes"]
        )

    with col2:

        resting_ecg = st.selectbox(
            "Resting ECG",
            ["Normal", "ST", "LVH"]
        )

        max_hr = st.number_input(
            "Maximum Heart Rate Achieved",
            min_value=60,
            max_value=220,
            value=150
        )

        exercise_angina = st.selectbox(
            "Exercise-Induced Angina",
            ["No", "Yes"]
        )

        old_peak = st.number_input(
            "Oldpeak (ST Depression)",
            min_value=0.0,
            max_value=6.0,
            value=1.0,
            step=0.1
        )

    st.divider()

    st.markdown(
        '<div class="section-title">03. Exercise ECG Indicator</div>',
        unsafe_allow_html=True
    )

    st_slope = st.selectbox(
        "Slope of the Peak Exercise ST Segment",
        ["Up", "Flat", "Down"]
    )

    st.markdown("<br>", unsafe_allow_html=True)

    submitted = st.form_submit_button(
        "Generate Prediction"
    )


# ==========================================
# 8. FEATURE ENGINEERING
# ==========================================

def create_input_dataframe():

    # Initialize all expected features with zero.
    data = {col: 0 for col in columns}

    # Numerical features.
    numerical_values = {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": 1 if fasting_bs == "Yes" else 0,
        "MaxHR": max_hr,
        "Oldpeak": old_peak
    }

    for feature, value in numerical_values.items():

        if feature in data:
            data[feature] = value

    # Categorical features.
    categorical_values = {
        "Sex": sex,
        "ChestPainType": chest_pain,
        "RestingECG": resting_ecg,
        "ExerciseAngina": "Y" if exercise_angina == "Yes" else "N",
        "ST_Slope": st_slope
    }

    prefix_options = {
        "Sex": ["Sex"],
        "ChestPainType": ["ChestPainType"],
        "RestingECG": ["RestingECG"],
        "ExerciseAngina": ["ExerciseAngina"],
        "ST_Slope": ["ST_Slope", "ST Slope"]
    }

    for feature, value in categorical_values.items():

        prefixes = prefix_options[feature]

        matching_columns = [
            col for col in columns
            if any(col.startswith(prefix + "_") for prefix in prefixes)
        ]

        if not matching_columns:
            raise ValueError(
                f"No one-hot encoded columns found for {feature}. "
                f"Check your training feature names."
            )

        # Reset this category's dummy columns.
        for col in matching_columns:
            data[col] = 0

        # Activate the selected category if its dummy column exists.
        possible_names = [
            f"{prefix}_{value}" for prefix in prefixes
        ]

        matched_name = next(
            (name for name in possible_names if name in data),
            None
        )

        # If the selected category has no column, it is treated
        # as the reference category (drop_first=True).
        if matched_name is not None:
            data[matched_name] = 1

    # Keep precisely the original training feature order.
    input_df = pd.DataFrame([data], columns=columns)

    return input_df


# ==========================================
# 9. PREDICTION
# ==========================================

if submitted:

    try:

        input_df = create_input_dataframe()

        # Check that all input values are numeric.
        input_df = input_df.astype(float)

        # Apply the same scaler used during training.
        input_scaled = scaler.transform(input_df)

        # Generate prediction.
        prediction = model.predict(input_scaled)[0]

        st.divider()

        st.markdown("## Prediction Results")

        if prediction == 1:

            st.error(
                "### Higher Risk Prediction\n"
                "The model predicts a higher likelihood of heart disease."
            )

        elif prediction == 0:

            st.success(
                "### Lower Risk Prediction\n"
                "The model predicts a lower likelihood of heart disease."
            )

        else:
            st.warning(
                f"The model returned an unexpected class: {prediction}"
            )

        # Display probability when supported.
        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(input_scaled)[0]

            classes = list(model.classes_)

            if 1 in classes:

                positive_index = classes.index(1)
                risk_probability = probabilities[positive_index]

                st.markdown("### Model Probability")

                st.progress(float(risk_probability))

                st.metric(
                    "Predicted probability for class 1",
                    f"{risk_probability * 100:.2f}%"
                )

        with st.expander("View processed model input"):

            st.dataframe(input_df, use_container_width=True)

    except Exception as e:

        st.error("Prediction could not be completed.")

        st.code(str(e))

        st.info(
            "Check that your training features, saved columns, "
            "scaler, and model all use the same encoding and order."
        )


# ==========================================
# 10. FOOTER
# ==========================================

st.divider()

st.markdown(
    '<div class="small-text">'
    'Disclaimer: This application is an educational machine learning '
    'project. It is not a medical diagnostic tool. Predictions should '
    'not replace professional medical advice or clinical evaluation.'
    '</div>',
    unsafe_allow_html=True
)
