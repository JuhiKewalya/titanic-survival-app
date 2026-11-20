import joblib
import pandas as pd
import streamlit as st

# Must match the order used in train_model.py
FEATURES = ["pclass", "sex", "age", "fare"]

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢")


@st.cache_resource
def load_model():
    """Load the model once; Streamlit keeps it in memory between clicks."""
    return joblib.load("model.joblib")


st.title("🚢 Would this passenger have survived?")
st.write("An educational model trained on the Titanic passenger list.")

# ---- Inputs ----
col1, col2 = st.columns(2)
with col1:
    pclass = st.selectbox("Ticket class", [1, 2, 3], index=2)
    sex = st.radio("Sex", ["female", "male"], horizontal=True)
with col2:
    age = st.slider("Age", 1, 80, 30)
    fare = st.slider("Fare paid (£)", 0, 500, 30)

# ---- Prediction ----
if st.button("Predict", type="primary"):
    try:
        model = load_model()
    except FileNotFoundError:
        st.error("The model file is missing. Run train_model.py first to create model.joblib.")
        st.stop()
    except Exception:
        st.error("The model could not be loaded. Please retrain it and try again.")
        st.stop()

    # Basic input validation (sliders already limit values, this is a safety net)
    if not (1 <= age <= 80) or not (0 <= fare <= 500):
        st.warning("Please choose an age between 1-80 and a fare between 0-500.")
        st.stop()

    row = pd.DataFrame(
        [[pclass, 1 if sex == "female" else 0, age, fare]],
        columns=FEATURES,
    )
    chance = model.predict_proba(row)[0][1]

    st.metric("Estimated chance of survival", f"{chance:.0%}")
    st.progress(float(chance))
    st.caption(
        "A learning exercise. A Random Forest model trained on the historical "
        "Titanic dataset. It is not advice."
    )
