# Titanic Survival Predictor (Streamlit)

An interactive web app that predicts a passenger's chance of survival on the Titanic
using a Random Forest model. Built as a learning project (not advice).

## Features
- Model saved and loaded with joblib
- Streamlit inputs: selectbox, radio, sliders
- Live prediction shown with `st.metric`
- Model cached with `@st.cache_resource`
- Friendly error messages (missing model, invalid input)

## Run locally
```bash
pip install -r requirements.txt
python train_model.py      # creates model.joblib
streamlit run app.py
```

## Deploy
Push this folder (including `model.joblib`) to GitHub, then deploy it on
Streamlit Community Cloud with `app.py` as the main file.

## Live app
<paste your deployed link here>
