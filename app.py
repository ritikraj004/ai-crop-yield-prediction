import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# If model doesn't exist, train it
if not os.path.exists("model.pkl"):
    import train_model  # this will generate model.pkl

model = pickle.load(open("model.pkl", "rb"))

# -----------------------------
# Title and Description
# -----------------------------

st.title("AI-Based Crop Yield Prediction & Analytics Tool")

st.write(
"This tool predicts crop yield using rainfall, fertilizer usage, pesticide usage, "
"and regional agricultural parameters. It also provides analytics and insights "
"from historical agricultural datasets."
)

st.caption(
"Data Source: Kaggle agricultural dataset. "
"The dataset is publicly available and was created by third-party contributors. "
"The developers of this application do not claim ownership of the data and are "
"not responsible for its accuracy or completeness."
)

# -----------------------------
# Load Dataset
# -----------------------------

data = pd.read_csv("dataset.csv")

# -----------------------------
# Load Model and Encoders
# -----------------------------

model = pickle.load(open("model.pkl", "rb"))
crop_encoder = pickle.load(open("crop_encoder.pkl", "rb"))
state_encoder = pickle.load(open("state_encoder.pkl", "rb"))
season_encoder = pickle.load(open("season_encoder.pkl", "rb"))

# -----------------------------
# Input Panel
# -----------------------------

st.header("Farm Input Parameters")

col1, col2 = st.columns(2)

with col1:
    crop = st.selectbox("Select Crop", sorted(data["Crop"].unique()))
    state = st.selectbox("Select State", sorted(data["State"].unique()))
    season = st.selectbox("Select Season", sorted(data["Season"].unique()))

with col2:
    rainfall = st.slider("Rainfall (mm)", 0, 4000, 1200)
    fertilizer = st.slider("Fertilizer ", 0, 1000000, 500000)
    pesticide = st.slider("Pesticide ", 0, 50000, 10000)

# -----------------------------
# Prediction
# -----------------------------

if st.button("Predict Yield"):

    crop_encoded = crop_encoder.transform([crop])[0]
    state_encoded = state_encoder.transform([state])[0]
    season_encoded = season_encoder.transform([season])[0]

    features = np.array([[
        crop_encoded,
        state_encoded,
        season_encoded,
        rainfall,
        fertilizer,
        pesticide
    ]])

    prediction = model.predict(features)[0]

    st.subheader("Prediction Result")
    st.success(f"Predicted Yield for {crop}: {prediction:.2f} ")

    # -----------------------------
    # Recommendation Logic
    # -----------------------------

    # 

# -----------------------------
# Dataset Statistics
# -----------------------------

st.header("Dataset Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Records", len(data))

with col2:
    st.metric("States Covered", data["State"].nunique())

with col3:
    st.metric("Crops Covered", data["Crop"].nunique())

# -----------------------------
# Analytics Dashboard
# -----------------------------

st.header("Agricultural Insights Dashboard")

crop_data = data[data["Crop"] == crop]

# Rainfall vs Yield
st.subheader("Rainfall vs Yield")

fig1, ax1 = plt.subplots()

ax1.scatter(crop_data["Annual_Rainfall"], crop_data["Yield"])
ax1.set_xlabel("Rainfall (mm)")
ax1.set_ylabel("Yield ")
ax1.set_title(f"Rainfall vs Yield for {crop}")

st.pyplot(fig1)

# Fertilizer vs Yield
st.subheader("Fertilizer vs Yield")

fig2, ax2 = plt.subplots()

ax2.scatter(crop_data["Fertilizer"], crop_data["Yield"])
ax2.set_xlabel("Fertilizer Usage")
ax2.set_ylabel("Yield ")
ax2.set_title(f"Fertilizer vs Yield for {crop}")

st.pyplot(fig2)

# State-wise Yield
st.subheader("State-wise Average Yield")

state_yield = crop_data.groupby("State")["Yield"].mean().sort_values()

fig3, ax3 = plt.subplots()

state_yield.plot(kind="barh", ax=ax3)

ax3.set_xlabel("Average Yield ")
ax3.set_ylabel("State")
ax3.set_title(f"Average {crop} Yield by State")

st.pyplot(fig3)

# -----------------------------
# Footer
# -----------------------------

st.markdown("---")

st.write("Developed by Ritik Raj | B.Sc. Agriculture | AgriTech Machine Learning Project")
