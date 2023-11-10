import streamlit as st
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from mapie.regression import MapieRegressor
from sklearn.datasets import make_s_curve

# Function to add homoskedastic noise to the ground truth
def add_homoskedastic_noise(y, noise_level):
    return y + np.random.normal(0, noise_level, len(y))

# Function to add heteroskedastic noise to the ground truth
def add_heteroskedastic_noise(y, noise_factor):
    return y + np.random.normal(0, noise_factor * np.abs(np.sin(2 * np.pi * X)), len(y))

def rmse(y_pred, y_true):
    return np.sqrt(np.mean((y_pred - y_true) ** 2))
# Streamlit UI
st.title("MAPIE Regression Demo with Homoskedastic/Heteroskedastic Noise")

# Sidebar for user input
st.sidebar.header("Parameters")
dataset_type = st.sidebar.selectbox("Dataset Type", ["Sine", "Cosine", "Polynomial"])
noise_type = st.sidebar.selectbox("Noise Type", ["Homoskedastic", "Heteroskedastic"])
noise_level = st.sidebar.slider("Noise Level/Factor", min_value=0.0, max_value=1.0, value=0.1, step=0.05)

# Generate synthetic data based on the selected dataset type
np.random.seed(0)
X = np.linspace(-1, 1, 100)

if dataset_type == "Sine":
    y_ground_truth = np.sin(2 * np.pi * X)
elif dataset_type == "Cosine":
    y_ground_truth = np.cos(2 * np.pi * X)
elif dataset_type == "Polynomial":
    degree = st.sidebar.slider("Degree", min_value=1.0, max_value=6.0, value=1.0, step=1.0)
    coefficients = np.random.uniform(-10, 10, int(degree))
    print(coefficients)
    y_ground_truth =  np.polyval(coefficients, X)

else:
    st.error("Invalid dataset type")
    st.stop()

# Add noise based on user selection
if noise_type == "Homoskedastic":
    y_noisy = add_homoskedastic_noise(y_ground_truth, noise_level)
else:  # "Heteroskedastic"
    y_noisy = add_heteroskedastic_noise(y_ground_truth, noise_level)

# Train a model using MAPIE
model = GradientBoostingRegressor()
mape_model = MapieRegressor(model)
mape_model.fit(X.reshape(-1, 1), y_noisy)

# Main content
st.write(f"### Ground Truth vs. Noisy Data vs. MAPIE Prediction ({noise_type} Noise)")

# Plotting the ground truth
st.line_chart({"Ground Truth": y_ground_truth, "Noisy Data": y_noisy, "MAPIE Prediction": mape_model.predict(X.reshape(-1, 1))})
st.scatter_chart({"Ground Truth": y_ground_truth, "Noisy Data": y_noisy, "MAPIE Prediction": mape_model.predict(X.reshape(-1, 1))})

# Display model performance
st.write("### Model Performance")
st.write(f"MAPIE Prediction R2 Score: {mape_model.score(X.reshape(-1, 1), y_noisy)}")
# st.write(f"MAPIE Prediction MAE: {mape_model.error(X.reshape(-1, 1), y_noisy)}")
st.write(f"MAPE Prediction RMSE: {rmse(mape_model.predict(X.reshape(-1, 1)), y_noisy)}")

# Display uncertainty intervals
st.write("### Prediction Uncertainty Intervals (90%)")
y_pred, y_pis = mape_model.predict(X.reshape(-1, 1), alpha=0.1)
lower_bound = [subarray[0, 0] for subarray in y_pis]
upper_bound = [subarray[1, 0] for subarray in y_pis]
st.line_chart({"Lower Bound": lower_bound, "Upper Bound": upper_bound, "MAPIE Prediction": mape_model.predict(X.reshape(-1, 1))})
