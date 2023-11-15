import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from mapie.regression import MapieRegressor
from sklearn.metrics import coverage_error

# Function to add homoskedastic noise to the ground truth
def add_homoskedastic_noise(y, noise_level):
    return y + np.random.normal(0, noise_level, len(y))

# Function to add heteroskedastic noise to the ground truth
def add_heteroskedastic_noise(y, noise_factor):
    return y + np.random.normal(0, noise_factor * X**2, len(y))

def rmse(y_pred, y_true):
    return np.sqrt(np.mean((y_pred - y_true) ** 2))

# Streamlit UI
st.set_page_config(
    page_title="MAPIE Regression Demo with Noise",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar for user input
st.sidebar.header("Parameters")
X_lower = st.sidebar.slider("Dataset X lower limit", min_value=-100.0, max_value=100.0, value=-15.0, step=1.0)
X_upper = st.sidebar.slider("Dataset X upper limit", min_value=-100.0, max_value=100.0, value=23.0, step=1.0)
dataset_type = st.sidebar.selectbox("Dataset Type", ["Sine", "Cosine", "Polynomial"])
noise_type = st.sidebar.selectbox("Noise Type", ["Homoskedastic", "Heteroskedastic"], index=1)
noise_level = st.sidebar.slider("Noise Level/Factor", min_value=0.0, max_value=1.0, value=0.1, step=0.05)


st.title("MAPIE Regression Demo with Noise")

if st.button("Description"):
    # Display a message when the button is clicked
    st.markdown(
        """
        <div style="border-radius: 10px; background-color: #f0f0f0; padding: 15px; margin-top: 10px;">
            <p style="font-size: 18px; color: #333;">This is a demo of using the MAPIE library for regression over synthetic data with noise. The user can select the type of dataset, noise type, and noise level/factor. The MAPIE prediction and prediction interval are displayed in the plot below. The model performance metrics and coverage are displayed in the table below the plot.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    



# Generate synthetic data based on the selected dataset type
np.random.seed(0)


X = np.linspace(X_lower, X_upper, 100)

if dataset_type == "Sine":
    y_ground_truth = np.sin(2 * np.pi * X)
elif dataset_type == "Cosine":
    y_ground_truth = np.cos(2 * np.pi * X)
elif dataset_type == "Polynomial":
    degree = st.sidebar.slider("Degree", min_value=1.0, max_value=6.0, value=1.0, step=1.0)
    coefficients = np.random.uniform(-10, 10, int(degree))
    y_ground_truth = np.polyval(coefficients, X)
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

# Dataset plot
st.write("## Dataset")
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(X, y_ground_truth, label='Ground Truth')
ax.scatter(X, y_noisy, label='Noisy Data', color='red', alpha=0.7)
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.legend()
st.pyplot(fig)

# Model prediction plot
st.write("## Dataset vs. MAPIE Prediction")
fig, ax = plt.subplots(figsize=(10, 6))
# ax.plot(X, y_ground_truth, label='Ground Truth')
ax.plot(X, mape_model.predict(X.reshape(-1, 1)), label='MAPIE Prediction', linestyle='--', color='blue')
ax.scatter(X, y_noisy, label='Noisy Data', color='red', alpha=0.7)
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.legend()
st.pyplot(fig)


# Model performance and prediction interval
st.write("## Model Performance and Confidence Interval")


# Create a table for displaying metrics with custom styling

# Uncertainty interval slider
confidence_interval = st.slider("Select confidence level", min_value=0.01, max_value=99.99, value=95.00, step=5.00, key="confidence_interval")
alpha = (100 - confidence_interval) / 100

# Prediction interval plot
y_pred, y_pis = mape_model.predict(X.reshape(-1, 1), alpha=alpha)
lower_bound = [subarray[0, 0] for subarray in y_pis]
upper_bound = [subarray[1, 0] for subarray in y_pis]

fig, ax = plt.subplots(figsize=(10, 6))
is_within_interval = np.logical_and(y_noisy >= y_pis[:, 0, 0], y_noisy <= y_pis[:, 1, 0])
coverage = np.mean(is_within_interval)

# Applying styling to the table using HTML
styled_table = f"""
    <style>
        table {{
            font-size: 18px;
            color: #000066; /* Dark blue color */
            border-collapse: collapse;
            width: 50%; /* Adjust the width as needed */
            margin: auto; /* Center the table */
            border-spacing: 0; /* Remove spacing between cells */
            background-color: #e6f7ff; /* Light blue background color */
        }}
        th, td {{
            padding: 20px 30px;
            text-align: left;
            border-bottom: 1px solid #ecf0f1; /* Light gray border between rows */
        }}
        th {{
            border-bottom: 2px solid #2980b9; /* Darker blue border for header row */
        }}
    </style>
    <table>
        <tr>
            <th>Metric</th>
            <th>Value</th>
        </tr>
        <tr>
            <td>MAPIE Prediction R2 Score</td>
            <td>{mape_model.score(X.reshape(-1, 1), y_noisy):.4f}</td>
        </tr>
        <tr>
            <td>MAPIE Prediction RMSE</td>
            <td>{rmse(mape_model.predict(X.reshape(-1, 1)), y_noisy):.4f}</td>
        </tr>
        <tr>
            <td>Coverage</td>
            <td>{coverage:.4f}</td>
        </tr>
    </table>
"""

# Display the styled table using markdown
st.markdown(styled_table, unsafe_allow_html=True)


# Display the styled table using markdown
# st.markdown(styled_table, unsafe_allow_html=True)

st.text(" ")
st.text(" ")

st.text(" ")


ax.scatter(X, y_noisy, label='Noisy Data', color='red', alpha=0.7)
ax.plot(X, y_pred, label='MAPIE Prediction', linestyle='--', color='blue')
ax.fill_between(X, lower_bound, upper_bound, color='gray', alpha=0.3, label='Confidence Interval')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_title(f'MAPIE Prediction and Prediction Interval ({confidence_interval}% Confidence)')
ax.legend()
st.pyplot(fig)
