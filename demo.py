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
    return y + np.random.normal(0,noise_factor * X**2, len(y))

def rmse(y_pred, y_true):
    return np.sqrt(np.mean((y_pred - y_true) ** 2))
# Streamlit UI
st.title("MAPIE Regression Demo with Noise")

# Sidebar for user input
st.sidebar.header("Parameters")
dataset_type = st.sidebar.selectbox("Dataset Type", ["Sine", "Cosine", "Polynomial"])
noise_type = st.sidebar.selectbox("Noise Type", ["Homoskedastic", "Heteroskedastic"], index=1)
noise_level = st.sidebar.slider("Noise Level/Factor", min_value=0.0, max_value=1.0, value=0.1, step=0.05)

# Generate synthetic data based on the selected dataset type
np.random.seed(0)

X_lower = st.slider("X lower limit", min_value=-100.0, max_value=100.0, value=-15.0, step=1.0)
X_upper = st.slider("X upper limit", min_value=-100.0, max_value=100.0, value=23.0, step=1.0)
X = np.linspace(X_lower, X_upper, 100)

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

st.write("### Dataset")

fig, ax = plt.subplots()

# Plot the ground truth as a line
ax.plot(X, y_ground_truth, label='Ground Truth')

# Scatter plot for noisy data
ax.scatter(X, y_noisy, label='Noisy Data', color='red')

# Set labels and title
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
# ax.set_xlim([-1,1])


# Show legend
ax.legend()

# Display the plot in Streamlit
st.pyplot(fig)



# Main content
st.write(f"### Dataset vs. MAPIE Prediction ({noise_type} Noise)")

# Plotting the ground truth
st.line_chart({"Noisy Data": y_noisy, "MAPIE Prediction": mape_model.predict(X.reshape(-1, 1))})


# st.write(f"MAPIE Prediction Coverage: {1 - coverage_error(y_noisy.reshape(-1, 1), mape_model.predict(X.reshape(-1, 1)))}")
confidence_interval = st.slider("Uncertainty Interval", min_value=0.01, max_value=99.99, value=95.00, step=5.00)
alpha = (100-confidence_interval) / 100
y_pred, y_pis = mape_model.predict(X.reshape(-1, 1), alpha=alpha)
lower_bound = [subarray[0, 0] for subarray in y_pis]
upper_bound = [subarray[1, 0] for subarray in y_pis]
fig, ax = plt.subplots()

is_within_interval = np.logical_and(y_noisy >= y_pis[:, 0, 0], y_noisy <= y_pis[:, 1, 0])

# Compute the coverage manually
coverage = np.mean(is_within_interval)
# Display model performance
st.write("### Model Performance")
st.write(f"MAPIE Prediction R2 Score: {mape_model.score(X.reshape(-1, 1), y_noisy)}")
# st.write(f"MAPIE Prediction MAE: {mape_model.error(X.reshape(-1, 1), y_noisy)}")
st.write(f"MAPE Prediction RMSE: {rmse(mape_model.predict(X.reshape(-1, 1)), y_noisy)}")
st.write(f"MAPIE Prediction Coverage: {coverage}")

# Plot the noisy data as a scatter plot
ax.scatter(X, y_noisy, label='Ground Truth', color='red', alpha=0.5)

# Plot the lower and upper bounds as a shaded area
ax.fill_between(X, lower_bound, upper_bound, color='gray', alpha=0.3, label='Prediction Interval with ' + str(confidence_interval) + '% Confidence')

# Plot the mean prediction
ax.plot(X, y_pred, label='MAPIE Prediction', color='blue')

# Set labels and title
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_title('MAPIE Prediction and Prediction Interval with ' + str(confidence_interval) + '% Confidence')
# ax.set_xlim([-1,1])
# ax.set_ylim([-8,8])

# Show legend
ax.legend()

# Display the plot in Streamlit
st.pyplot(fig)




