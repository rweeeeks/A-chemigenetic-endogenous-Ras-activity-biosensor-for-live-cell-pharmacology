import numpy as np
import pandas as pd
import os

# Define the file path - replace with your actual file path
file_path = r""  # Update this path

# Read the uploaded file into a pandas DataFrame
data = pd.read_excel(file_path, header=None, skiprows=1)  # Skip the first row with labels

# Extract x and y values
x = data.iloc[:, 0].values
y = data.iloc[:, 1:].values

# Calculate the baseline (average of the first 10 y values)
baseline = np.mean(y[:10, :], axis=0)
baseline_std = np.std(y[:10, :], axis=0)  # Standard deviation of the baseline

# Calculate the maximum using scanning of three consecutive cells outside the baseline
maximum = np.array([
    max((y[i, j] + y[i + 1, j] + y[i + 2, j]) / 3 for i in range(10, y.shape[0] - 2))
    for j in range(y.shape[1])
])

# Initialize half_maximum with NaN values
half_maximum = np.full(maximum.shape, np.nan)

# Calculate the half maximum only if the maximum is greater than the baseline + 1 std
valid_indices = maximum > (baseline + baseline_std)
half_maximum[valid_indices] = (baseline[valid_indices] + maximum[valid_indices]) / 2

# Define tolerance for floating-point comparisons, necessary for checking if interpolated T1/2 values fall within the baseline x range
tolerance = 1e-6
baseline_x_range = (x[0], x[9])  

# Find the time to half maximum with linear interpolation
time_to_half_max = np.full(y.shape[1], np.nan)
for i in range(y.shape[1]):
    if valid_indices[i]:  # Ensure max is sufficiently higher than baseline
        above_half_max = np.where(y[:, i] > half_maximum[i])[0]
        # Exclude T1/2 values within the first 10 data points
        above_half_max = above_half_max[above_half_max >= 10]
        if above_half_max.size > 0:
            idx = above_half_max[0]
            if idx > 0 and idx < len(x) - 1:
                x1, x2 = x[idx - 1], x[idx]
                y1, y2 = y[idx - 1, i], y[idx, i]
                # Linear interpolation
                interpolated_t1_2 = x1 + (half_maximum[i] - y1) * (x2 - x1) / (y2 - y1)
                # Check if the interpolated value falls within the baseline x range
                if not (baseline_x_range[0] - tolerance <= interpolated_t1_2 <= baseline_x_range[1] + tolerance):
                    time_to_half_max[i] = interpolated_t1_2

# Print the baseline and maximum values
print('Baseline values:')
print(baseline)
print('Maximum values:')
print(maximum)

# Create a DataFrame with additional metrics
results_df = pd.DataFrame([
    time_to_half_max,
    baseline,
    maximum,
    half_maximum
], index=['TimeToHalfMax', 'Baseline', 'Maximum', 'HalfMaximum'])

# Save the results to a new Excel file
output_path = "time_to_half_max.xlsx"
results_df.to_excel(output_path, index=True, header=True)

print(f"Results saved to {os.path.abspath(output_path)}")