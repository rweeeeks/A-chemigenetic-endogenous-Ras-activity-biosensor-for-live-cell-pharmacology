# Import necessary libraries
import pandas as pd
import numpy as np
import os

# Define the file path - replace with your actual file path
file_path = r"xxx"  # Update this path

# Read the Excel file
df = pd.read_excel(file_path, header=None, skiprows=1)  

# Initialize dictionaries to store the results
sustained_activity_metrics = {}
baseline_averages = {}

# Iterate over each y-axis column (skipping the first column which is X values)
for i, column in enumerate(df.columns[1:], 1):
    # Extract the baseline and response values
    baseline = df[column].iloc[:10]
    response = df[column].iloc[10:]

    # Calculate the baseline average
    baseline_avg = baseline.mean()
    baseline_averages[f"Column_{i}"] = baseline_avg

    # Find the maximum value as defined
    max_value = max((response.iloc[i] + response.iloc[i+1]) / 2 for i in range(len(response)-1))

    # Get the final value of the response
    final_value = response.iloc[-1]

    # Check if (max_value - baseline_avg) or (final_value - baseline_avg) are less than zero
    if (max_value - baseline_avg) < 0 or (final_value - baseline_avg) < 0:
        # If the condition is met, store NaN
        sustained_activity_metrics[f"Column_{i}"] = np.nan
    else:
        # Calculate the sustained activity metric
        sustained_activity_metric = (final_value - baseline_avg) / (max_value - baseline_avg)
        # Store the result
        sustained_activity_metrics[f"Column_{i}"] = sustained_activity_metric

# Convert the results to a DataFrame with a single row
results_df = pd.DataFrame([
    sustained_activity_metrics,
    baseline_averages
], index=['SAM', 'Baseline'])

# Save the results to a new Excel file
output_path = "sustained_activity_metrics.xlsx"
results_df.to_excel(output_path, index=True, header=True)

print(f"Results saved to {os.path.abspath(output_path)}")