import pandas as pd


workload_name = "workload-8"
# Read the original CSV file
original_df = pd.read_csv(f"{workload_name}.csv")

# Define the number of repetitions
num_repetitions = 80

# Create a new DataFrame by repeating the original DataFrame
repeated_df = pd.concat([original_df] * num_repetitions, ignore_index=True)

# Increment job_ids and arrival_times based on the pattern
for i in range(num_repetitions):
    repeated_df.loc[i * len(original_df):(i + 1) * len(original_df) - 1, 'name'] += i * len(original_df)
    repeated_df.loc[i * len(original_df):(i + 1) * len(original_df) - 1, 'time'] += i * original_df.loc[(len(original_df) - 1), 'time']

# Write the new DataFrame to a new CSV file
repeated_df.to_csv(f"{workload_name}-noheader.csv", index=False)