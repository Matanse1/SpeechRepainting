import pandas as pd

# Load the CSV file
df = pd.read_csv('/dsi/gannot-lab1/datasets/FSD50K/FSD50K.ground_truth/eval.csv')

# Filter rows where 'explosion' appears in the labels
filtered_df = df[df['labels'].str.contains('explosion', case=False, na=False)]

# Save the filtered DataFrame to a new CSV file
filtered_df.to_csv('/dsi/gannot-lab1/datasets/FSD50K/FSD50K.ground_truth/eval_explosion_labels.csv', index=False)

print("Rows related to 'explosion' have been saved to explosion_labels.csv")
