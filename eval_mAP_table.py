import os
import pandas as pd
import matplotlib.pyplot as plt

# Define the paths to the directories containing the files
base_dir = 'mAP'
sub_dirs = ['test', 'test_aug', 'train_mAP', 'val_mAP']
augmentation_strategies = ['original-low-res', 'cutmix', 'cutout', 'mixup', 'random-aug', 'bernoulli']

# Dictionary to store the results
results_map50 = {strategy: {} for strategy in augmentation_strategies}
results_map50_95 = {strategy: {} for strategy in augmentation_strategies}

def extract_map_values(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        # Assuming mAP50 is on line 2 and mAP50-95 is on line 3
        map50 = float(lines[1].split()[5])
        map50_95 = float(lines[1].split()[6])
        return map50, map50_95

# Iterate over each subdirectory and file to extract the mAP values
for sub_dir in sub_dirs:
    for strategy in augmentation_strategies:
        file_path = os.path.join(base_dir, sub_dir, f'mAP_table_last-{strategy}.pt.txt')
        if os.path.exists(file_path):
            map50, map50_95 = extract_map_values(file_path)
            results_map50[strategy][f'{sub_dir}:mAP50'] = map50
            results_map50_95[strategy][f'{sub_dir}:mAP50-95'] = map50_95
        else:
            print(f"File not found: {file_path}")

# Convert the results to DataFrames
df_map50 = pd.DataFrame(results_map50).T
df_map50_95 = pd.DataFrame(results_map50_95).T

# Reorder the columns
column_order_map50 = [
    'train_mAP:mAP50', 'val_mAP:mAP50', 'test:mAP50', 'test_aug:mAP50'
]
column_order_map50_95 = [
    'train_mAP:mAP50-95', 'val_mAP:mAP50-95', 'test:mAP50-95', 'test_aug:mAP50-95'
]

df_map50 = df_map50[column_order_map50]
df_map50_95 = df_map50_95[column_order_map50_95]

# Print the DataFrames
print(df_map50)
print(df_map50_95)

# Save the DataFrames to CSV files
df_map50.to_csv('mAP50_results.csv')
df_map50_95.to_csv('mAP50-95_results.csv')

# Optionally, plot the tables as images
def plot_table(df, filename):
    plt.figure(figsize=(12, 6))
    plt.axis('tight')
    plt.axis('off')
    table = plt.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(df.columns))))
    plt.savefig(filename)
    plt.show()

plot_table(df_map50, 'mAP50_results.png')
plot_table(df_map50_95, 'mAP50-95_results.png')
