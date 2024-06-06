import pandas as pd
import matplotlib.pyplot as plt

# Load the data from CSV files
file_names = [
    'training_plots/results_low_res.csv', 'training_plots/results_cutmix.csv', 'training_plots/results_cutout.csv', 
    'training_plots/results_mixup.csv', 'training_plots/results_random.csv', 'training_plots/results_bernoulli.csv',
]

dataframes = [pd.read_csv(f'{file}', skipinitialspace=True,) for file in file_names]

train_metrics = ['train/box_loss', 'train/cls_loss', 'train/dfl_loss']
val_metrics = ['val/box_loss', 'val/cls_loss', 'val/dfl_loss']
other_metrics = ['metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)']

legend_labels = ["Original", "CutMix", "CutOut", "MixUp", "Uniform", "Bernoulli"]

colors = ['b', 'g', 'r', 'c', 'm', 'y']


def create_plot(metrics, title, filename):
    fig, axes = plt.subplots(nrows=1, ncols=len(metrics), figsize=(15, 5))
    for i, metric in enumerate(metrics):
        ax = axes[i]
        for j, df in enumerate(dataframes):
            ax.plot(df['epoch'], df[metric], label=legend_labels[j], color=colors[j])
        ax.set_title(metric)
        ax.legend()
    # plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

# Plot train-related metrics
create_plot(train_metrics, 'Train Metrics', 'train_metrics.png')

# Plot validation-related metrics
create_plot(val_metrics, 'Validation Metrics', 'val_metrics.png')

# Plot other metrics
create_plot(other_metrics, 'Other Metrics', 'other_metrics.png')