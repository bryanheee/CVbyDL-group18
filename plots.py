# import pandas as pd
# import matplotlib.pyplot as plt

# # Load the data from CSV files
# file_names = [
#     'training_plots/results_low_res.csv', 'training_plots/results_cutmix.csv', 'training_plots/results_cutout.csv', 
#     'training_plots/results_mixup.csv', 'training_plots/results_random.csv', 'training_plots/results_bernoulli.csv',
# ]

# dataframes = [pd.read_csv(f'{file}', skipinitialspace=True,) for file in file_names]

# train_metrics = ['train/box_loss', 'train/cls_loss', 'train/dfl_loss']
# val_metrics = ['val/box_loss', 'val/cls_loss', 'val/dfl_loss']
# other_metrics = ['metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)']
# plot_title = {
#     'train/box_loss' : 'train/cIoU_loss', 
#     'train/cls_loss' : 'train/BCE_loss', 
#     'train/dfl_loss' : 'train/DFL_loss', 
#     'val/box_loss' : 'val/cIoU_loss',
#     'val/cls_loss' : 'val/BCE_loss',
#     'val/dfl_loss': 'val/DFL_loss', 
# }
# legend_labels = ["Original", "CutMix", "CutOut", "MixUp", "Uniform", "Bernoulli"]

# colors = ['b', 'g', 'r', 'c', 'm', 'y']


# def create_plot(metrics, title, filename):
#     fig, axes = plt.subplots(nrows=1, ncols=len(metrics), figsize=(15, 5))
#     for i, metric in enumerate(metrics):
#         ax = axes[i]
#         for j, df in enumerate(dataframes):
#             ax.plot(df['epoch'], df[metric], label=legend_labels[j], color=colors[j])
#         ax.set_title(plot_title[metric])
#         ax.legend()
#     # plt.suptitle(title)
#     plt.tight_layout()
#     plt.savefig(filename)
#     plt.show()

# # Plot train-related metrics
# # create_plot(train_metrics, 'Train Metrics', 'train_metrics.png')

# # Plot validation-related metrics
# # create_plot(val_metrics, 'Validation Metrics', 'val_metrics.png')

# # Plot other metrics
# # create_plot(other_metrics, 'Other Metrics', 'other_metrics.png')

# def create_combined_plot(train_metrics, val_metrics, title, filename):
#     fig, axes = plt.subplots(nrows=1, ncols=len(train_metrics), figsize=(15, 5))
#     for i, (train_metric, val_metric) in enumerate(zip(train_metrics, val_metrics)):
#         ax = axes[i]
#         for j, df in enumerate(dataframes):
#             ax.plot(df['epoch'], df[train_metric], label=legend_labels[j] + ' Train', color=colors[j])
#             ax.plot(df['epoch'], df[val_metric], label=legend_labels[j] + ' Val', color=colors[j], linestyle='--')
#         ax.set_title(plot_title[train_metric])
#         ax.legend()
#     plt.tight_layout()
#     plt.savefig(filename)
#     plt.show()


# # Plot the training and validation losses with the legend outside the plots
# def create_combined_plot(train_metrics, val_metrics, title, filename):
#     fig, axes = plt.subplots(nrows=1, ncols=len(train_metrics), figsize=(15, 5))
#     for i, metric in enumerate(train_metrics):
#         ax = axes[i]
#         for j, df in enumerate(dataframes):
#             ax.plot(df['epoch'], df[metric], label=f"{legend_labels[j]} Train", color=colors[j])
#             ax.plot(df['epoch'], df[val_metrics[i]], label=f"{legend_labels[j]} Val", linestyle='--', color=colors[j])
#         ax.set_title(plot_title[metric])
#     handles, labels = ax.get_legend_handles_labels()
#     fig.legend(handles, labels, loc='lower center', ncol=3, fontsize='small')
#     plt.tight_layout(rect=[0, 0.1, 1, 1])  # Adjust the rect parameter to make space for the legend
#     plt.savefig(filename)
#     plt.show()

# # Define the paths to the CSV files
# file_names = [
#     'training_plots/results_low_res.csv', 'training_plots/results_cutmix.csv', 'training_plots/results_cutout.csv', 
#     'training_plots/results_mixup.csv', 'training_plots/results_random.csv', 'training_plots/results_bernoulli.csv',
# ]

# dataframes = [pd.read_csv(f'{file}', skipinitialspace=True) for file in file_names]

# train_metrics = ['train/box_loss', 'train/cls_loss', 'train/dfl_loss']
# val_metrics = ['val/box_loss', 'val/cls_loss', 'val/dfl_loss']
# plot_title = {
#     'train/box_loss' : 'train/cIoU_loss', 
#     'train/cls_loss' : 'train/BCE_loss', 
#     'train/dfl_loss' : 'train/DFL_loss', 
#     'val/box_loss' : 'val/cIoU_loss',
#     'val/cls_loss' : 'val/BCE_loss',
#     'val/dfl_loss': 'val/DFL_loss', 
# }
# legend_labels = ["Original", "CutMix", "CutOut", "MixUp", "Uniform", "Bernoulli"]

# colors = ['b', 'g', 'r', 'c', 'm', 'y']

# # Plot the training and validation losses
# create_combined_plot(train_metrics, val_metrics, 'Train and Validation Metrics', 'train_val_metrics_combined.png')
# # Plot combined train and validation metrics
# create_combined_plot(train_metrics, val_metrics, 'Combined Train and Validation Metrics', 'combined_metrics.png')
import pandas as pd
import matplotlib.pyplot as plt

# Define the paths to the CSV files
file_names = [
    'training_plots/results_low_res.csv', 'training_plots/results_cutmix.csv', 'training_plots/results_cutout.csv', 
    'training_plots/results_mixup.csv', 'training_plots/results_random.csv', 'training_plots/results_bernoulli.csv',
]

dataframes = [pd.read_csv(f'{file}', skipinitialspace=True) for file in file_names]

train_metrics = ['train/box_loss', 'train/cls_loss', 'train/dfl_loss']
val_metrics = ['val/box_loss', 'val/cls_loss', 'val/dfl_loss']
plot_title = {
    'train/box_loss' : 'train/cIoU_loss', 
    'train/cls_loss' : 'train/BCE_loss', 
    'train/dfl_loss' : 'train/DFL_loss', 
    'val/box_loss' : 'val/cIoU_loss',
    'val/cls_loss' : 'val/BCE_loss',
    'val/dfl_loss': 'val/DFL_loss', 
}
legend_labels = ["Original", "CutMix", "CutOut", "MixUp", "Uniform", "Bernoulli"]

colors = ['b', 'g', 'r', 'c', 'm', 'y']

# Function to create combined plots with legend outside
def create_combined_plot(train_metrics, val_metrics, title, filename):
    fig, axes = plt.subplots(nrows=1, ncols=len(train_metrics), figsize=(15, 5))
    for i, metric in enumerate(train_metrics):
        ax = axes[i]
        for j, df in enumerate(dataframes):
            ax.plot(df['epoch'], df[metric], label=f"{legend_labels[j]} Train", color=colors[j])
            ax.plot(df['epoch'], df[val_metrics[i]], label=f"{legend_labels[j]} Val", linestyle='--', color=colors[j])
        ax.set_title(plot_title[metric])
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=6, fontsize='x-large', bbox_to_anchor=(0.5, 0.00))
    plt.tight_layout(rect=[0, 0.15, 1, 1])  # Adjust the rect parameter to make space for the legend
    plt.savefig(filename)
    plt.show()

# Plot the training and validation losses
create_combined_plot(train_metrics, val_metrics, 'Train and Validation Metrics', 'train_val_metrics_combined.png')
