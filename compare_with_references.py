# Improved compare_with_references.py script

import matplotlib.pyplot as plt
import pandas as pd

# Load data
# Your code to load the data would go here.

def plot_range(data):
    study_name = 'Study Name'  # Replace with actual study name
    auc_value = data['AUC']  # Example metric
    plt.figure(figsize=(10, 5))
    plt.plot(data['Range'], auc_value, marker='o')
    plt.title(f'{study_name} (AUC)')  # Update title
    plt.xlabel('Range')
    plt.ylabel('AUC')
    plt.savefig('output/range_plot.png')


def plot_bar(data):
    study_name = 'Study Name'  # Replace with actual study name
    metrics = data['Metric']  # Example metric
    plt.figure(figsize=(10, 5))
    plt.bar(metrics['Name'], metrics['Value'])
    plt.title(f'{study_name} (Metric)')  # Update title
    plt.xlabel('Metrics')
    plt.ylabel('Value')
    plt.savefig('output/bar_plot.png')

# Your main execution would go here where you call the plotting functions.
