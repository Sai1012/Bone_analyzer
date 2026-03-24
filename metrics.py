import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import json

# Load health scores and diagnosis labels
def load_data(health_scores_path, diagnosis_labels_path):
    health_scores = pd.read_csv(health_scores_path)
    diagnosis_labels = pd.read_csv(diagnosis_labels_path)
    return health_scores, diagnosis_labels

# Compute confusion matrix and classification metrics

def compute_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    return cm, report

# Generate correlation analysis

def correlation_analysis(df):
    corr = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title('Correlation heatmap')
    plt.savefig('output/correlation_heatmap.png')

# Save results to JSON

def save_results(results, filepath):
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=4)

# Main function

def main():
    health_scores_path = 'output/health_scores.csv'
    diagnosis_labels_path = 'output/diagnosis_labels.csv'  # Make sure to have this file
    health_scores, diagnosis_labels = load_data(health_scores_path, diagnosis_labels_path)

    # Assuming health_scores has a column 'score' and diagnosis_labels a column 'label'
    y_true = diagnosis_labels['label']
    y_pred = health_scores['score'].apply(lambda x: 1 if x > threshold else 0)  # threshold to be defined

    # Compute metrics
    cm, report = compute_metrics(y_true, y_pred)
    correlation_analysis(health_scores)

    # Saving results
    results = {
        'confusion_matrix': cm.tolist(),
        'classification_report': report
    }
    save_results(results, 'output/metrics_summary.json')

if __name__ == '__main__':
    main()