import matplotlib.pyplot as plt
import seaborn as sns
from config import PLOT_DIR
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score

sns.set(style="whitegrid")

def plot_age_distribution(df):
    plt.figure(figsize=(10, 6))
    sns.histplot(df['age'], bins=range(0, 86, 5), kde=False)
    plt.title('Distribution of Patient Ages', fontsize=18)
    plt.xlabel('Age', fontsize=18)
    plt.ylabel('Count', fontsize=18)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.savefig(f'{PLOT_DIR}/age_distribution.png', format='png')
    plt.show()

def plot_class_distribution(df):
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='cell_type', order=df['cell_type'].value_counts().index)
    plt.title('Distribution of Diagnosis Classes', fontsize=18)
    plt.xlabel('Diagnosis Class', fontsize=18)
    plt.ylabel('Number of Samples', fontsize=18)
    plt.xticks(rotation=60, fontsize=15)
    plt.yticks(fontsize=15)
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/class_distribution.png', format='png')
    plt.show()

def plot_diagnosis_by_age(df):
    grouped = df.groupby(['age_group', 'cell_type'], observed=False).size().unstack().fillna(0)
    grouped.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='tab20')
    plt.title('Diagnosis Class Distribution Across Age Groups', fontsize=18)
    plt.xlabel('Age Group', fontsize=18)
    plt.ylabel('Number of Samples', fontsize=18)
    plt.xticks(rotation=0, fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(title='Diagnosis Class', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/diagnosis_by_age_group.png', format='png')
    plt.show()

def new_plot():
    lesions = ['akiec', 'bcc', 'bkl', 'df', 'nv', 'mel', 'vasc']

    ground_truth = [66, 102, 235, 23, 1355, 230, 29]
    baseline = [67, 115, 149, 16, 1470, 190, 33]
    reweighting = [74, 94, 167, 25, 1418, 228, 34]
    regularizer = [51, 86, 178, 1, 1565, 122, 37]
    adversarial = [66, 125, 162, 12, 1456, 190, 29]

    data = [ground_truth, baseline, reweighting, regularizer, adversarial]
    labels = ['Ground Truth', 'Baseline', 'Re-weighting', 'Regularizer', 'Adversarial']

    x = np.arange(len(lesions))
    width = 0.15

    fig, ax = plt.subplots(figsize=(8, 6))
    tab20 = plt.get_cmap('tab20').colors
    custom_colors = [tab20[i] for i in [4, 12, 6, 3, 0]]

    for i, values in enumerate(data):
        ax.bar(x + i*width - 2*width, values, width, label=labels[i], color=custom_colors[i])

    ax.set_title('Prediction Counts per Lesion Type by Model', fontsize=18)
    ax.set_ylabel('Count', fontsize=18)
    ax.set_xlabel('Lesion Type', fontsize=18)
    ax.set_xticks(x)
    ax.set_xticklabels(lesions, fontsize=15)
    ax.yaxis.set_tick_params(labelsize=15)

    ax.legend(title='Model', loc='upper left', fontsize=12, title_fontsize=13, frameon=True)

    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/ground_truth2.png', format='png')
    plt.show()

def confusion():
    lesion_labels = ['akiec', 'bcc', 'bkl', 'df', 'nv', 'mel', 'vasc']

    # Model result file paths (update with actual paths)
    model_files = {
    'Baseline': '../Outputs/baseline_results.csv',
    'Re-weighting': '../Outputs/reweighting_results.csv',
    'Regularizer': '../Outputs/equalized_odds_results.csv',  # previously eo_df
    'Adversarial': '../Outputs/adversarial_debiasing_results.csv'
    }

    # Setup 2x2 subplot grid
    fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    axs = axs.flatten()

    for i, (model_name, csv_file) in enumerate(model_files.items()):
        df = pd.read_csv(csv_file)
        
        y_true = df['Label']
        y_pred = df['Prediction']
        
        cm = confusion_matrix(y_true, y_pred, labels=range(len(lesion_labels)))
        
        sns.heatmap(cm, annot=True, fmt='d',
                    xticklabels=lesion_labels, yticklabels=lesion_labels,
                    cmap='Blues', ax=axs[i])
        
        axs[i].set_title(f'{model_name} Confusion Matrix', fontsize=14)
        axs[i].set_xlabel('Predicted Label')
        axs[i].set_ylabel('True Label')

    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/confusion_matrices.png')
    plt.show()

def accuracy():
    model_files = {
        'Baseline': '../Outputs/baseline_results.csv',
        'Re-weighting': '../Outputs/reweighting_results.csv',
        'Regularizer': '../Outputs/equalized_odds_results.csv',
        'Adversarial': '../Outputs/adversarial_debiasing_results.csv'
    }

    for model_name, csv_file in model_files.items():
        df = pd.read_csv(csv_file)
        
        print(f"\nAccuracy per Age Group for {model_name}:")
        
        for age_group, group_df in df.groupby('Age_Group'):
            acc = accuracy_score(group_df['Label'], group_df['Prediction'])
            print(f"  Age Group {age_group}: {acc:.3f}")
        
        overall_acc = accuracy_score(df['Label'], df['Prediction'])
        print(f"Overall Accuracy: {overall_acc:.3f}")