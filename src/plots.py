import matplotlib.pyplot as plt
import seaborn as sns
from config import PLOT_DIR

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
