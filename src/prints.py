import os
import pandas as pd
from dataset import load_metadata, create_fairness_df
from config import PLOT_DIR

def main():
    tile_df = load_metadata()
    fairness_df = create_fairness_df(tile_df)

    #counts = fairness_df.groupby(['age_group', 'cell_type']).size().unstack(fill_value=0)

    #print("Number of instances per age group and lesion type:")
    #print(counts)

    print("\n\n=== PREDICTION COUNTS PER MODEL ===")

    results = {
        "Baseline": os.path.join(PLOT_DIR, "baseline_results.csv"),
        "Reweighting": os.path.join(PLOT_DIR, "reweighting_results.csv"),
        "Equalized Odds": os.path.join(PLOT_DIR, "equalized_odds_results.csv"),
        "Adversarial": os.path.join(PLOT_DIR, "adversarial_debiasing_results.csv"),
    }

    for name, path in results.items():
        print(f"\n>> Prediction counts for {name} model:")
        df = pd.read_csv(path)

        if 'Prediction' not in df.columns:
            print("Column 'predicted_class' not found in the data!")
            continue

        counts = df['Prediction'].value_counts().sort_index()

        # Print counts per predicted class
        for cls, count in counts.items():
            print(f"Class {cls}: {count}")

    for name, path in results.items():
        print(f"\n>> True label counts for {name} model:")
        df = pd.read_csv(path)

        if 'Label' not in df.columns:
            print("Column 'true_class' not found in the data!")
            continue

        true_counts = df['Label'].value_counts().sort_index()

        for cls, count in true_counts.items():
            print(f"Class {cls}: {count}")

def tpr_fpr():
    results = {
    "Baseline": os.path.join(PLOT_DIR, "baseline_results.csv"),
    "Reweighting": os.path.join(PLOT_DIR, "reweighting_results.csv"),
    "Equalized Odds": os.path.join(PLOT_DIR, "equalized_odds_results.csv"),
    "Adversarial": os.path.join(PLOT_DIR, "adversarial_debiasing_results.csv"),
    }

    tpr_fpr_dict = {}

    for method, filepath in results.items():
        df = pd.read_csv(filepath)
        labels = df['Label']
        preds = df['Prediction']
        
        all_classes = sorted(labels.unique())

        TP = FP = FN = TN = 0
        
        for cls in all_classes:
            tp = ((preds == cls) & (labels == cls)).sum()
            fp = ((preds == cls) & (labels != cls)).sum()
            fn = ((preds != cls) & (labels == cls)).sum()
            tn = ((preds != cls) & (labels != cls)).sum()
            
            TP += tp
            FP += fp
            FN += fn
            TN += tn

        TPR = TP / (TP + FN) if (TP + FN) > 0 else float('nan')
        FPR = FP / (FP + TN) if (FP + TN) > 0 else float('nan')
        
        tpr_fpr_dict[method] = {'TPR': TPR, 'FPR': FPR}

    # Print the micro-averaged results
    for method, scores in tpr_fpr_dict.items():
        print(f"{method}: TPR = {scores['TPR']:.3f}, FPR = {scores['FPR']:.3f}")

if __name__ == "__main__":
    tpr_fpr()
