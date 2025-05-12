import numpy as np
from sklearn.metrics import confusion_matrix


def equalized_odds(df, class_id, group_col="Age_Group"):
    """Compute Equalized Odds (TPR and FPR) for a given class."""
    df = df.copy()
    df["label_bin"] = (df["Label"] == class_id).astype(int)
    df["pred_bin"] = (df["Prediction"] == class_id).astype(int)

    tpr_fpr = {}
    for group in sorted(df[group_col].dropna().unique()):
        group_df = df[df[group_col] == group]
        y_true = group_df["label_bin"]
        y_pred = group_df["pred_bin"]

        try:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        except ValueError:
            tn = fp = fn = tp = 0

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        tpr_fpr[group] = {"TPR": tpr, "FPR": fpr}
    return tpr_fpr

def tpr_fpr_variance(eo_dict):
    """Returns variance of TPR and FPR across groups."""
    tprs = np.array([v["TPR"] for v in eo_dict.values()])
    fprs = np.array([v["FPR"] for v in eo_dict.values()])
    return np.var(tprs), np.var(fprs)


def tpr_fpr_mean_diff(eo_dict):
    """Returns mean absolute difference from the mean for TPR and FPR."""
    tprs = np.array([v["TPR"] for v in eo_dict.values()])
    fprs = np.array([v["FPR"] for v in eo_dict.values()])
    return np.abs(tprs - tprs.mean()).mean(), np.abs(fprs - fprs.mean()).mean()


def calculate_fairness_metrics(df, num_classes=7, group_col="Age_Group"):
    tpr_dev_list = []
    fpr_dev_list = []

    for cls in range(num_classes):
        print(f"\n=== Class {cls} (vs. Rest) ===")
        # Equalized Odds
        eo = equalized_odds(df, class_id=cls, group_col=group_col)

        var_tpr, var_fpr = tpr_fpr_variance(eo)

        mean_diff_tpr, mean_diff_fpr = tpr_fpr_mean_diff(eo)
        
        tpr_dev_list.append(mean_diff_tpr)
        fpr_dev_list.append(mean_diff_fpr)

        print("Equalized Odds:")

        print(f"TPR Variance: {var_tpr:.5f}")
        print(f"FPR Variance: {var_fpr:.5f}")
    
    avg_tpr_dev= np.mean(tpr_dev_list)
    avg_fpr_dev = np.mean(fpr_dev_list)

    print("\n=== Overall Mean Absolute Deviation ===")
    print(f"TPR = {avg_tpr_dev:.5f}, FPR = {avg_fpr_dev:.5f}")

    class_counts = df["Label"].value_counts().sort_index()

    # Convert to relative frequencies
    total = class_counts.sum()
    class_weights = class_counts / total

    # Weighted mean
    weighted_tpr = np.average(tpr_dev_list, weights=class_weights)
    weighted_fpr = np.average(fpr_dev_list, weights=class_weights)

    print("\n=== Weighted Overall Mean Absolute Deviation ===")
    print(f"TPR = {weighted_tpr:.5f}, FPR = {weighted_fpr:.5f}")
