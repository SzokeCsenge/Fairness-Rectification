import os
import pandas as pd
from fairness_metrics import calculate_fairness_metrics
from config import PLOT_DIR

print("\n\n=== FAIRNESS COMPARISON ===")

results = {
    "Baseline": os.path.join(PLOT_DIR, "baseline_results.csv"),
    "Reweighting": os.path.join(PLOT_DIR, "reweighting_results.csv"),
    "Equalized Odds": os.path.join(PLOT_DIR, "equalized_odds_results.csv"),
    "Adversarial": os.path.join(PLOT_DIR, "adversarial_debiasing_results.csv"),
}

for name, path in results.items():
    print(f"\n>> Fairness Metrics for {name} model:")
    df = pd.read_csv(path)
    calculate_fairness_metrics(df, num_classes=7, group_col="Age_Group")
