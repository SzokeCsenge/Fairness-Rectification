import os

# BASE_SKIN_DIR = "/Users/csengeszoke/programming/Thesis/Code/Fairness-Rectification/Data"

BASE_SKIN_DIR = "/scratch/s5139090/Fairness-Rectification/Data"

METADATA_PATH = os.path.join(BASE_SKIN_DIR, "HAM10000_metadata.csv")

# Output directory for figures or model checkpoints
# PLOT_DIR = "/Users/csengeszoke/programming/Thesis/Code/Fairness-Rectification/Data"

PLOT_DIR = "/scratch/s5139090/Fairness-Rectification/Outputs"

LESION_TYPE_DICT = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis-like lesions',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}

AGE_GROUP_DICT = {
    '0–25': 0,
    '25-40': 1,
    '40–55': 2,
    '55–70': 3,
    '70+': 4
}