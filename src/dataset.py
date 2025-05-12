import os
import pandas as pd
from glob import glob
from config import BASE_SKIN_DIR, METADATA_PATH, LESION_TYPE_DICT

def load_image_paths(base_dir):
    return {
        os.path.splitext(os.path.basename(x))[0]: x
        for x in glob(os.path.join(base_dir, '*', '*.jpg'))
    }

def load_metadata():
    tile_df = pd.read_csv(METADATA_PATH)
    imageid_path_dict = load_image_paths(BASE_SKIN_DIR)

    tile_df['path'] = tile_df['image_id'].map(imageid_path_dict.get)
    tile_df['cell_type'] = tile_df['dx'].map(LESION_TYPE_DICT.get)
    tile_df['cell_type_idx'] = pd.Categorical(tile_df['cell_type']).codes

    return tile_df

def create_fairness_df(tile_df):
    df = pd.DataFrame()
    df['filepaths'] = tile_df['path']
    df['labels'] = tile_df['cell_type_idx']
    df['cell_type'] = tile_df['cell_type']
    df['age'] = tile_df['age']
    df['sex'] = tile_df['sex']
    df['lesion_id'] = tile_df['lesion_id']

    bins = [0, 25, 40, 55, 70, 100]
    labels = ["0–25", "25-40", "40–55", "55–70", "70+"]
    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, include_lowest=True)

    df = df.dropna(subset=['age_group', 'age'])

    return df