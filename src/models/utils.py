# Created by Pablo Campillo at 6/1/23
import pandas as pd


def get_X_y(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path, sep=';', index_col=0)

    df['foundation_type'] = df['foundation_type'].astype('category')
    df['concrete_code'] = df['concrete_code'].astype('category')
    df['load_code'] = df['load_code'].astype('category')
    df['wind_code'] = df['wind_code'].astype('category')
    df['seismic_code'] = df['seismic_code'].astype('category')

    df = df.drop(['center_x', 'center_y', 'cdm_x', 'cdm_y', 'cdr_x', 'cdr_y'], axis=1)
    X = df.drop("reinf_steel_ratio", axis=1)
    y = df["reinf_steel_ratio"].copy()

    return X, y
