# Created by Pablo Campillo at 7/1/23
import pandas as pd
import numpy as np
from sklearn.model_selection import learning_curve
from sklearn.metrics import mean_absolute_percentage_error, make_scorer
from matplotlib import pyplot as plt

def train_and_plot_learning_curve(model, X, y):
    train_sizes, train_scores, valid_scores = learning_curve(model, X, y,
                                                             train_sizes = np.linspace(0.01, 1.0, 40),
                                                             cv=5,
                                                             scoring=make_scorer(mean_absolute_percentage_error),
                                                            n_jobs=-1)
    train_errors = train_scores.mean(axis=1)
    valid_errors = valid_scores.mean(axis=1)

    fig, ax = plt.subplots()
    ax.plot(train_sizes, train_errors, "r-+", linewidth=2, label="train")
    ax.plot(train_sizes, valid_errors, "b-", linewidth=3, label="valid")
    ax.set_xlabel('Number of samples')
    ax.set_ylabel('MAPE')
    return fig, ax


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