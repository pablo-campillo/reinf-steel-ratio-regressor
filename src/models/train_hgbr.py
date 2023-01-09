import logging
import pickle
from pathlib import Path

import click
import numpy as np
import yaml
from dotenv import find_dotenv, load_dotenv
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder

from models.utils import get_X_y


def train(dataset_path:str, model_output_path: str):
    params = yaml.safe_load(open("params.yaml"))["train"]['hgbr']
    learning_rate = params['learning_rate']
    max_depth = params['max_depth']
    max_leaf_nodes = params['max_leaf_nodes']
    min_samples_leaf = params['min_samples_leaf']
    l2_regularization = params['l2_regularization']

    def get_preprocessing_pipeline() -> ColumnTransformer:
        hgbr_cat_pipeline = make_pipeline(
            SimpleImputer(strategy="most_frequent"),
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan))

        hgbr_default_num_pipeline = make_pipeline(SimpleImputer(strategy="median"))

        return ColumnTransformer([
            ("cat", hgbr_cat_pipeline, make_column_selector(dtype_include='category')),
        ],
            remainder=hgbr_default_num_pipeline,
            verbose_feature_names_out=False,
        )

    X, y = get_X_y(dataset_path)
    preprocessing_pipeline = get_preprocessing_pipeline()

    hgbr = make_pipeline(preprocessing_pipeline,
                                HistGradientBoostingRegressor(learning_rate=learning_rate,
                                                              max_depth=max_depth,
                                                              max_leaf_nodes=max_leaf_nodes,
                                                              min_samples_leaf=min_samples_leaf,
                                                              l2_regularization=l2_regularization))
    hgbr.fit(X, y)

    with open(model_output_path, "wb") as fd:
        pickle.dump(hgbr, fd)


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn processed data from (../split) into
        split data ready to be processed (saved in ../agg).
    """
    logger = logging.getLogger(__name__)
    logger.info('making train and test data set from processed data')

    output_path = Path(output_filepath)
    output_path.mkdir(parents=True, exist_ok=True)

    train(input_filepath, output_path / 'model.pkl')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
