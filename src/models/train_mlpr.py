import logging
import pickle
from pathlib import Path

import click
import yaml
from dotenv import find_dotenv, load_dotenv
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from models.utils import get_X_y

MAX_ITER = 10_000

def train(dataset_path:str, model_output_path: str):
    params = yaml.safe_load(open("params.yaml"))["train"]['mlpr']
    batch_size = params['batch_size']
    hidden_layer_sizes = params['hidden_layer_sizes']
    learning_rate_init = params['learning_rate_init']

    def get_preprocessing_pipeline() -> ColumnTransformer:
        cat_pipeline = make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore"))
        default_num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())

        return ColumnTransformer(
            [
                ("cat", cat_pipeline, make_column_selector(dtype_include='category')),
            ],
            remainder=default_num_pipeline)

    X, y = get_X_y(dataset_path)
    preprocessing_pipeline = get_preprocessing_pipeline()

    mlpr = make_pipeline(preprocessing_pipeline, MLPRegressor(early_stopping=True,
                                                              batch_size=batch_size,
                                                              hidden_layer_sizes=hidden_layer_sizes,
                                                              learning_rate_init=learning_rate_init,
                                                              random_state=42,
                                                              max_iter=MAX_ITER))
    mlpr.fit(X, y)

    with open(model_output_path, "wb") as fd:
        pickle.dump(mlpr, fd)
        pickle.dump(mlpr, fd)


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
