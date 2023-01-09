# Created by Pablo Campillo at 8/1/23
import json
import logging
import pickle
from pathlib import Path

import click
from dotenv import find_dotenv, load_dotenv
from sklearn.metrics import mean_absolute_percentage_error, r2_score

from models.utils import get_X_y


def evaluate(model_filepath, dataset_filepath, metrics_filepath):
    X_train, y_train = get_X_y(dataset_filepath / 'train.csv')
    X_test, y_test = get_X_y(dataset_filepath / 'test.csv')

    final_model = pickle.load(open(model_filepath, 'rb'))

    train_predictions = final_model.predict(X_train)
    test_predictions = final_model.predict(X_test)

    train_mape = mean_absolute_percentage_error(y_train, train_predictions)
    train_r2_score = r2_score(y_train, train_predictions)
    test_mape = mean_absolute_percentage_error(y_test, test_predictions)
    test_r2_score = r2_score(y_test, test_predictions)

    result = {
        'train': {
            'mape': train_mape,
            'r2': train_r2_score,
        },
        'test': {
            'mape': test_mape,
            'r2': test_r2_score,
        },
    }

    metrics = json.dumps(result, indent=4)

    with (metrics_filepath / 'metrics.json').open("w") as outfile:
        outfile.write(metrics)


@click.command()
@click.argument('model_filepath', type=click.Path(exists=True))
@click.argument('train_dataset_filepath', type=click.Path(exists=True))
@click.argument('metrics_filepath', type=click.Path())
def main(model_filepath, train_dataset_filepath, metrics_filepath):
    """ Runs data processing scripts to turn processed data from (../split) into
        split data ready to be processed (saved in ../agg).
    """
    logger = logging.getLogger(__name__)
    logger.info('making train and test data set from processed data')

    metrics_filepath = Path(metrics_filepath)
    metrics_filepath.mkdir(parents=True, exist_ok=True)

    evaluate(Path(model_filepath), Path(train_dataset_filepath), metrics_filepath)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
