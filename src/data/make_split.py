import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

def get_split_steel_stratified_by_project_code(df):
    stats = df['reinf_steel_ratio'].describe()
    bins = [0,stats['25%'],stats['50%'],stats['75%'],1000]
    labels = ['LIGHT', 'LIGHT-MED', 'MED-HEAVY', 'HEAVY']
    df['reinf_steel_ratio_cat'] = pd.cut(df['reinf_steel_ratio'], bins=bins, labels=labels)

    print(df[['reinf_steel_ratio_cat', 'project_code']].info())
    
    sgkf = StratifiedGroupKFold(n_splits=10, random_state=43, shuffle=True)
    train, test = next(sgkf.split(df, df['reinf_steel_ratio_cat'], groups=df['project_code']))
    
    train_df = df.iloc[train, :]
    test_df = df.iloc[test, :]
    return train_df, test_df


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn processed data from (../processed) into
        split data ready to be processed (saved in ../split).
    """
    logger = logging.getLogger(__name__)
    logger.info('making train and test data set from processed data')
    
    df = pd.read_csv(Path(input_filepath) / 'structure_projects.csv', sep=';', index_col=0)
    logger.info(f"Input Number of samples: {len(df)}")
    logger.info(f"Input Number of projects: {len(df['project_code'].unique())}")
    
    train_df, test_df = get_split_steel_stratified_by_project_code(df)
    n_ori_train_projects = train_df.groupby(by='project_code')['project_code'].last().str.contains('_C00_I00_H00_Q00_R00_V00_S00_O01').sum()
    n_ori_test_projects = test_df.groupby(by='project_code')['project_code'].last().str.contains('_C00_I00_H00_Q00_R00_V00_S00_O01').sum()
    logger.info(f"Number of original project at train dataset: {n_ori_train_projects}")
    logger.info(f"Number of original project at test dataset: {n_ori_test_projects}")
    
    output_path = Path(output_filepath)
    output_path.mkdir(parents=True, exist_ok=True)
    
    train_df.to_csv(Path(output_filepath) / 'train.csv', sep=';')
    test_df.to_csv(Path(output_filepath) / 'test.csv', sep=';')
    

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()