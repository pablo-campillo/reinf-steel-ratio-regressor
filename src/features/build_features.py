import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def add_features(df):
    # Alturas
    df['slab_area_mean'] = df['slab_total_area'] / df['floor_number']
    df['height_mean'] = df['height'] / df['floor_number']
    df['double_height_ratio'] = df['double_height'] / df['slab_total_area']
    df['triple_height_ratio'] = df['triple_height'] / df['slab_total_area']
    df['slenderness'] = df['slab_area_mean'] / df['total_height']
    
    # Soportes
    df['column_area'] = df['support_area'] - df['walls_area']
    df['swall_num'] = df['walls_number'] - df['retaining_walls_number']
    df['swall_area'] = df['walls_area'] - df['retaining_walls_area']

    df['swall_area_mean'] = df['swall_area'] / df['swall_num']
    df['col_area_mean'] = df['column_area'] / df['support_column']
    df['rwalls_area_mean'] = df['retaining_walls_area'] / df['retaining_walls_number']
    
    df['swall_area_mean_m2'] = df['swall_area_mean'] / df['slab_area_mean']
    df['col_area_mean_m2'] = df['col_area_mean'] / df['slab_area_mean']
    df['rwalls_area_mean_m2'] = df['rwalls_area_mean'] / df['slab_area_mean']
    
    df.loc[:, ['rwalls_area_mean', 'rwalls_area_mean_m2']] = df.loc[:, ['rwalls_area_mean', 'rwalls_area_mean_m2']].fillna(0)
    
    df['swall_area_ratio'] = df['swall_area'] / df['slab_total_area']
    df['col_area_ratio'] = df['column_area'] / df['slab_total_area']
    df['rwals_area_ratio'] = df['retaining_walls_area'] / df['slab_total_area']
    
    # Cargas
    df['LL_m2'] = df['LL'] / df['slab_area_mean']
    df['DL_m2'] = df['DL'] / df['slab_area_mean']
    df['LL_max_m2'] = df['LL_max'] / df['slab_area_mean']
    df['DL_max_m2'] = df['DL_max'] / df['slab_area_mean']
    df['wind_load_x_m2'] = df['wind_load_x'] / df['slab_area_mean']
    df['wind_load_y_m2'] = df['wind_load_y'] / df['slab_area_mean']
    df['seismic_acc_m2'] = df['seismic_acc'] / df['slab_area_mean']

    df['wind_load_x_h'] = df['wind_load_x'] / df['slenderness']
    df['wind_load_y_h'] = df['wind_load_y'] / df['slenderness']

    # Forjados
    df['flat_slab_area_ratio'] = df['flat_slab_area'] / df['slab_total_area']
    df['waffle_slab_area_ratio'] = df['waffle_slab_area'] / df['slab_total_area']
    df['drop_panel_area_ratio'] = df['drop_panel_area'] / df['slab_total_area']
    df['domes_area_ratio'] = (df['domes_area'] / df['slab_total_area'])

    df['drop_panel_area_support'] = df['drop_panel_area'] / (df['support_column'] + df['swall_num'])
    df['drop_panel_area_support_m2'] = df['drop_panel_area_support'] / df['slab_area_mean']

    # Forma
    df['eccentricity'] = ((df['cdm_x'] - df['cdr_x']) * (df['cdm_x'] - df['cdr_x']) + 
                      (df['cdm_y'] - df['cdr_y']) * (df['cdm_y'] - df['cdr_y']))**0.5
    df['eccentricity_m2'] = df['eccentricity'] / df['slab_area_mean']


def process_file(input_filepath, output_filepath, file_name):
    df = pd.read_csv(Path(input_filepath) / file_name, sep=';', index_col=0)
    add_features(df)
    df.to_csv(Path(output_filepath) / file_name, sep=';')
    
    
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
    
    process_file(input_filepath, output_filepath, 'train.csv')
    process_file(input_filepath, output_filepath, 'test.csv')
    

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()