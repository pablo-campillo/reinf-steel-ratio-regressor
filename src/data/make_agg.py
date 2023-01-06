import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

agg_dict = {
    # General
    'foundation_type': 'last',
    'concrete_m3_total': 'sum',
    'concrete_code': 'last',
    'load_code': 'last',
    'wind_code': 'last',
    'seismic_code': 'last',
    # Alturas
    'floor_number': 'last', 
    'total_height': 'last', 
    'height': 'sum',
    'foundation_height': 'last',
    'double_height': 'sum',
    'triple_height': 'sum',
    # Soportes
    'support_column': 'sum',
    'unsupported_column': 'sum', 
    'walls_number': 'sum',
    'retaining_walls_number': 'sum',
    'support_area': 'sum',
    'walls_area': 'sum',
    'retaining_walls_area': 'sum', 
    # Cargas
    'LL': 'median', 
    'DL': 'median', 
    'LL_max': 'median',
    'DL_max': 'median', 
    'wind_load_x': 'median',
    'wind_load_y': 'median',
    'seismic_acc': 'median',
    # Forjados
    'slab_total_area': 'sum',
    'flat_slab_area': 'sum', 
    'waffle_slab_area': 'sum',
    'depth': 'median', 
    'drop_panel_area': 'sum', 
    'domes_area': 'sum',
    'domes_number': 'sum',
    'inter_axis_distance': 'last',
    # Forma
    'shape_factor': 'median',
    'x_length': 'median', 
    'y_length': 'median',
    'center_x': 'median',
    'center_y': 'median',
    'cdm_x': 'median',
    'cdm_y': 'median',
    'cdr_x': 'median',
    'cdr_y': 'median',
    # Objetivo
    'reinf_steel_total': 'sum',
}

def agg_floors(df):
    df['reinf_steel_total'] = df['reinf_steel_ratio'] * df['slab_total_area']
    df['concrete_m3_total'] = df['concrete_m3_ratio'] * df['slab_total_area']
    
    b_df = df[['project_code']+list(agg_dict.keys())].groupby(by='project_code').agg(agg_dict)
    b_df['reinf_steel_ratio'] = b_df['reinf_steel_total'] / b_df['slab_total_area']
    b_df['concrete_m3_ratio'] = b_df['concrete_m3_total'] / b_df['slab_total_area']
    del b_df['reinf_steel_total']
    del b_df['concrete_m3_total']
    b_df.reset_index(inplace=True)
    del b_df['project_code']
    return b_df


def process_file(input_filepath, output_filepath, file_name):
    df = pd.read_csv(Path(input_filepath) / file_name, sep=';', index_col=0)
    building_df = agg_floors(df)
    building_df.to_csv(Path(output_filepath) / file_name, sep=';')
    
    
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