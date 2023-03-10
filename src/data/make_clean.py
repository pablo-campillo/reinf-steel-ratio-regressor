# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path

import yaml
from dotenv import find_dotenv, load_dotenv
import pandas as pd


def fix_project_data(df):
    df.loc[df['floor_code'] == 'PYAS17020_C00_I00_H00_Q00_R00_V00_S00_O01_F1', 'slab_total_area'] = 73.74
    df.loc[df['floor_code'] == 'PYAS17020_C00_I00_H00_Q00_R00_V00_S00_O01_F1', 'flat_slab_area'] = 73.74 
    df.loc[df['floor_code'] == 'PYAS17020_C00_I00_H00_Q00_R00_V00_S00_O01_F1', 'ignored'] = 'NO'
    df.loc[df['ignored'].isnull(), 'ignored'] = 'NO'

    
def remove_ignored_floors(df):
    return df[df['ignored'] != 'YES']


def get_non_reinf_steel_ratio_project_codes(df):
    aux = df.groupby(by='project_code')['reinf_steel_ratio'].sum()
    return aux[aux == 0].index.tolist()


def get_non_slab_total_area(df):
    return df[df['slab_total_area'] <= 0]['project_code'].unique().tolist()


def get_project_too_much_kg_m2(df):
    df['total_reinf_steel'] = df['reinf_steel_ratio'] * df['slab_total_area']
    aux = df.groupby(by='project_code')[['total_reinf_steel', 'slab_total_area']].sum()
    aux['reinf_steel_ratio'] = aux['total_reinf_steel'] / aux['slab_total_area']
    aux.reset_index(inplace=True)
    return aux[aux['reinf_steel_ratio'] > 36.7]['project_code'].tolist()


def get_project_all_flat_slabs(df):
    aux = df.groupby(by='project_code')['waffle_slab_area'].sum()
    return aux[aux == 0].index.tolist()


def remove_project_code_samples(df, project_codes):
    return df.loc[~df['project_code'].isin(project_codes), :]

DH_M2_THRESHOLD = 0.2
TH_M2_THRESHOLD = 0.2

def fix_double_height(df):
    df['double_height'] = df.groupby(by='project_code')['slab_total_area'].shift(-1) - df['slab_total_area']
    df.loc[df['double_height'] < 0, 'double_height'] = 0
    df['double_height'] = df['double_height'].fillna(0)
    df['double_height_m2'] = df['double_height'] / df['slab_total_area']
    df.loc[df['double_height_m2'] < DH_M2_THRESHOLD, 'double_height'] = 0
    df['double_height_m2'] = df['double_height'] / df['slab_total_area']
    
def fix_triple_height(df):
    df['triple_height'] = df.groupby(by='project_code')['slab_total_area'].shift(-2) - df['slab_total_area']
    df.loc[df['triple_height'] < 0, 'triple_height'] = 0
    df['triple_height'] = df['triple_height'].fillna(0)
    df['triple_height_m2'] = df['triple_height'] / df['slab_total_area']
    df.loc[df['triple_height_m2'] < DH_M2_THRESHOLD, 'triple_height'] = 0
    df['triple_height_m2'] = df['triple_height'] / df['slab_total_area']


def fix_cdm_cdr(df):    
    df['eccentricity'] = ((df['cdm_x'] - df['cdr_x'])*(df['cdm_x'] - df['cdr_x']) + 
                      (df['cdm_y'] - df['cdr_y']) * (df['cdm_y'] - df['cdr_y']))**0.5
    cog_cm_cond = (df['double_height'] < 10) & (df['double_height'] > 40)
    df.loc[cog_cm_cond, 'cdm_x'] /= 100
    df.loc[cog_cm_cond, 'cdm_y'] /= 100
    df.loc[cog_cm_cond, 'cdr_x'] /= 100
    df.loc[cog_cm_cond, 'cdr_y'] /= 100

    
FEATURES = [
     'floor_name',
     'project_code',
     'floor_code',
     'slab_type',
     'depth',
     'floor_number',
     'total_height',
     'level',
     'height',
     'foundation_height',
     'support_column',
     'unsupported_column',
     'walls_number',
     'retaining_walls_number',
     'inter_axis_distance',
     'LL',
     'DL',
     'LL_max',
     'DL_max',
     'wind_load_x',
     'wind_load_y',
     'seismic_acc',
     'double_height',
     'flat_slab_area',
     'waffle_slab_area',
     'drop_panel_area',
     'domes_area',
     'support_area',
     'shape_factor',
     'x_length',
     'y_length',
     'stair_area',
     'slope_area',
     'walls_area',
     'retaining_walls_area',
     'foundation_type',
     'concrete_m3_ratio',
     'reinf_steel_ratio',
     'domes_number',
     'deflection_ratio',
     'elasticiti_modulii',
     'cdm_x',
     'cdm_y',
     'cdr_x',
     'cdr_y',
     'concrete_code',
     'load_code',
     'wind_code',
     'seismic_code',
     'support_mean_span_dist',
     'support_max_span_dist',
     'support_min_span_dist',
     'cantilever_mean_dist',
     'cantilever_max_dist',
     'cantilever_min_dist',
     'cantilever_area',
     'span_dists',
     'cantilever_dists',
     'center_x',
     'center_y',
     'slab_total_area',
     'triple_height'
]


def select_originals(df):
    ori_cond1 = df['project_code'].str.contains('_C00_I00_H00_Q00_R00_V00_S00_O01')
    ori_cond2 = df['project_code'].str.contains('_C00_I00_H00_Q00_R00_V00_S00_O01')
    return df[ori_cond1 & ori_cond2]


def remove_overground_floors(df):
    return df[df['level'] > 0]


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    params = yaml.safe_load(open("params.yaml"))["clean"]
    originals = bool(params['originals'])
    overground = bool(params['overground'])

    df = pd.read_csv(Path(input_filepath) / 'structure_projects.csv', sep=';')
    if originals:
        df = select_originals(df)
    if overground:
        df = remove_overground_floors(df)

    logger.info(f"Input Number of samples: {len(df)}")
    logger.info(f"Input Number of projects: {len(df['project_code'].unique())}")
    
    fix_project_data(df)
    
    df = remove_ignored_floors(df)
    
    to_remove_codes = get_non_reinf_steel_ratio_project_codes(df) + get_non_slab_total_area(df) + get_project_too_much_kg_m2(df) + get_project_all_flat_slabs(df)
    
    to_remove_codes += ['PYAS20020_C00_I00_H00_Q00_R00_V00_S00_O02']
    
    processed_df = remove_project_code_samples(df, to_remove_codes).copy()
    
    fix_double_height(processed_df)
    fix_triple_height(processed_df)
    fix_cdm_cdr(processed_df)
    
    logger.info(f"Number of samples removed: {len(df) - len(processed_df)}")
    logger.info(f"Number of projects removed: {len(df['project_code'].unique()) - len(processed_df['project_code'].unique())}")
    
    output_path = Path(output_filepath)
    output_path.mkdir(parents=True, exist_ok=True)
    processed_df[FEATURES].to_csv(output_path / 'structure_projects.csv', sep=';')
    

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
