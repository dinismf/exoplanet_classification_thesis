# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import matplotlib.pyplot as plt
import src.third_party.preprocess as preprocess

# Path to raw data and TCE table
KEPLER_DATA_DIR = 'E:\\new_fits'
TCE_TABLE_DIR = 'E:\\new_fits\\q1_q17_dr24_tce.csv'


def generate_tce_data(tce_table, plot_show=False):


    # Initialise dataframes to populate with processed data
    flattened_fluxes_df = pd.DataFrame()
    folded_fluxes_df = pd.DataFrame()
    globalbinned_fluxes_df = pd.DataFrame()
    localbinned_fluxes_df = pd.DataFrame()

    # Processing metrics
    num_tces = len(tce_table)
    processed_count = 0
    failed_count = 0


    # Iterate over every TCE in the table
    for _, tce in tce_table.iterrows():

        try:

            # Process the TCE and retrieve the processed data.
            flattened_flux, folded_flux, global_view, local_view = process_tce(tce)

            if plot_show:
                plt.plot(range(len(flattened_flux)), flattened_flux, 'o', markersize=1, label='FLUX')
                plt.show()
                plt.plot(range(len(folded_flux)), folded_flux, 'o', markersize=1, label='FLUX')
                plt.show()
                plt.plot(range(len(global_view)), global_view, 'o', markersize=1, label='FLUX')
                plt.show()
                plt.plot(range(len(local_view)), local_view, 'o', markersize=1, label='FLUX')
                plt.show()

            # Append processed flux light curves for each TCE to output dataframes.
            flattened_fluxes_df = flattened_fluxes_df.append(pd.Series(flattened_flux), ignore_index=True)
            folded_fluxes_df = folded_fluxes_df.append(pd.Series(folded_flux), ignore_index=True)
            globalbinned_fluxes_df = globalbinned_fluxes_df.append(pd.Series(global_view), ignore_index=True)
            localbinned_fluxes_df = localbinned_fluxes_df.append(pd.Series(local_view), ignore_index=True)

            print('Kepler ID: {} processed'.format(tce.kepid))
            print("Processed Percentage: ", ((processed_count + failed_count) / num_tces) * 100, "%")

            processed_count += 1
        except:
            print('Kepler ID: {} failed'.format(tce.kepid))
            failed_count += 1

    return flattened_fluxes_df, folded_fluxes_df, globalbinned_fluxes_df, localbinned_fluxes_df



def process_tce(tce):
  """Processes the light curve for a Kepler TCE and returns processed data

  Args:
    tce: Row of the input TCE table.

  Returns:
    Processed TCE data at each stage (flattening, folding, binning).

  Raises:
    IOError: If the light curve files for this Kepler ID cannot be found.
  """
  # Read and process the light curve.
  time, flattened_flux = preprocess.read_and_process_light_curve(tce.kepid, KEPLER_DATA_DIR)

  time, folded_flux = preprocess.phase_fold_and_sort_light_curve(time, flattened_flux, tce.tce_period, tce.tce_time0bk)

  # Generate the local and global views.
  global_view = preprocess.global_view(time, folded_flux, tce.tce_period, num_bins=2001, bin_width_factor=1 / 2001)

  return flattened_flux, folded_flux, global_view



@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    # Load TCE Table
    tce_table = pd.read_csv(filepath_or_buffer=TCE_TABLE_DIR, comment='#')

    tce_table["tce_duration"] /= 24  # Convert hours to days.

    # Name of the target column and labels to use as training labels.
    _LABEL_COLUMN = "av_training_set"
    _ALLOWED_LABELS = {"PC", "AFP", "NTP"}

    # Discard other labels from TCE table other than the allowed labels.
    allowed_tces = tce_table[_LABEL_COLUMN].apply(lambda l: l in _ALLOWED_LABELS)
    tce_table = tce_table[allowed_tces]
    num_tces = len(tce_table)

    # Randomly shuffle the TCE table.
    np.random.seed(123)
    tce_table = tce_table.iloc[np.random.permutation(num_tces)]

    # Split positive and negative TCE samples
    neg = ['AFP', 'NTP']
    positive_tce_table = tce_table.loc[tce_table['av_training_set'] == "PC"]
    negative_tce_table = tce_table.loc[tce_table['av_training_set'].isin(neg)]

    # Process the TCE tables
    positive_flattened_df, positive_folded_df, positive_globalbinned_df, positive_localbinned_df = generate_tce_data(positive_tce_table)
    logger.info('Succesfully processed positive TCE samples.')

    negative_flattened_df, negative_folded_df, negative_globalbinned_df, negative_localbinned_df = generate_tce_data(negative_tce_table)
    logger.info('Succesfully processed negative TCE samples.')



    # Store processed data
    positive_flattened_df.to_csv(output_filepath + 'positives_flattened.csv', na_rep='nan', index=False)
    positive_folded_df.to_csv(output_filepath + 'positives_folded.csv', na_rep='nan', index=False)
    positive_globalbinned_df.to_csv(output_filepath + 'positives_globalbinned.csv', na_rep='nan', index=False)
    positive_localbinned_df.to_csv(output_filepath + 'positives_localbinned.csv', na_rep='nan', index=False)

    positive_flattened_df.to_pickle(output_filepath + 'positives_flattened.pkl')
    positive_folded_df.to_pickle(output_filepath + 'positives_folded.pkl')
    positive_globalbinned_df.to_pickle(output_filepath + 'positives_globalbinned.pkl')
    positive_localbinned_df.to_pickle(output_filepath + 'positives_localbinned.pkl')

    logger.info('Succesfully saved positive TCE samples.')


    negative_flattened_df.to_csv(output_filepath + 'negatives_flattened.csv', na_rep='nan', index=False)
    negative_folded_df.to_csv(output_filepath + 'negatives_folded.csv', na_rep='nan', index=False)
    negative_globalbinned_df.to_csv(output_filepath + 'negatives_globalbinned.csv', na_rep='nan', index=False)
    negative_localbinned_df.to_csv(output_filepath + 'negatives_localbinned.csv', na_rep='nan', index=False)

    negative_flattened_df.to_pickle(output_filepath + 'negatives_flattened.pkl')
    negative_folded_df.to_pickle(output_filepath + 'negatives_folded.pkl')
    negative_globalbinned_df.to_pickle(output_filepath + 'negatives_globalbinned.pkl')
    negative_localbinned_df.to_pickle(output_filepath + 'negatives_localbinned.pkl')

    logger.info('Succesfully saved negative TCE samples.')

    # Label dataframes
    positive_flattened_df['LABEL'] = 1
    positive_folded_df['LABEL'] = 1
    positive_globalbinned_df['LABEL'] = 1
    positive_localbinned_df['LABEL'] = 1
    negative_flattened_df['LABEL'] = 0
    negative_folded_df['LABEL'] = 0
    negative_globalbinned_df['LABEL'] = 0
    negative_localbinned_df['LABEL'] = 0

    # TODO Merge dataframes into final and TEST
    final_globalbinned_df = negative_globalbinned_df.append(positive_globalbinned_df)



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
