from pathlib import Path
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm  # For progress bar

import src.helpers.third_party.preprocess as preprocess
from definitions import OUTPUT_DIR, TCE_TABLE_DIR, KEPLER_DATA_DIR


def process_tce(tce):
    """Processes the light curve for a Kepler TCE and returns processed data."""
    time, flattened_flux = preprocess.read_and_process_light_curve(tce.kepid, KEPLER_DATA_DIR)
    time, folded_flux = preprocess.phase_fold_and_sort_light_curve(time, flattened_flux, tce.tce_period, tce.tce_time0bk)

    # Generate local and global views
    local_view = preprocess.local_view(time, folded_flux, tce.tce_period, tce.tce_duration, num_bins=201,
                                       bin_width_factor=0.16, num_durations=4)
    global_view = preprocess.global_view(time, folded_flux, tce.tce_period, num_bins=2001, bin_width_factor=1 / 2001)

    return flattened_flux, folded_flux, local_view, global_view


def generate_tce_data(tce_table):
    """Processes TCE data and returns DataFrames for flattened, folded, globalbinned, and localbinned fluxes."""
    # Initialize lists to accumulate data
    flattened_fluxes_list = []
    folded_fluxes_list = []
    globalbinned_fluxes_list = []
    localbinned_fluxes_list = []

    # Use tqdm for a progress bar
    total_tces = len(tce_table)
    with tqdm(total=total_tces, desc="Processing TCEs", unit="tce") as pbar:
        for idx, tce in tce_table.iterrows():
            try:
                flattened_flux, folded_flux, local_view, global_view = process_tce(tce)

                # Append processed data to corresponding lists
                flattened_fluxes_list.append(flattened_flux)
                folded_fluxes_list.append(folded_flux)
                globalbinned_fluxes_list.append(global_view)
                localbinned_fluxes_list.append(local_view)

                pbar.update(1)

            except FileNotFoundError as fnf_error:
                logging.error(f"Row ID: {idx} failed due to missing file: {str(fnf_error)}")
            except Exception as e:
                logging.error(f"Row ID: {idx} failed due to: {str(e)}")

    # Convert lists to DataFrames and return
    flattened_fluxes_df = pd.DataFrame(flattened_fluxes_list)
    folded_fluxes_df = pd.DataFrame(folded_fluxes_list)
    globalbinned_fluxes_df = pd.DataFrame(globalbinned_fluxes_list)
    localbinned_fluxes_df = pd.DataFrame(localbinned_fluxes_list)

    return flattened_fluxes_df, folded_fluxes_df, globalbinned_fluxes_df, localbinned_fluxes_df


def save_dataframe(df, name):
    """Helper function to save DataFrame to CSV and pickle formats."""
    csv_path = Path(OUTPUT_DIR) / f'{name}.csv'
    pkl_path = Path(OUTPUT_DIR) / f'{name}.pkl'

    df.to_csv(csv_path, na_rep='nan', index=False)
    df.to_pickle(pkl_path)

    logging.info(f"Saved {name} to {csv_path} and {pkl_path}.")


def main():
    """Runs data processing scripts to turn raw data into cleaned data."""
    logger = logging.getLogger(__name__)
    logger.info('Starting data processing')

    # Load TCE Table
    tce_table = pd.read_csv(TCE_TABLE_DIR, comment='#')
    tce_table["tce_duration"] /= 24  # Convert hours to days

    # Filter allowed labels
    _LABEL_COLUMN = "av_training_set"
    _ALLOWED_LABELS = {"PC", "AFP", "NTP"}
    allowed_tces = tce_table[_LABEL_COLUMN].apply(lambda l: l in _ALLOWED_LABELS)
    tce_table = tce_table[allowed_tces]

    # Randomly shuffle the TCE table
    np.random.seed(123)
    tce_table = tce_table.iloc[np.random.permutation(len(tce_table))].reset_index(drop=True)

    # Split into positive and negative samples
    positive_tce_table = tce_table[tce_table[_LABEL_COLUMN] == "PC"]
    negative_tce_table = tce_table[tce_table[_LABEL_COLUMN].isin(["AFP", "NTP"])]

    # Process positive and negative TCE samples
    logger.info('Processing positive samples...')
    positive_flattened_df, positive_folded_df, positive_globalbinned_df, positive_localbinned_df = generate_tce_data(
        positive_tce_table)

    logger.info('Processing negative samples...')
    negative_flattened_df, negative_folded_df, negative_globalbinned_df, negative_localbinned_df = generate_tce_data(
        negative_tce_table)

    # Label dataframes
    for df in [positive_flattened_df, positive_folded_df, positive_globalbinned_df, positive_localbinned_df]:
        df['LABEL'] = 1

    for df in [negative_flattened_df, negative_folded_df, negative_globalbinned_df, negative_localbinned_df]:
        df['LABEL'] = 0

    # Merge positive and negative datasets
    final_flattened_df = pd.concat([positive_flattened_df, negative_flattened_df], ignore_index=True)
    final_folded_df = pd.concat([positive_folded_df, negative_folded_df], ignore_index=True)
    final_globalbinned_df = pd.concat([positive_globalbinned_df, negative_globalbinned_df], ignore_index=True)
    final_localbinned_df = pd.concat([positive_localbinned_df, negative_localbinned_df], ignore_index=True)

    # Save the final datasets
    save_dataframe(final_flattened_df, 'final_flattened')
    save_dataframe(final_folded_df, 'final_folded')
    save_dataframe(final_globalbinned_df, 'final_globalbinned')
    save_dataframe(final_localbinned_df, 'final_localbinned')

    logger.info('Data processing completed and datasets saved.')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
