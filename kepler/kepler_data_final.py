import pandas as pd
from pyke import *
import kepler.third_party.preprocess as preprocess

KEPLER_DATA_DIR = 'E:\\new_fits'
TCE_TABLE_DIR = 'E:\\new_fits\\q1_q17_dr24_tce.csv'

def main():

    # Load TCE Table
    tce_table = pd.read_csv(filepath_or_buffer=TCE_TABLE_DIR, comment='#' )

    tce_table["tce_duration"] /= 24  # Convert hours to days.

    # Name and values of the target column in the input TCE table to use as training labels.
    _LABEL_COLUMN = "av_training_set"
    _ALLOWED_LABELS = {"PC", "AFP", "NTP"}

    # Filter TCE table to allowed labels.
    allowed_tces = tce_table[_LABEL_COLUMN].apply(lambda l: l in _ALLOWED_LABELS)
    tce_table = tce_table[allowed_tces]
    num_tces = len(tce_table)

    # Randomly shuffle the TCE table.
    np.random.seed(123)
    tce_table = tce_table.iloc[np.random.permutation(num_tces)]

    neg = ['AFP', 'NTP']
    tce_table = tce_table.loc[tce_table['av_training_set'].isin(neg)]


    # positive_flattened_df, positive_folded_df, positive_globalbinned_df, positive_localbinned_df = generate_tce_data(tce_table, positive_class=True, plot_show=False)
    # positive_globalbinned_df.to_csv('csv_data//positives_globalbinned.csv', na_rep='nan', index=False)
    # positive_localbinned_df.to_csv('csv_data//positives_localbinned.csv', na_rep='nan', index=False)
    # print('Succesfully processed and saved positive samples.')
    # print('\n')

    negative_flattened_df, negative_folded_df, negative_globalbinned_df, negative_localbinned_df = generate_tce_data(tce_table[0:3500], positive_class=False)
    negative_globalbinned_df.to_csv('csv_data//negatives_globalbinned1.csv', na_rep='nan', index=False)
    negative_localbinned_df.to_csv('csv_data//negatives_localbinned1.csv', na_rep='nan', index=False)
    print('Succesfully processed and saved negative samples. (1)')
    print('\n')

    negative_flattened_df, negative_folded_df, negative_globalbinned_df, negative_localbinned_df = generate_tce_data(tce_table[3501:7000], positive_class=False)
    negative_globalbinned_df.to_csv('csv_data//negatives_globalbinned2.csv', na_rep='nan', index=False)
    negative_localbinned_df.to_csv('csv_data//negatives_localbinned2.csv', na_rep='nan', index=False)
    print('Succesfully processed and saved negative samples. (2)')
    print('\n')

    negative_flattened_df, negative_folded_df, negative_globalbinned_df, negative_localbinned_df = generate_tce_data(tce_table[7001:10000], positive_class=False)
    negative_globalbinned_df.to_csv('csv_data//negatives_globa lbinned3.csv', na_rep='nan', index=False)
    negative_localbinned_df.to_csv('csv_data//negatives_localbinned3.csv', na_rep='nan', index=False)
    print('Succesfully processed and saved negative samples. (3)')
    print('\n')

    negative_flattened_df, negative_folded_df, negative_globalbinned_df, negative_localbinned_df = generate_tce_data(
        tce_table[10001:], positive_class=False)
    negative_globalbinned_df.to_csv('csv_data//negatives_globalbinned4.csv', na_rep='nan', index=False)
    negative_localbinned_df.to_csv('csv_data//negatives_localbinned4.csv', na_rep='nan', index=False)
    print('Succesfully processed and saved negative samples. (4)')
    print('\n')



    # positive_globalbinned_df.to_pickle('pickled_data//positives_globalbinned.pkl')
    # positive_localbinned_df.to_pickle('pickled_data//positives_localbinned.pkl')
    # negative_globalbinned_df.to_pickle('pickled_data//negatives_globalbinned.pkl')
    # negative_localbinned_df.to_pickle('pickled_data//negatives_localbinned.pkl')


def generate_tce_data(tce_table, positive_class, plot_show=False):

    # if positive_class:
    #     tce_table = tce_table.loc[tce_table['av_training_set'] == 'PC']
    # else:
    #     neg = ['AFP','NTP']
    #     tce_table = tce_table.loc[tce_table['av_training_set'].isin(neg)]

    # Load and Process Confirmed and Candidate Planet Light Curves
    flattened_fluxes_df = pd.DataFrame()
    folded_fluxes_df = pd.DataFrame()
    globalbinned_fluxes_df = pd.DataFrame()
    localbinned_fluxes_df = pd.DataFrame()


    num_tces = len(tce_table)
    processed_count = 0
    failed_count = 0

    for _, tce in tce_table.iterrows():

        try:
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

            # Append retrieved and processed fluxes to output dataframes
            # flattened_fluxes_df = flattened_fluxes_df.append(pd.Series(flattened_flux), ignore_index=True)
            # folded_fluxes_df = folded_fluxes_df.append(pd.Series(folded_flux), ignore_index=True)
            globalbinned_fluxes_df = globalbinned_fluxes_df.append(pd.Series(global_view), ignore_index=True)
            localbinned_fluxes_df = localbinned_fluxes_df.append(pd.Series(local_view), ignore_index=True)

            print('Kepler ID: {} processed'.format(tce.kepid))
            print("Processed Percentage: ", ((processed_count + failed_count) / num_tces) * 100, "%")

            processed_count += 1
        except:
            #print('Kepler ID: {} failed'.format(tce.kepid))
            failed_count += 1





    return flattened_fluxes_df, folded_fluxes_df, globalbinned_fluxes_df, localbinned_fluxes_df



def process_tce(tce):
  """Processes the light curve for a Kepler TCE and returns an Example proto.

  Args:
    tce: Row of the input TCE table.

  Returns:
    A tensorflow.train.Example proto containing TCE features.

  Raises:
    IOError: If the light curve files for this Kepler ID cannot be found.
  """
  # Read and process the light curve.
  time, flattened_flux = preprocess.read_and_process_light_curve(tce.kepid, KEPLER_DATA_DIR)

  time, folded_flux = preprocess.phase_fold_and_sort_light_curve(time, flattened_flux, tce.tce_period, tce.tce_time0bk)

  # Generate the local and global views.
  global_view = preprocess.global_view(time, folded_flux, tce.tce_period, num_bins=2001, bin_width_factor=1 / 2001)
  local_view = preprocess.local_view(time, folded_flux, tce.tce_period,
                                     tce.tce_duration, num_bins=201, bin_width_factor=0.16, num_durations=4)

  return flattened_flux, folded_flux, global_view, local_view



if __name__ == "__main__":
    main()




