import os
import numpy as np
import pandas as pd
from pyke import *
from kepler.preprocess import *


def main():

    # Load Cummulative KOI Table
    cummulative_table_df = pd.read_csv('C://Users//DYN//Google Drive//Intelligent_Systems_MSc//MSc_Project//data//kepler_planets//cumulative.csv')
    cummulative_table_confirmed_df = cummulative_table_df[cummulative_table_df['koi_disposition'].isin(['CONFIRMED','CANDIDATE']) ]
    cummulative_table_falsepositives_df = cummulative_table_df[cummulative_table_df['koi_disposition'].isin(['FALSE POSITIVE']) ]

    # Remove duplicates (TEMPORARY FIX FOR MULTIPLANET SYSTEMS)
    #cummulative_table_confirmed_df = cummulative_table_confirmed_df.drop_duplicates('kepid', keep=False)
    #cummulative_table_confirmed_df = cummulative_table_confirmed_df.set_index('kepid')
    #cummulative_table_falsepositives_df = cummulative_table_falsepositives_df.drop_duplicates('kepid', keep=False)
    #cummulative_table_falsepositives_df = cummulative_table_falsepositives_df.set_index('kepid')


    # COLUMN kepoi_name:     KOI Name
    # COLUMN kepler_name:    Kepler Name
    # COLUMN koi_period:     Orbital Period [days]
    # COLUMN koi_time0bk:    Transit Epoch [BKJD]
    # COLUMN koi_duration:   Transit Duration [hrs]
    # COLUMN koi_model_snr:  Transit Signal-to-Noise

    root_dict = {'confirmed': 'E://fits//confirmed//',
                 'false_positives': 'E://fits//false_positives//',
                 'uncategorized': 'E://fits//uncategorized//'}

    # Read directory files
    confirmed_fits = os.listdir(root_dict['confirmed'])
    falsepositives_fits = os.listdir(root_dict['false_positives'])
    uncategorized_fits = os.listdir(root_dict['uncategorized'])

    flattened_confirmed_fluxes, folded_confirmed_fluxes, binned_confirmed_fluxes  = LoadConfirmedFits(root_dict['confirmed'], confirmed_fits, cummulative_table_confirmed_df)
    flattened_falsepositive_fluxes, folded_falsepositive_fluxes, binned_falsepositive_fluxes_flux = LoadConfirmedFits(root_dict['false_positives'], falsepositives_fits, cummulative_table_falsepositives_df)
    #unconfirmed_flux_df = LoadUncategorizedFits(root_dict['uncategorized'], uncategorized_fits)


    # flattened_confirmed_fluxes.to_pickle('pickled_data//flattened_confirmed_candidates.pkl')
    # folded_confirmed_fluxes.to_pickle('pickled_data//folded_confirmed_candidates.pkl')
    #binned_confirmed_fluxes.to_pickle('pickled_data//binned_confirmed_candidates.pkl')

    # flattened_falsepositive_fluxes.to_pickle('pickled_data//flattened_falsepositives.pkl')
    # folded_falsepositive_fluxes.to_pickle('pickled_data//folded_falsepositives.pkl')
    #binned_falsepositive_fluxes_flux.to_pickle('pickled_data//binned_falsepositives.pkl')

    #folded_confirmed_fluxes.to_csv('folded_confirmed.csv', na_rep='nan', index=False)
    #folded_falsepositive_fluxes.to_csv('folded_falsepositives.csv', na_rep='nan', index=False)
    binned_confirmed_fluxes.to_csv('binned_confirmed.csv', na_rep='nan', index=False)
    binned_falsepositive_fluxes_flux.to_csv('binned_falsepositives.csv', na_rep='nan', index=False)


def LoadConfirmedFits(path, confirmed_fits_list, cummulative_table_confirmed_df, plot_show = True):

    # Load and Process Confirmed and Candidate Planet Light Curves
    flattened_fluxes_df = pd.DataFrame()
    folded_fluxes_df = pd.DataFrame()
    binned_fluxes_df = pd.DataFrame()

    for file in confirmed_fits_list:
        try:
            if ".fits" in file:

                # Read Original FITS into Light Curve structure
                og_lc = KeplerLightCurveFile(path=path + file)
                # print(og_lc.header())

                # Retrieve PDCSAP_FLUX data from .fits
                og_lc_pdcsap = og_lc.get_lightcurve('PDCSAP_FLUX')
                print(og_lc_pdcsap.keplerid)

                # Flatten the PDC_SAP flux signal into a detrended version
                flattened_lc = og_lc_pdcsap.flatten()

                # Detect best period
                # postlist, trial_periods, best_period = box_period_search(flattened_lc, nperiods=2000)
                # print('Best period: ', best_period)

                ids = cummulative_table_confirmed_df.loc[cummulative_table_confirmed_df['kepid'] == og_lc_pdcsap.keplerid  ]
                #ids = ids.set_index('kepid')
                rows = ids.values.shape[0]

                folded_lcs_df = pd.DataFrame()
                binned_lcs_df = pd.DataFrame()

                for lc in range(rows):

                    # Obtain period and transit centre
                    period = ids.iloc[lc]['koi_period']
                    bjk0 = ids.iloc[lc]['koi_time0bk']

                    # Other attributes
                    snr = ids.iloc[lc]['koi_model_snr']
                    duration = ids.iloc[lc]['koi_duration']
                    depth = ids.iloc[lc]['koi_depth']

                    # print('SNR:', snr)
                    # print('Duration: ', duration)
                    # print('Depth: ', depth)

                    # PYKE
                    # # Phase fold the detrended light curve
                    # folded_lc = flattened_lc.fold(period=period, phase=bjk0)
                    # # Bin the folded light curve
                    #
                    #
                    # binned_lc_global = folded_lc.bin(binsize=len(folded_lc.flux)/1001 ,method='median')

                    time, flux = phase_fold_and_sort_light_curve(flattened_lc.time, flattened_lc.flux, period, bjk0)

                    global_sequence = global_view(time, flux, period)
                    x_global = np.array(range(len(global_sequence)))

                    local_sequence = local_view(time, flux, period, duration)
                    x_local =  np.array(range(len(local_sequence)))

                    if plot_show:
                        plt.plot(time, flux, 'o', markersize=1, label='FLUX')
                        plt.show()
                        plt.plot(x_global, global_sequence, 'o', markersize=1, label='FLUX')
                        plt.show()
                        plt.plot(x_local, local_sequence, 'o', markersize=1, label='FLUX')
                        plt.show()

                    # flattened_fluxes_df = flattened_fluxes_df.append(pd.Series(flattened_lc.flux), ignore_index=True)
                    # folded_fluxes_df = folded_fluxes_df.append(pd.Series(folded_lc.flux), ignore_index=True)
                    # binned_fluxes_df = binned_fluxes_df.append(pd.Series(binned_lc_global.flux), ignore_index=True)


        except:
            print('Kepler ID: {} failed'.format(og_lc_pdcsap.keplerid))

    return flattened_fluxes_df, folded_fluxes_df, binned_fluxes_df

def LoadUncategorizedFits(path, uncategorized_fits_list, plot_show = False):

    # Load and Process Confirmed and Candidate Planet Light Curves
    uncategorized_fluxes_df = pd.DataFrame()

    for file in uncategorized_fits_list:
        try:
            if ".fits" in file:

                # Read Original FITS into Light Curve structure
                og_lc = KeplerLightCurveFile(path=path + file)
                # print(og_lc.header())

                # Retrieve PDCSAP_FLUX data from .fits
                og_lc_pdcsap = og_lc.get_lightcurve('PDCSAP_FLUX')
                print(og_lc_pdcsap.keplerid)

                # Flatten the PDC_SAP flux signal into a detrended version
                flattened_lc = og_lc_pdcsap.flatten()

                if plot_show:
                    plt.plot(flattened_lc.time, flattened_lc.flux, 'x', markersize=1, label='FLUX')
                    plt.show()

                # Add folded DET_FLUX to dataframe
                det_flux = pd.Series(flattened_lc.flux)
                uncategorized_fluxes_df = uncategorized_fluxes_df.append(det_flux, ignore_index=True)

                print('Kepler ID: {} processed succesfully.'.format(og_lc_pdcsap.keplerid))
        except:
            print('Kepler ID: {} failed gargantuously'.format(og_lc_pdcsap.keplerid))

    return uncategorized_fluxes_df


def LoadFalsePositiveFits(path, falsepositive_fits_list, cummulative_table_falsepositive_df, plot_show = False):

    # Load and Process Confirmed and Candidate Planet Light Curves
    falsepositive_fluxes_df = pd.DataFrame()

    for file in falsepositive_fits_list:
        try:
            if ".fits" in file:

                # Read Original FITS into Light Curve structure
                og_lc = KeplerLightCurveFile(path=path + file)
                # print(og_lc.header())

                # Retrieve PDCSAP_FLUX data from .fits
                og_lc_pdcsap = og_lc.get_lightcurve('PDCSAP_FLUX')
                print(og_lc_pdcsap.keplerid)

                # Flatten the PDC_SAP flux signal into a detrended version
                flattened_lc = og_lc_pdcsap.flatten()

                # Detect best period
                # postlist, trial_periods, best_period = box_period_search(flattened_lc, nperiods=2000)
                # print('Best period: ', best_period)

                # Phase fold the detrended light curve
                period = cummulative_table_falsepositive_df.get_value(og_lc_pdcsap.keplerid, 'koi_period')
                bjk0 = cummulative_table_falsepositive_df.get_value(og_lc_pdcsap.keplerid, 'koi_time0bk')

                folded_lc = flattened_lc.fold(period=period, phase=bjk0)

                # Bin the folded light curve
                binned_lc = folded_lc.bin(binsize=100, method='median')

                if plot_show:
                    plt.plot(folded_lc.time, folded_lc.flux, 'x', markersize=1, label='FLUX')
                    plt.show()

                # Add folded DET_FLUX to dataframe
                det_flux = pd.Series(folded_lc.flux)
                falsepositive_fluxes_df = falsepositive_fluxes_df.append(det_flux, ignore_index=True)
        except:
            print('Kepler ID: {} not dispositioned as Confirmed or Candidate'.format(og_lc_pdcsap.keplerid))

    return falsepositive_fluxes_df

if __name__ == "__main__":
    main()




