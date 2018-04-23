import os
import numpy as np
import pandas as pd
from pyke import *

def main():

    # Load Cummulative KOI Table
    cummulative_table_df = pd.read_csv('C://Users//DYN//Google Drive//Intelligent_Systems_MSc//MSc_Project//data//kepler_planets//cumulative.csv')
    cummulative_table_confirmed_df = cummulative_table_df[cummulative_table_df['koi_disposition'].isin(['CONFIRMED','CANDIDATE']) ]
    cummulative_table_falsepositives_df = cummulative_table_df[cummulative_table_df['koi_disposition'].isin(['FALSE POSITIVE']) ]

    # Remove duplicates (TEMPORARY FIX FOR MULTIPLANET SYSTEMS)
    cummulative_table_confirmed_df = cummulative_table_confirmed_df.drop_duplicates('kepid')
    cummulative_table_confirmed_df = cummulative_table_confirmed_df.set_index('kepid')
    cummulative_table_falsepositives_df = cummulative_table_falsepositives_df.drop_duplicates('kepid')
    cummulative_table_falsepositives_df = cummulative_table_falsepositives_df.set_index('kepid')


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

    #confirmed_fluxes_df = LoadConfirmedFits(root_dict['confirmed'], confirmed_fits, cummulative_table_confirmed_df)
    #falsepositive_flux_df = LoadConfirmedFits(root_dict['false_positives'], falsepositives_fits, cummulative_table_falsepositives_df)
    unconfirmed_flux_df = LoadUncategorizedFits(root_dict['uncategorized'], uncategorized_fits)


    #confirmed_fluxes_df.to_csv('confirmed_candidates.csv', na_rep='nan', index=False)
    unconfirmed_flux_df.to_csv('unconfirmed.csv', na_rep='nan', index=False)



def LoadConfirmedFits(path, confirmed_fits_list, cummulative_table_confirmed_df, plot_show = False):

    # Load and Process Confirmed and Candidate Planet Light Curves
    confirmed_fluxes_df = pd.DataFrame()

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

                # Phase fold the detrended light curve
                period = cummulative_table_confirmed_df.get_value(og_lc_pdcsap.keplerid, 'koi_period')
                bjk0 = cummulative_table_confirmed_df.get_value(og_lc_pdcsap.keplerid, 'koi_time0bk')

                folded_lc = flattened_lc.fold(period=period, phase=bjk0)

                # Bin the folded light curve
                binned_lc = folded_lc.bin(binsize=100, method='median')

                if plot_show:
                    plt.plot(folded_lc.time, folded_lc.flux, 'x', markersize=1, label='FLUX')
                    plt.show()

                # Add folded DET_FLUX to dataframe
                det_flux = pd.Series(folded_lc.flux)
                confirmed_fluxes_df = confirmed_fluxes_df.append(det_flux, ignore_index=True)
        except:
            print('Kepler ID: {} not dispositioned as Confirmed or Candidate'.format(og_lc_pdcsap.keplerid))

    return confirmed_fluxes_df

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




