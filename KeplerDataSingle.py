import kplr
import numpy as np
import pandas as pd

def main():
    # Initialise the Kplr API
    client = kplr.API()

    # Obtain Kepler planet using its given Kepler name (Confirmed)
    kepler_name = "CoRoT-1 b"
    kepID = 757076
    final_df = pd.DataFrame(retrieveLightCurve(client, kepID, isStar=True)).transpose()

    # Output the final dataframe to .csv
    final_df.to_csv('C://Users//DYN//Google Drive//Intelligent_Systems_MSc//MSc_Project//data//main//single.csv', na_rep='nan', index=False)


def retrieveLightCurve(client, id, isStar = False,  candence_flag = False):


    if (isStar):
        star = client.star(id)
    else:
        # Retrieve the planet using the provided Kepler name
        planet = client.planet(id)
        #planet = client.koi(id)

        # Retrive the star of the planet
        star = planet.star

    # Retrieve the light curves from the star
    lightcurves = star.get_light_curves(short_candence=False)

    # Display all the lightcurve (.fits) filenames.
    for lc in lightcurves:
        print(lc.filename)

    # Declare lists for the various attributes to retrieve from the lightcurve time series data
    time, flux, ferr, quality = [], [], [], []

    for lc in lightcurves:
        with lc.open() as f:
            hdu_data = f[1].data
            time.append(hdu_data["time"])
            flux.append(hdu_data["pdcsap_flux"])
            ferr.append(hdu_data["pdcsap_flux_err"])
            quality.append(hdu_data["sap_quality"])

    # Initialise pandas dataframes from retrieved data for easy manipulation
    df_time = pd.DataFrame(time).transpose()
    df_flux = pd.DataFrame(flux).transpose()
    df_error = pd.DataFrame(ferr).transpose()
    df_quality = pd.DataFrame(quality).transpose()

    print(df_quality.info())
    print(df_quality.describe())

    # Select columns with more than n samples
    n = 4000
    df_quality = df_quality.loc[:, df_quality.count() > n]
    #df_error = df_error.loc[:, df_error.count() > n]


    # Select the column with the lowest mean quality (Research into better approach to determine the best time series data)
    df_mean_quality = df_quality.mean()
    #df_mean_error = df_error.mean()

    lowest_quality_mean = df_mean_quality.min()
    #lowest_error_mean = df_mean_error.min()


    df_quality = df_quality.loc[:, df_quality.mean() == lowest_quality_mean]
    #df_error = df_error.loc[:, df_error.mean() == lowest_error_mean]

    target_column = int(df_quality.columns.values)

    # Using the target column obtained for the quarter which has the best quality light curve,
    # select the correct data column from the PCD_FLUX dataframe
    df_flux_output = df_flux.loc[:, target_column]

    print('Succesfully retrieved the lightcurve for star: ' + str(id))

    return df_flux_output


if __name__ == "__main__":
    main()

    # np_time = np.empty((0,1))
    #
    # print(np_time.shape)
    #
    # for t in time:
    #
    #     temp_np = np.array(t)[np.newaxis]
    #     temp_np = temp_np.transpose()
    #     temp_np = np.reshape(temp_np,(-1,1))
    #     print(temp_np.shape)
    #
    #     np_time = np.stack((np_time, temp_np))
    #
    # #time_np_array = np.concatenate(time,axis=0)
    # time_np_array = np.hstack(time)
    #
    # time_np_array = np.vstack([time_np_array, time]) if time_np_array.size else time
    #
    # flux_np_array = np.hstack(flux)


