import kplr

client = kplr.API()

koi = client.koi(01274.01)
print(koi.koi_period)

star = koi.star
print(star.kic_teff)

lightcurves = koi.get_light_curves(short_cadence=False)

for lc in lightcurves:
    print(lc.filename)

time, flux,ferr,quality = [],[],[],[]
for lc in lightcurves:
    with lc.open() as f:
        hdu_data = f[1].data
        time.append(hdu_data["time"])
        flux.append(hdu_data["sap_flux"])
        ferr.append(hdu_data["sap_flux_err"])
        quality.append(hdu_data["sap_quality"])

print("Goddamn!")

