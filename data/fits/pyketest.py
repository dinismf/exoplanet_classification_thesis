from pyke import *

# kepdraw('kplr012557548-2012179063303_llc.fits', plottype='fast', datacol='SAP_FLUX')
# kepdraw('kplr012557548-2012179063303_llc.fits', plottype='fast', datacol='PDCSAP_FLUX')
#
#
# kepflatten(infile='kplr012557548-2012179063303_llc.fits', nsig=3, stepsize=1, npoly=2,niter=10, overwrite=True)
#
# kepdraw('kplr012557548-2012179063303_llc-kepflatten.fits', plottype='fast', datacol='DETSAP_FLUX')


#kepfold(infile='kplr012557548-2012179063303_llc-kepflatten.fits', period=0.65355361, bjd0=2454965.059, datacol='DETSAP_FLUX', bindata=True, nbins=100, overwrite=True)

#kepdraw('folded_lc.fits', datacol='FLUX',  plottype='fast')




#### Using lower level functions ####

# Read Original FITS into Light Curve structure
og_lc = KeplerLightCurveFile(path='kplr012557548-2012179063303_llc.fits')
print(og_lc.header())
og_lc_pdcsap = og_lc.get_lightcurve('PDCSAP_FLUX')

print(og_lc_pdcsap.keplerid)
flattened_lc = og_lc_pdcsap.flatten()

# Detect best period
# postlist, trial_periods, best_period = box_period_search(flattened_lc, nperiods=2000)
# print('Best period: ', best_period)

# Fold light curve
folded_lc = flattened_lc.fold(period=0.65355361, phase=2454965.059)

binned_lc = folded_lc.bin(binsize=100, method='median')

plt.plot(folded_lc.time, folded_lc.flux, 'x', markersize=1, label='FLUX')
plt.show()