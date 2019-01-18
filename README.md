# Exoplanet Transit Detection using Deep Neural Networks

MSc dissertation project that focuses on the use of deep neural networks for the detection of exoplanets in solar systems other than our own in the Milky Way. Two deep learning architectures, convolutional (CNN) and long-short term memory (LSTM), are presented as automated vetting methods for the accurate detection of exoplanets in data produced by NASA's Kepler telescope survey. 
The networks were optimized to recognise light curve patterns in the photometric time-series data to solve a binary classification task, i.e to determine whether a transit signal is indeed caused by a real transiting planet, or a false positive such as an eclipsing binary star, stellar variability or instrumental artefacts. 
The final model configurations were able to correctly classify Kepler Threshold-Crossing Events (TCEs) with a recall score of ≈ 93.5%. 
In the future, automated vetting procedures like the methods proposed in this research will be commonplace in the Astronomy domain due to the large influx of data from new telescope surveys with the goal of observing celestial objects including exoplanets and many other phenomena in the universe. 

## Kepler Data Overview 

Labelled data from the Q1-Q17 Data Release 24 catalogue was obtained. The DR24 catalogue is composed of ≈ 15740 Threshold-Crossing Events (TCEs). TCEs can be defined as any detected transit-like event that cross a predefined threshold degree of certainty, and are then subjected to further analysis by the Kepler team to determine if the transit is caused by an exoplanet or other celestial objects/phenomena. 
![An illustration of an exoplanet transit is shown below](TRANSIT.gif)

To get a more in depth understanding of the work behind this project, consult my MSc dissertation [here](reports/dissertation.pdf) and the papers in the credits section below. 

For a shorter summary, a powerpoint presentation was also prepared -> [] 

# Project

## Dependencies

  * [Python3](https://www.continuum.io/downloads)
  * [Numpy](http://www.numpy.org/)
  * [Keras](https://keras.io/)
  * [TensorFlow](https://www.tensorflow.org/)
  * [Matplotlib](https://matplotlib.org/)
  * [Scikit-learn](http://scikit-learn.org/stable/)
  * [Hyperopt](https://github.com/hyperopt/hyperopt/)

## Structure

------------
    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   ├── split          <- The train/test split of the processed data. 
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
        ├── __init__.py    <- Makes src a Python module
        │
        ├── data           <- Scripts to download or generate data
        │   └── make_dataset.py
        │
        ├── helpers
        │   | 
        |   ├──────── thirdparty  <- Third party helper classes for preprocessing light curves 
        |   ├── import_helpers.py
        |   └── train_helpers.py 
        |
        ├── models         <- Scripts to train models and predict new data.
        │   │
        |   ├── model.py
        |   ├── optimize.py
        |   ├── predict_model.py
        │   └── train_model.py
        │
        └── visualization  <- Scripts to create exploratory and results oriented visualizations
            └── visualize.py

    <p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
--------

## File Guide 

`make_dataset.py` - Reads Kepler light curves from .fits files


# Walkthrough 

## Setup 


## Download Kepler Data 




The lightcurve data is retrieved from the [NASA Exoplanet
Archive](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=q1_q17_dr24_tce).


* `rowid`: Integer ID of the row in the TCE table.
* `kepid`: Kepler ID of the target star.
* `tce_plnt_num`: TCE number within the target star.
* `tce_period`: Period of the detected event, in days.
* `tce_time0bk`: The time corresponding to the center of the first detected
      event in Barycentric Julian Day (BJD) minus a constant offset of
      2,454,833.0 days.
* `tce_duration`: Duration of the detected event, in hours.
* `av_training_set`: Autovetter training set label; one of PC (planet candidate),
      AFP (astrophysical false positive), NTP (non-transiting phenomenon),
      UNK (unknown).
      

## Process Kepler Data




## Improvements 


## Credits 

Shallue, C. J., & Vanderburg, A. (2018). Identifying Exoplanets with Deep
Learning: A Five-planet Resonant Chain around Kepler-80 and an Eighth Planet
around Kepler-90. *The Astronomical Journal*, 155(2), 94. 
[*The Astronomical Journal*](http://iopscience.iop.org/article/10.3847/1538-3881/aa9e09/meta).





## Credits

