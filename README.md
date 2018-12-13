# Exoplanet Transit Detection using Deep Neural Networks

MSc dissertation project that focuses on the use of deep neural networks for the detection of exoplanets in orbit of stars in the Milky Way. MLP, CNN and LSTM networks were built and optimized to recognise light curve patterns in the photometric time-series data to determine whether a transit signal is indeed caused by a real transiting planet, or a false positive such as an eclipsing binary star.
The data used for this project was produced by NASA's Kepler telescope survey, and was retrieved from the NASA Exoplanet Archive. 

To get a more in depth understanding of the work behind this project, consult my MSc dissertation [here]() and the papers in the credits section below. 

For a shorter summary, a powerpoint presentation was also prepared -> 

## Dependencies

  * [Python3](https://www.continuum.io/downloads)
  * [Numpy](http://www.numpy.org/)
  * [Keras](https://keras.io/)
  * [TensorFlow](https://www.tensorflow.org/)
  * [Matplotlib](https://matplotlib.org/)
  * [Scikit-learn](http://scikit-learn.org/stable/)
  * [Hyperopt](https://github.com/hyperopt/hyperopt/)

## Getting started


## Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   ├── split          <- The train/test split of the processed data. 
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py


--------


## Comprehensive File Guide 

`make_dataset.py` - Reads Kepler light curves from .fits files

## Ways to improve the research 



## Credits

