# Short-Term Probabilistic Load Forecasting at Low Aggregation Levels Using Convolutional Neural Networks

*Lowly aggregated load profiles such as of individual households or buildings are more fluctuating and relative forecast errors are comparatively high. Therefore, the prevalent point forecasts are not sufficiently capable of optimally capturing uncertainty and hence lead to non-optimal decisions in different operational tasks. We propose an approach for short-term load quantile forecasting based on convolutional neural networks (STLQF-CNN). Historical load and temperature are encoded in a three-dimensional input to enforce locality of seasonal data. The model efficiently minimizes the pinball loss over all desired quantiles and the forecast horizon at once. We evaluate our approach for day-ahead and intra-day predictions on 222 house-holds and different aggregations from the Pecan Street dataset. The evaluation shows that our model consistently outperforms a naïve and an established linear quantile regression benchmark model, e.g., between 21 and 29 % better than the best benchmark on aggregations of 10, 20 and 50 households from Austin.*

A. Elvers, M. Voß and S. Albayrak, "Short-Term Probabilistic Load Forecasting at Low Aggregation Levels Using Convolutional Neural Networks," *2019 IEEE Milan PowerTech*, Milan, Italy, 2019, pp. 1-6.
[doi:10.1109/PTC.2019.8810811](https://ieeexplore.ieee.org/document/8810811)


## Installation

This project is tested on Python 3.6.6. It is recommended to install the project in a virtual environment. One method of creating a virtual environment is by running `python3.6 -m venv env`. When you want to use the project, you have to activate it first by running `source env/bin/activate`. Updating pip before installing other packages is always a good idea, you can do this with `pip install -U pip`.

If you only want to run experiments, you can install the project with `pip install '.[tfcpu]'` or `pip install '.[tfgpu]'` (depending on whether you want to use the model on GPU or CPU). If you also want to run tests and edit the code, you should install it in editable mode and with development requirements using `pip install -e '.[dev,tfcpu]'` or `pip install -e '.[dev,tfgpu]'`.


## Command line usage

The installation includes a console script `stplfcnn`:

```
$ stplfcnn
Usage: stplfcnn [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  apply_partitioner  Apply a partitioner to a range of issue...
  data_summary       Show a summary of the data.
  hyperopt           Hyperparameter optimization with Hyperopt.
  predict            Predict loads using a model of a given round...
  train              Train a model on a dataset with given rounds...
```

Each command has an own help text.

Usage example:

```
$ stplfcnn train --estimator naive-1d-add-hor-24h-H-q-7 --data_reader boulder-H-agg_10-10_1 --partitions 1_5y-freq1d-offset14d-boulder-cv-f5-r4
```


## Database

The project uses a structure of parameters and models that is called the database.
The base path of the database is the current folder but can be changed by setting
the `STPLFCNN_BASE_PATH` environment variable.

### Parameters

All parameters are defined by yaml files. The structure of the parameter folder
looks as follows:

```
$STPLFCNN_BASE_PATH/params
├── data_readers
│   └── data_reader_params.yaml
├── estimators
│   └── estimator_params.yaml
├── hyperopt
│   └── hyperopt_params.yaml
├── issue_times
│   └── issue_times_params.yaml
├── partitioners
│   └── partitioner_params.yaml
├── partitions
│   └── partitions_params.yaml
├── quantile_levels
│   └── quantile_levels_params.yaml
└── time_horizons
    └── time_horizons_params.yaml
```

Not all of them are required. When calling a command with parameter files, the
file suffix has to be removed.

We provide some examples in `params_examples`.

### Models

In `$STPLFCNN_BASE_PATH/models`, a directory is created in each training run.
There might also be a trails folder if hyperparameter optimization was used.
A model folder contains the untrained model, the state of a model learned
on the training data of a partition, predictions and errors of the predictions.


## Workflow

* create parameter files for data readers, quantile levels and time horizons
* if partitions (iterations of data splits into train and test) should be created
  manually, create the parameter files manually
* otherwise, create issue times and partitioners (e.g., k-fold cross-validation)
  parameter files and apply them using the `apply_partitioner` command
* if hyperopt should be used to find the best estimator parameters, create the
  parameter file for hyperopt and run the `hyperopt optimize` command (have a
  look at the options like `--warm-start`, `--dump_trained_state` and
  `--dump_predictions` before)
* if estimators should be created manually (e.g., benchmark models), create the
  parameter files manually
* run `train` on the estimators you want to investigate (benchmark models,
  best parameters found in hyperparameter optimization
* if you want, run `predict` to generate predictions for other issue times using
  a trained model


## TensorFlow configuration

You can configure TensorFlow using a `tensorflow.yaml` in the working directory.
Example:

```
allow_soft_placement: true
gpu_options:
  allow_growth: true
#log_device_placement: true
```
