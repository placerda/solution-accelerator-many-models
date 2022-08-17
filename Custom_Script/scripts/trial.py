# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse, joblib, os
import datetime
from azureml.core import Dataset, Run
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import GradientBoostingClassifier

from timeseries_utilities import ColumnDropper, SimpleLagger, SimpleCalendarFeaturizer, SimpleForecaster
from utilities import set_telemetry_scenario

# Get the experiment run context
run = Run.get_context()

# Get script arguments
parser = argparse.ArgumentParser()

# Input parameters
parser.add_argument("--input-data", type=str, dest='input_data', help='training dataset')
parser.add_argument("--file_name", type=str, dest='file_name', help='specific training filename')
parser.add_argument("--target_column", type=str, required=True, help="input target column")
parser.add_argument("--timestamp_column", type=str, required=True, help="input timestamp column")
parser.add_argument("--timeseries_id_columns", type=str, nargs='*', required=True,
                    help="input columns identifying the timeseries")
parser.add_argument("--drop_columns", type=str, nargs='*', default=[],
                    help="list of columns to drop prior to modeling")
parser.add_argument("--model_type", type=str, required=True, help="input model type")
parser.add_argument("--test_size", type=int, required=True, help="number of observations to be used for testing")

# Hyperparameters
parser.add_argument('--learning_rate', type=float, dest='learning_rate', default=0.1, help='learning rate')
parser.add_argument('--n_estimators', type=int, dest='n_estimators', default=100, help='number of estimators')

# Add arguments to args collection
args = parser.parse_args()

# Log Hyperparameter values
run.log('learning_rate',  np.float(args.learning_rate))
run.log('n_estimators',  np.int(args.n_estimators))

start_datetime = datetime.datetime.now()

# get the input dataset by ID
dataset = run.input_datasets['training_data']

# Get the right csv file from the dataset
csv_file_path = f'{dataset}/{args.file_name}'

file_name = os.path.basename(csv_file_path)[:-4]
model_name = args.model_type + '_' + file_name

# 1.0 Read the data from CSV - parse timestamps as datetime type and put the time in the index
data = (pd.read_csv(csv_file_path, parse_dates=[args.timestamp_column], header=0)
        .set_index(args.timestamp_column)
        .sort_index(ascending=True))

# 2.0 Split the data into train and test sets
train = data[:-args.test_size]
test = data[-args.test_size:]

# 3.0 Create and fit the forecasting pipeline
# The pipeline will drop unhelpful features, make a calendar feature, and make lag features
lagger = SimpleLagger(args.target_column, lag_orders=[1, 2, 3, 4])
transform_steps = [('column_dropper', ColumnDropper(args.drop_columns)),
                    ('calendar_featurizer', SimpleCalendarFeaturizer()), ('lagger', lagger)]
forecaster = SimpleForecaster(transform_steps, GradientBoostingClassifier(learning_rate=args.learning_rate, n_estimators=args.n_estimators), 
                                args.target_column, args.timestamp_column)
            
forecaster.fit(train)
print('Featurized data example:')
print(forecaster.transform(train).head())

# 4.0 Get predictions on test set
forecasts = forecaster.forecast(test)
compare_data = test.assign(forecasts=forecasts).dropna()

# 5.0 Calculate accuracy metrics for the fit
mse = mean_squared_error(compare_data[args.target_column], compare_data['forecasts'])
rmse = np.sqrt(mse)
mae = mean_absolute_error(compare_data[args.target_column], compare_data['forecasts'])
actuals = compare_data[args.target_column].values
preds = compare_data['forecasts'].values
mape = np.mean(np.abs((actuals - preds) / actuals) * 100)

# 6.0 Log metrics
run.log('mse', mse)
run.log('rmse', rmse)
run.log('mae', mae)
run.log('mape', mape)

# Simulating the 10 seconds run to test concurrency
import time
time.sleep(10)

# 8.0 Save the forecasting pipeline
os.makedirs('outputs', exist_ok=True)
joblib.dump(forecaster, filename=os.path.join('./outputs/', model_name))

end_datetime = datetime.datetime.now()
print('ending (' + csv_file_path + ') ' + str(end_datetime))

run.complete()