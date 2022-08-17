# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from azureml.core import Run
import pandas as pd
import numpy as np
import os
import argparse
import datetime
import joblib

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import GradientBoostingClassifier

from timeseries_utilities import ColumnDropper, SimpleLagger, SimpleCalendarFeaturizer, SimpleForecaster
from utilities import set_telemetry_scenario

from azureml.core import Environment, ScriptRunConfig, Experiment
from azureml.core.conda_dependencies import CondaDependencies
from azureml.train.hyperdrive import GridParameterSampling, HyperDriveConfig, PrimaryMetricGoal, choice

from inspect import getsourcefile
from os.path import abspath

# 0.0 Parse input arguments
parser = argparse.ArgumentParser("split")
parser.add_argument("--target_column", type=str, required=True, help="input target column")
parser.add_argument("--timestamp_column", type=str, required=True, help="input timestamp column")
parser.add_argument("--timeseries_id_columns", type=str, nargs='*', required=True,
                    help="input columns identifying the timeseries")
parser.add_argument("--drop_columns", type=str, nargs='*', default=[],
                    help="list of columns to drop prior to modeling")
parser.add_argument("--model_type", type=str, required=True, help="input model type")
parser.add_argument("--test_size", type=int, required=True, help="number of observations to be used for testing")

args, _ = parser.parse_known_args()

current_run = None


def init():
    global current_run
    current_run = Run.get_context()
    # Update step run with the right traits to denote it is training
    set_telemetry_scenario(current_run, 'ManyModelsCustomScriptTrain')


def set_telemetry(run):
    prop = {"azureml.runsource": "azureml.ManyModelsCustomTrain"}
    try:
        run.add_properties(prop)
    except Exception:
        pass

def get_best_hyperparameters(file_name):
    # does the hyper parameters optimization
    # Create a Python environment for the experiment
    trial_env = Environment(name="many_models_environment")
    trial_conda_deps = CondaDependencies.create(pip_packages=['sklearn', 'pandas', 'joblib', 'azureml-defaults', 'azureml-core'])
    trial_env.python.conda_dependencies = trial_conda_deps

    # Get the training dataset
    dataset_name = 'oj_data_small_train'
    ws = current_run.experiment.workspace
    dataset = ws.datasets.get(dataset_name)

    # Create a script config
    cpu_cluster_name = "cpucluster5"
    
    current_script_path = abspath(getsourcefile(lambda:0))
    current_script_dir = current_script_path[:current_script_path.rindex("/")]
   
    script_config = ScriptRunConfig(source_directory=current_script_dir,
                                    script='trial.py',
                                    # Add non-hyperparameter arguments -in this case, the training dataset
                                    arguments = ['--input-data', dataset.as_named_input('training_data').as_mount(),
                                                 '--file_name', f'{file_name}.csv',
                                                 '--target_column', 'Quantity', 
                                                 '--timestamp_column', 'WeekStarting', 
                                                 '--timeseries_id_columns', 'Store', 'Brand',
                                                 '--drop_columns', 'Revenue', 'Store', 'Brand',
                                                 '--model_type', 'gb',
                                                 '--test_size', 20],
                                    environment=trial_env,
                                    compute_target = cpu_cluster_name)

    # Sample a range of parameter values
    params = GridParameterSampling(
        {
            # Hyperdrive will try 6 combinations, adding these as script arguments
            '--learning_rate': choice(0.01, 0.1, 1.0),
            '--n_estimators' : choice(10, 100)
        }
    )

    # Configure hyperdrive settings
    hyperdrive = HyperDriveConfig(run_config=script_config, 
                            hyperparameter_sampling=params, 
                            policy=None, # No early stopping policy
                            primary_metric_name='mae', # Find the lowest mae metric
                            primary_metric_goal=PrimaryMetricGoal.MINIMIZE, 
                            max_total_runs=6, # Restict the experiment to 6 iterations
                            max_duration_minutes=10, # Restrict execution to 10 minutes
                            max_concurrent_runs=2) # Run up to 2 iterations in parallel

    # Run the experiment
    experiment_name = f'trial_{file_name}'.replace('.', '_')
    experiment = Experiment(workspace=ws, name=experiment_name)
    run = experiment.submit(config=hyperdrive)

    # Show the status in the notebook as the experiment runs
    run.wait_for_completion()

    # Print all child runs, sorted by the primary metric
    for child_run in run.get_children_sorted_by_primary_metric():
        print(child_run)

    # Get the best run, and its metrics and arguments
    best_run = run.get_best_run_by_primary_metric()
    best_run_metrics = best_run.get_metrics()
    script_arguments = best_run.get_details() ['runDefinition']['arguments']

    # get best trial hyperparameters values
    learning_rate = script_arguments[script_arguments.index('--learning_rate')+1]
    n_estimators = script_arguments[script_arguments.index('--n_estimators')+1]

    return float(learning_rate), int(n_estimators)


def run(input_data):
    # 1.0 Set up output directory and the results list
    os.makedirs('./outputs', exist_ok=True)
    result_list = []

    # 2.0 Loop through each file in the batch
    # The number of files in each batch is controlled by the mini_batch_size parameter of ParallelRunConfig
    for idx, csv_file_path in enumerate(input_data):
        result = {}
        start_datetime = datetime.datetime.now()

        file_name = os.path.basename(csv_file_path)[:-4]
        model_name = args.model_type + '_' + file_name

        # 1.0 Read the data from CSV - parse timestamps as datetime type and put the time in the index
        data = (pd.read_csv(csv_file_path, parse_dates=[args.timestamp_column], header=0)
                .set_index(args.timestamp_column)
                .sort_index(ascending=True))

        # 2.0 Split the data into train and test sets
        train = data[:-args.test_size]
        test = data[-args.test_size:]

        child_run = None

        # try:

        child_run = current_run.child_run(name=model_name)

        # Do not remove the following code
        set_telemetry(child_run)

        # learning_rate, n_estimators = float('0.1'), int('100')
        learning_rate, n_estimators = get_best_hyperparameters(file_name)

        # 3.0 Create and fit the forecasting pipeline
        # The pipeline will drop unhelpful features, make a calendar feature, and make lag features
        lagger = SimpleLagger(args.target_column, lag_orders=[1, 2, 3, 4])
        transform_steps = [('column_dropper', ColumnDropper(args.drop_columns)),
                            ('calendar_featurizer', SimpleCalendarFeaturizer()), ('lagger', lagger)]
        forecaster = SimpleForecaster(transform_steps, GradientBoostingClassifier(learning_rate=learning_rate, n_estimators=n_estimators), 
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
        child_run.log(model_name + '_mse', mse)
        child_run.log(model_name + '_rmse', rmse)
        child_run.log(model_name + '_mae', mae)
        child_run.log(model_name + '_mape', mape)

        # 7.0 Train model with full dataset
        forecaster.fit(data)

        # Simulating the 10 sec run to test concurrency
        import time
        time.sleep(10)

        # 8.0 Save the forecasting pipeline
        joblib.dump(forecaster, filename=os.path.join('./outputs/', model_name))

        # 9.0 Register the model to the workspace
        # Uses the values in the timeseries id columns from the first row of data to form tags for the model
        child_run.upload_file(model_name, os.path.join('./outputs/', model_name))
        ts_id_dict = {id_col: str(data[id_col].iloc[0]) for id_col in args.timeseries_id_columns}
        tags_dict = {**ts_id_dict, 'ModelType': args.model_type}
        tags_dict.update({'InputData': os.path.basename(csv_file_path)})
        tags_dict.update({'StepRunId': current_run.id})
        tags_dict.update({'RunId': current_run.parent.id})
        child_run.register_model(model_path=model_name, model_name=model_name,
                                    model_framework=args.model_type, tags=tags_dict)

        child_run.complete()
        # 10.0 Add data to output
        end_datetime = datetime.datetime.now()
        result.update(ts_id_dict)
        result['model_type'] = args.model_type
        result['file_name'] = file_name
        result['model_name'] = model_name
        result['start_date'] = str(start_datetime)
        result['end_date'] = str(end_datetime)
        result['duration'] = str(end_datetime-start_datetime)
        result['mse'] = mse
        result['rmse'] = rmse
        result['mae'] = mae
        result['mape'] = mape
        result['index'] = idx
        result['num_models'] = len(input_data)
        result['status'] = child_run.get_status()
        result['run_id'] = str(child_run.id)

        print('ending (' + csv_file_path + ') ' + str(end_datetime))
        result_list.append(result)

    # Data returned by this function will be available in parallel_run_step.txt
    return pd.DataFrame(result_list)
