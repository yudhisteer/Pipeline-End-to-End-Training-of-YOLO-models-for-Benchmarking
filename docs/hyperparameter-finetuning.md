This document explains how to use the HyperparameterTuner to tune the hyperparameters of the model.

## Table of Contents
1. [Hyperparameter Configs](#hyperparameter-configs)
2. [HyperparameterTuner](#hyperparametertuner)
   - [Search Space: Hyperparameter Ranges](#search-space-hyperparameter-ranges)
   - [Objective: Objective Metric, Type and Definitions](#objective-objective-metric-type-and-definitions)
   - [Strategy: Strategy, Max Jobs, Max Parallel Jobs and Base Tuning Job Name](#strategy-strategy-max-jobs-max-parallel-jobs-and-base-tuning-job-name)
3. [Fit the Tuner](#fit-the-tuner)

--------------------------------------------------
<a name="1-hyperparameter-configs"></a>
## 1. Hyperparameter Configs

Similarly to other component in our pipeline, all the variables and parameters are defined in the `config.yaml` file to avoid hard-coding them in the code. The important section is the `hyperparameter_ranges` section which is used to define the ranges of the hyperparameters that we want to tune. You can remove or add more hyperparameters to the section as you wish. See `all_hyperparams_available` section for all the hyperparameters available.


```yaml
# config.yaml
tuning_config = {
    "tuning": {
    "enabled": True,                 
    "update_config": True,              
    "max_jobs": 6,                     
    "max_parallel_jobs": 2,        
    "objective_metric": "mAP_0.5",      
    "objective_type": "Maximize",         

    "hyperparameter_ranges": {
        "lr0": {
        "type": "continuous",  
        "scaling_type": "Logarithmic",
        "min": 0.1,
        "max": 0.2,
        },
        "batch": {
        "type": "categorical",
        "values": [2, 4],
        },
        "optimizer": {  
        "type": "categorical",
        "values": ["SGD", "Adam"],
        }
        }
    }
}
```

--------------------------------------------------
<a name="2-hyperparametertuner"></a>
## 2. HyperparameterTuner

We start by creating a HyperparameterTuner instance using `from sagemaker.tuner import HyperparameterTuner`. It takes an estimator to obtain configuration information for training jobs that are created as the result of a hyperparameter tuning job. Note that we explain more about the estimator in the [estimator](docs/sagemaker-training.md) section.

``` python
def create_tuner(config_path: str = "config.yaml", tuning_config: dict = tuning_config) -> HyperparameterTuner:
    """Create the SageMaker hyperparameter tuner."""

    # create trainer
    trainer = YOLOSageMakerTrainer(config_path=config_path)
    
    # create estimator
    estimator = trainer.create_estimator(verbose=False)
    
    # get hyperparameter ranges
    hyperparameter_ranges = get_hyperparameter_ranges_from_config(tuning_config)
    
    # create tuner
    tuner = HyperparameterTuner(
        estimator=estimator,
        hyperparameter_ranges=hyperparameter_ranges,
        objective_type="Maximize",
        objective_metric_name="yolo:mAP_0.5",
        metric_definitions=estimator.metric_definitions,
        max_jobs=6,
        max_parallel_jobs=2,
        strategy="Bayesian",
        base_tuning_job_name="yolo-hpt-test"
    )
    
    print(f"Created hyperparameter tuner:")
    
    return tuner
```

--------------------------------------------------
<a name="21-search-space-hyperparameter-ranges"></a>
### 2.1. Search Space: Hyperparameter Ranges
So the HyperparameterTuner takes a hyperparameter_ranges parameter which is the search space that the tuner will explore. It is described as:

`hyperparameter_ranges (dict[str, sagemaker.parameter.ParameterRange])` – Dictionary of parameter ranges. These parameter ranges can be one of three types: `Continuous`, `Integer`, or `Categorical`. The keys of the dictionary are the **names** of the hyperparameter, and the values are the **appropriate parameter range class** to represent the range.

We write a helper function that will take the config file at the hyperparameter_ranges section and convert it to a dictionary of SageMaker parameter objects. Note that we also have a `scaling_type (str)` used for searching the range during tuning (default: ‘Auto’). Valid values are `Auto`, `Linear`, `Logarithmic` and `ReverseLogarithmic`.

``` python
def get_hyperparameter_ranges_from_config(tuning_config: dict) -> Dict:
    """
    Create SageMaker hyperparameter ranges from tuning configuration.
    
    Args:
        tuning_config: Tuning configuration dictionary from config.yaml
        
    Returns:
        Dictionary of SageMaker parameter objects
    """
    ranges = {}
    hyperparameter_ranges = tuning_config.get('tuning', {}).get('hyperparameter_ranges', {})
    
    for param_name, param_config in hyperparameter_ranges.items():
        param_type = param_config.get('type')

        if param_type == 'continuous':
            ranges[param_name] = ContinuousParameter(
                param_config['min'], 
                param_config['max']
            )
        elif param_type == 'categorical':
            ranges[param_name] = CategoricalParameter(param_config['values'])
        elif param_type == 'integer':
            ranges[param_name] = IntegerParameter(
                param_config['min'], 
                param_config['max']
            )
        else:
            print(f"Warning: Unknown parameter type '{param_type}' for {param_name}")

    return ranges
```

The function above will return a dictionary of SageMaker parameter objects as shown below:

``` python
# Get hyperparameter ranges
ranges = get_hyperparameter_ranges_from_config(tuning_config)
print("Ranges: ", ranges)

# Output:
Ranges:
{
    'lr0': <sagemaker.parameter.ContinuousParameter object at 0x71291e9bf770>,
    'batch': <sagemaker.parameter.CategoricalParameter object at 0x71291e9bf8c0>,
    'optimizer': <sagemaker.parameter.CategoricalParameter object at 0x71291ebc9310>
}
```

--------------------------------------------------
<a name="22-objective-objective-metric-type-and-definitions"></a>
### 2.2. Objective: Objective Metric, Type and Definitions

Next we want to measure how well the training job is performing. We do this by defining the metric(s) used to evaluate the training jobs.

`metric_definitions (list[dict])` - a list of dictionaries that defines the metric(s) used to evaluate the training jobs. Each dictionary contains two keys: `Name` for the name of the metric, and `Regex` for the regular expression used to extract the metric from the logs. This should be defined **only** for hyperparameter tuning jobs that don’t use an Amazon algorithm.

`objective_type (str)` – the type of the objective metric for evaluating training jobs. This value can be either `Minimize` or `Maximize` (default: `Maximize`).

`objective_metric_name (str)` – Name of the metric for evaluating training jobs.

```yaml
...
  objective_type="Maximize",
  objective_metric_name="yolo:mAP_0.5",
  metric_definitions=estimator.metric_definitions,
...

```

In our `entrypoint_trainer.py` file, we define the metric definitions as shown below. In our example we choose to `Maximize` the `mAP_0.5` metric.

```python
metric_definitions = [
    {
        "Name": "yolo:recall",
        "Regex": r"recall: ([0-9]*\.?[0-9]+)"
    },
    {
        "Name": "yolo:mAP_0.5",
        "Regex": r"mAP@0\.5: ([0-9\.]+)"
    },
    {
        "Name": "yolo:mAP_0.5_0.95",
        "Regex": r"mAP@0\.5:0\.95: ([0-9\.]+)"
    },
    {
        "Name": "yolo:precision",
        "Regex": r"precision: ([0-9\.]+)"
    },
    {
        "Name": "yolo:train_loss",
        "Regex": r"train/box_loss: ([0-9\.]+)"
    },
    {
        "Name": "yolo:val_loss",
        "Regex": r"val/box_loss: ([0-9\.]+)"
    }
]
```

--------------------------------------------------
<a name="23-strategy-strategy-max-jobs-max-parallel-jobs-and-base-tuning-job-name"></a>
### 2.3. Strategy: Strategy, Max Jobs, Max Parallel Jobs and Base Tuning Job Name

The HyperparameterTuner takes a strategy parameter which is the strategy to be used for hyperparameter estimations, max_parallel_jobs, max_jobs and base_tuning_job_name parameters. It is described as:

- `strategy (str)` – Strategy to be used for hyperparameter estimations (default: `Bayesian`).  Possible values are `Bayesian`, `Random`, `Grid`, `Hyperband`.
- `max_parallel_jobs (int)` – Maximum number of parallel training jobs to start (default: 1). For example, if set to 2, only 2 training jobs will run at the same time, even if you have more resources available.
- `max_jobs (int)` – Maximum total number of training jobs to start for the hyperparameter tuning job (default: 1). For example, if set to 6, SageMaker will test 6 different combinations of hyperparameters.
- `base_tuning_job_name (str)` – Prefix for the hyperparameter tuning job name when the `fit()` method launches. If not specified, a default job name is generated, based on the training image name and current timestamp.


```yaml
max_jobs=6,
max_parallel_jobs=2,
strategy="Bayesian",
base_tuning_job_name="yolo-hpt-test"
```


--------------------------------------------------
<a name="3-fit-the-tuner"></a>
## 3. Fit the Tuner
We can now fit the tuner using the `fit()` method.



``` python
tuner.fit(wait=False)
```

## References:
[1] "Amazon SageMaker Hyperparameter Tuning." *AWS Documentation*. Available at: https://sagemaker.readthedocs.io/en/stable/api/training/tuner.html

[2] "Use an Algorithm to Run a Hyperparameter Tuning Job" *AWS Documentation*. Available at: https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-mkt-algo-tune.html

[3] Medium. (2022). "How to Run Machine Learning Hyperparameter Optimization in the Cloud - Part 3." *Medium*. Available at: https://medium.com/data-science/how-to-run-machine-learning-hyperparameter-optimization-in-the-cloud-part-3-f66dddbe1415
