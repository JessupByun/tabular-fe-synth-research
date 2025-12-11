# alfred-analytica
# Table of Contents
1. Set Up
2. Basic Instruction
3. Format of input dictionaries
4. Adding custom metrics and plotters
5. Known Bugs

## Set Up

Install using `pip install git+https://github.com/ucla-trustworthy-ai-synthdata/alfred-analytica.git`

If the option above does not work, install the package using `pip install git+$GITHUB_URL` where `$GITHUB_URL` is the HTTPS url of the repository

**Note: If testing in pycharm, disable "Show plots in tool window" in Preferences > Tools > Python Scientific**

## Basic Instruction
Users can decide to run the program in two ways:
1. Use the automated `EvaluationPipeline`
2. Manually create each class

For each case, user needs to prepare the following files
* `real_data`: A csv file containing the real data
* `synth_data`: A csv file containing the synthetic data
* `column_name_to_datatype`: A dictionary mapping each column name to a string representing its datatype (see next section for format)
* `config`: A dictionary containing the configuration for the evaluation pipeline (see next section for format)
* `save_path`: A string representing the path to a directory to save the tables and plots (MUST LEAD TO A DIRECTORY)

### Method 1: Automated Evaluation Pipeline
To run the automated evaluation pipeline, run the following code
```python
from evaluator import *

evaluation_pipeline = EvaluationPipeline(real_data=real_data, synth_data=synth_data, column_name_to_datatype=column_name_to_datatype, config=config, save_path=save_path)
evaluation_pipeline.run_pipeline()
```
The result will be saved in the `save_path` directory.

### Method 2: Manual Evaluation Pipeline
The manual pipeline requires individual factories and modules to be initialized

```python
from evaluator import *

# Instantiating Factories
metric_factory = MetricFactory()
plotter_factory = PlotterFactory()
evaluation_factory = EvaluationFactory(metric_factory, plotter_factory, real_data, synth_data, column_name_to_datatype,
                                       config)

# Instantiating Evaluator Object
ev = Evaluator(evaluation_factory, real_data=real_data, synth_data=synth_data,
               column_name_to_datatype=column_name_to_datatype,
               config=config)

# Evaluating
ev.evaluate_all()

# Get created reports.
report = ev.get_report()

# Instantiating and using Saver
save_path = "evaluator/tests/test_results/plot_directory"
saver = Saver(reportPath=result_save_dir)
saver.saveReport(report)

```

## Format of Input Dictionaries

### Format of column_name_to_datatype

This dictionary maps string to string. The key represents the name of a column. The value represents the datatype. Pick the value that most accurately represents the datatype (e.g. pick datetime over numerical if applicable)

Currently supported datatypes are:
* `numerical`
* `categorical`
* `datetime`

A sample dictionary is as follows:
```python
column_name_to_datatype = {
        "start_hours_from_admit": "numerical",
        "drug_name": "categorical",
        "ethnicity": "categorical",
        "insurance": "categorical",
        "gender": "categorical",
        "starttime":"datetime",
        "admittime": "datetime",
        "dod": "datetime",
        "label":"categorical"
    }
```

### Format of config

Config is a dictionary that determines additional details to specific metrics. It is optional to include config. Some key configs are:
* `target`: The target column name, for metrics involving model fitting
* `chunk_size`: Chunk size for DCR calculation
* `test_size`: Size for train test split for utility models
* `random_state`: Random state for utility models
* `fidelity_metrics`: List of fidelity metrics to evaluate
* `utility_metrics`: List of utility metrics to evaluate
* `privacy_metrics`: List of privacy metrics to evaluate

The example below sets target column to "label" and specifies fidelity metrics to evaluate `SumStats` and `ColumnShape` only. Privacy and utility metrics will be evaluated according to default settings.
```python
config = {
    "target": "label",
    "fidelity_metrics": ["SumStats", "ColumnShape"]
}
```

## Adding Custom Metrics and Plotters

If using the factory method to operate the evaluator, user can add in custom metrics and plotters through the following steps
1. Import desired metric/plotter classes
2. Creating a new dictionary that maps a custom name to each class (not to an instance of the class, just to the class directly)
3. Add the dictionary as a parameter when creating the factory

An example:
```python
import CustomMetricA
import CustomMetricB
import CustomPlotterA
import CustomPlotTypeA

custom_metric_A = CustomMetricA
custom_metric_B = CustomMetricB
custom_plotter_A = CustomPlotterA

custom_metric_dict = {
    "custom_name_A": custom_metric_A,
    "custom_name_B": custom_metric_B
}

# CustomPlotTypeA is an enumeration, see PlotTypes class in interfaces/metric_interface.py for more details
# To plot with custom_plotter_A, the metric class must have the plot_type defined as the CustomPlotTypeA enumeration
custom_plotter_dict = {
    CustomPlotTypeA: custom_plotter_A
}

# The config below adds custom metric A and B as two fidelity metrics, in addition to the default ColumnShape and PairwiseSimilarity metrics
config=["target_column": "target",fidelity_metrics"custom_name_A, custom_name_B, ColumnShape, PairwiseSimilarity"]

metric_factory = MetricFactory(custom_metric_dict)
plotter_factory = PlotterFactory(custom_plotter_dict)

... # Continue the rest as normal

```

## Known Bugs (And Fixes)

* In order to successfully save plots, the `kaleido` package requires the directory path to contain no space characters
* LightGBM model crashes Python, run on Mac OS Ventura
* XGBoost package requires binary values [0, 1] to be passed in, cannot take hash values
* Binomial variable roc_auc_score requires different dimensionality from typical classifers.

## Third-Party Libraries and Licenses

This project integrates or wraps functionality from the following upstream libraries. We acknowledge their authors and cite their licenses:

- SynthCity (vanderschaarlab): https://github.com/vanderschaarlab/synthcity — License: Apache License 2.0
- Synth-MIA (UCLA Trustworthy AI Lab): https://github.com/UCLA-Trustworthy-AI-Lab/Synth-MIA — License: Unavailable/Not detected (repository private or missing LICENSE)
- SDMetrics (sdv-dev): https://github.com/sdv-dev/SDMetrics — License: MIT License

If the Synth-MIA repository becomes publicly accessible or provides an explicit LICENSE file, please update this section accordingly or open an issue with the license details.

## License

This project is licensed under the Apache License 2.0. See the `LICENSE` file for details.
