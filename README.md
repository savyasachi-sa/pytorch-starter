# pytorch-starter-code

A generic starter code for PyTorch projects. A config driven architecture which takes care of the boilerplate code for any project - 
* Ability to run different experiments with just config changes.
* Saving logs, plots and models as we run the experiment with an ability to resume experiments.
* A standard training method which can be tweaked based on the project.
* Single command execution.  

## Usage

* Define the configuration for your experiment in `./config` directory. See `default.json` to see the structure and available options.
* Implement factories to return project specific models, datasets and transforms based on config. Add more flags as per requirement in the config.
* Tweak `experiment.py` based on the project requirements.
* After defining the configuration (say `my_exp.json`) - simply run `python3 main.py my_exp` to start the experiment
* The logs, stats, plots and saved models would be stored in `../experiment_data/my_exp` dir. This can be configured in `contants.py`
* To resume an ongoing experiment, simply run the same command again. It will load the latest stats and models and resume training.

## Files
- main.py: Main driver class
- experiment.py: Main experiment class. Initialized based on config - takes care of training, saving stats and plots, logging and resuming experiments.
- dataset_factory: Factory to build datasets based on config
- transforms_factory.py: Factory to build image transforms based on config
- model_factory.py: Factory to build models based on config
- constants.py: constants used across the project
- file_utils.py: utility functions for handling files 
