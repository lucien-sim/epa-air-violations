# EPA Air Violations

The model takes in information about a given inspection under the Clean Air Act (CAA), and outputs a probability that the inspection will reveal a violation. The current output for all pollution sources regulated under the CAA is available through this [web application](https://epa-air-violations.herokuapp.com). Comprehensive documentation for the project is available through the web applications's [documentation page](https://epa-air-violations.herokuapp.com/documentation). The model was created as part of my capstone project for [The Data Incubator](https://www.thedataincubator.com). 

## Getting Started

### What's in this repository?
* `execute_retraining.sh`: Script that instructs the model retraining and pushes the result to Heroku. 
* `prepare_to_retrain.py`: Script that prepares the directory for model retraining (mostly prepares the log file).
* `download_data.ipynb`: Notebook that downloads all data for the model. 
* `download_data.py`: Script version of `download_data.ipynb`. 
* `link_inspections_violations.ipynb`: Notebook that links inspections and violations. 
* `link_inspections_violations.py`: Script version of `link_inspections_violations.ipynb`. 
* `prepare_nei_data.ipynb`: Notebook that prepares National Emissions Inventories data for use in the model. 
* `prepare_nei_data.py`: Notebook version of `prepare_nei_data.ipynb`. 
* `build_evaluate_model.ipynb`: Notebook that creates the training/test datasets, and then trains and evaluates the model. 
* `build_evaluate_model.py`: Script version of `build_evaluate_model.ipynb`. 
* `tests.py`: Script that contains some unit tests. 
* `external_variables.py`: Contains a few variables that the system needs in order to run. 

### Prerequisites
* Python 3
* pandas
* numpy
* scipy
* scikit-learn
* imbalanced-learn
* joblib
* requests
* matplotlib
* bokeh

### Installing and runnning

With anaconda: 
1. Install anaconda
2. Create a new Python 3 environment (`myenv` can be whatever you want):
```
conda create -n myenv python=3
```
3. Enter the new environment: 
```
source activate myenv
```
4. Install dependencies: 
```
conda install -c conda-forge pandas numpy scipy scikit-learn imbalanced-learn joblib requests matplotlib bokeh
```
5. Change into the repository's main directory on your local machine. 
6. Type `execute_retraining.sh` to train the model. 
7. If you wish to deploy the model, set up your own [Heroku](https://www.heroku.com) application to do so. If you do not wish to deploy the model, just comment the last few lines of `execute_retraining.sh`. 

## Running the tests

The tests run automatically during the data download/model retraining process. If a test fails, an AssertionError is written to the log file and the retrained model is not uploaded to Heroku. 

### Test descriptions

1. TEST 1: Check filenames of all the downloaded data. This ensures that the code is able to locate the files it needs to access, and ensures that no serious changes occur in the EPA data without our knowing. These tests are executed in `download_data.py`. 
2. TEST 2: Check column names in all of the downloaded data. This ensures that the input data has a format that the model is able to deal with. These tests are executed in `download_data.py`. 
3. TEST 3: Check the percentage of violations that are linked to inspections. If this drops below 60%, the test fails. When I developed the model, the percentage was hovering around 75%. This test is executed in `link_inspections_violations.py`. 
4. TEST 4: Check the model's performance against a baseline. An error is raised if the model does not meet the standards. This test is executed in `tests.py`. 
5. TEST 5: Check that the data file for the web-app looks good. Make sure it has all the columns that the web-app needs, and make sure that the file has predictions for at least 150,000 facilities (it should have predictions for ~190,000). This test is executed in `tests.py`. 

## Authors

* **Lucien Simpfendoerfer** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Inspiration came from a recent paper by [Hino et al. (2018)](https://static1.squarespace.com/static/5bf34064c3c16a648f15d85b/t/5bf3d37503ce64eaeba7bab2/1542706045258/Hino+Benami+Brooks+2018+Machine+learning+for+environmental+monitoring.pdf)
