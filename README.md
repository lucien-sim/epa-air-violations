# EPA Air Violations

## Description
The model takes in information about a given inspection under the Clean Air Act (CAA), and outputs a probability that the inspection will reveal a violation. The current output for all pollution sources regulated under the CAA can be accessed through this [web application](https://epa-air-violations.herokuapp.com). Comprehensive documentation for the project is available through the web app's [documentation page](https://epa-air-violations.herokuapp.com/documentation). The model was created as part of my The Data Incubator capstone project. 

## What's in this repository?
* `link_insp_viol_final.ipynb`: Notebook that links inspections and violations. 
* `add_nei_data_final.ipynb`: Notebook that prepares National Emissions Inventories data for use in the model. 
* `build_and_evaluate_model_final.ipynb`: Notebook that creates the training/test datasets, and then trains and evaluates the model. 

These files contain all the code I used to build and test the model. In the next few days, I'll be adding code that automatically downloads/processes the data and trains the model. I'll also be adding tests to be executed each time the model is trained. 