# Median housing value prediction

The housing data can be downloaded from https://raw.githubusercontent.com/ageron/handson-ml/master/. The script has codes to download the data. We have modelled the median house value on given housing data. 

The following techniques have been used: 

 - Linear regression
 - Decision Tree
 - Random Forest

## Steps performed
 - We prepare and clean the data. We check and impute for missing values.
 - Features are generated and the variables are checked for correlation.
 - Multiple sampling techinuqies are evaluated. The data set is split into train and test.
 - All the above said modelling techniques are tried and evaluated. The final metric used to evaluate is mean squared error.

## To excute the script
python < scriptname.py >

## Setup Instructions

### 1. Clone the Repository
Clone the repository to your local machine using the command:
git clone https://github.com/Vidya-TA/mle-training
cd Vidya-TA

### 2. Set Up Conda Environment
Create the Conda environment using the provided env.yml file:
conda env create -f env.yml
After the environment is created, activate it:
conda activate mle-dev

### 3. Run the Code
Once the environment is activated, run the Python script using:
python nonstandardcode.py

### 4. Deactivate the Environment
When you're done working, deactivate the environment with:
conda deactivate
