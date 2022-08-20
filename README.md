# Capstone Project: Azure Machine Learning Engineer

This project gave me the opportunity to use the knowledge i obtained from the Machine Learning Engineer for Microsoft Azure  Nanodegree to solve an interesting problem of predicting diabetes. In this project, i created two models: one using Automated ML (denoted as AutoML from now on) and one customized model whose hyperparameters are tuned using HyperDrive. The two models performance was compared and best performing model was deployed

The objectives of the project where to:
1. Demonstrate the ability to use an external dataset in my workspace
2. Train a model using the different tools available in the AzureML framework as well as 
3. My ability to deploy the model as a web service.

### Project Flow

<img width="510" alt="project flow" src="https://user-images.githubusercontent.com/111194883/185763444-799fe977-aa13-4bfb-be71-41cd6ddd418b.PNG">

## Dataset

### Overview

In this project i used a diabetes dataset i obtained from kaggle. The data set gave features that would normally be used my medical practioners in their diagnosis of diabetic patients and parameter they use to determine if someone has a possibility of getting diabetes or not. Features in the dataset included PatientID, Number of Pregnancies, PlasmaGlucose, DiastolicBloodPressure, TricepsThickness, SerumInsulin, BMI, DiabetesPedigree, Age and the target column Diabetic.

The link to the dataset : https://github.com/mothebeng2/Capstone-Project/blob/main/diabetes%20(1)%20(1).csv

### Task

This is a classification task. The goal of the project was to predict using medical history and data if a patience is diabetic or not. The features used in the dataset are PatientID, Number of Pregnancies, PlasmaGlucose, DiastolicBloodPressure, TricepsThickness, SerumInsulin, BMI, DiabetesPedigree, Age and the target column Diabetic. 

### Access

The dataset was downloaded to my computer from kaggle and then uploaded to the workspace using the "From Local Files" option in Machine Learning Studio under dataset registration. Second method of accessing the data could be using the from_delimited_files('webURL') of the TabularDatasetFactory. The first option was used and verified using code in the jupyter Notebook.

## Automated ML

The AutoML settings I have used are as follows :

automl_settings = {
    "experiment_timeout_minutes": 30,
    "max_concurrent_iterations": 4,
    "primary_metric" : 'AUC_weighted',
    "n_cross_validations": 5
}

automl_config = AutoMLConfig(name='Automated ML Experiment',
                             task='classification',
                             compute_target=training_cluster,
                             training_data = train_ds,
                             validation_data = test_ds,
                             label_column_name='Diabetic',
                             iterations=4,
                             primary_metric = 'AUC_weighted',
                             max_concurrent_iterations=2,
                             featurization='auto'

### Results

The best performing algorithm was the VotingEmsemble based on the primary metric AUC Weighted. This algorithm had a AUC weighted of 0.99217 and a accuracy of 0.955872594558726. Other metrics can be seen below.
![Screenshot (49)](https://user-images.githubusercontent.com/111194883/185763959-4974150d-a000-488a-b1b1-f0a12b19903f.png)

The performance of the algorithm could have been improved with feature engineering. More influencial features that could lead to a higher accurancy.
![AutoML Best Model](https://user-images.githubusercontent.com/111194883/185764027-cb665b22-762d-4c41-b518-5c3bd71a7cce.png)
![automl RunDetails](https://user-images.githubusercontent.com/111194883/185764073-f7e96a6e-b565-4bfd-9373-05465a3038f6.png)
![Automl Best Run ID](https://user-images.githubusercontent.com/111194883/185764043-b39dcf32-3682-4503-9ea2-a08688372ec1.png)



## Hyperparameter Tuning

For the hyperparameter Tuning i used the GradientBoostingClassifier. Compared to most models the GradientBoostingClassifier Often provides higher predictive accuracy, Lots of flexibility - can optimize on different loss functions and provides several hyper parameter tuning options that make the function fit very flexible and Handles missing data - imputation not required. I used the learning_rate and n_estimators parameters. Below are some portions of code to show the parameters :

# Hyperparameters
parser.add_argument('--learning_rate', type=float, dest='learning_rate', default=0.1, help='learning rate')
parser.add_argument('--n_estimators', type=int, dest='n_estimators', default=100, help='number of estimators')

 # Hyperdrive will try 6 combinations, adding these as script arguments
        '--learning_rate': choice(0.01, 0.1, 1.0),
        '--n_estimators' : choice(10, 100)

### Results

The results of the model were very impressive actually with a AUC Weighted of 0.9885804604667666 and Accuracy of 0.9457777777777778. Below are screenshots to display the Run details and best run. The parameter are shown below


 # Hyperdrive will try 6 combinations, adding these as script arguments
        '--learning_rate': choice(0.01, 0.1, 1.0),
        '--n_estimators' : choice(10, 100)

![Hyper Run ID](https://user-images.githubusercontent.com/111194883/185764660-43de8c06-4ff7-4ec0-b74f-c9e015e8291a.png)
![Hyper RunDetails](https://user-images.githubusercontent.com/111194883/185764699-0bbfbf7c-c23b-4b8d-a5e5-1e529e01bf8a.png)
![Best model](https://user-images.githubusercontent.com/111194883/185764762-a104ed9d-8d81-4f19-8203-9ec80e750d38.png)

The improvement opportunities are the same as those above regarding feature engineering as it improves model efficiency and a the algorithm will better fit the data

## Model Deployment

The best model deployed was that of the Automated ML run as it had a better AUC Weighted and Accuracy. Here is the overview of the best model:
![Screenshot (51)](https://user-images.githubusercontent.com/111194883/185764880-a21955e0-e7c7-4126-872a-f900395bc245.png)

The model was successfully deployed as depicted in the following figure
![Screenshot (52)](https://user-images.githubusercontent.com/111194883/185764924-0bcb46b5-283f-4048-8ed2-c75d99145c16.png)

The model was deployed as a webservice and the inference config was called . It was then tested using sample data and it managed to predict that a patient is Diabetic as shown below

![Screenshot (53)](https://user-images.githubusercontent.com/111194883/185765006-1010aa1e-0988-46af-a6d1-6ffeab33729d.png)

## Screen Recording
Link to screencast: https://youtu.be/xSKlwb8xhOI
