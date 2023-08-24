# Disaster Response Pipeline Project

## Aim for the Project and Context
Creating a DS pipeline for a Disaster DB cleaning data, putting it 
through ETL then and ML pipeline, resulting in a web app where a new
piece of news can be submitted and checked for potential categorization.


- Extract, Transform, Load (ETL) process:
  - Read dataset, clean data
  - Store cleaned data in SQLite database
  - Data cleaning with pandas
  - Load using pandas .to_sql() with SQLAlchemy
  - Cleaning code in process_data.py


- Machine Learning Pipeline:
  - Split data into train and test sets
  - Build ML pipeline with NLTK and scikit-learn
  - Use Pipeline, GridSearchCV for model (prior to pipeline, in jupyter notebooks)
  - Predict 36 categories from 'message' column
  - Export model to pickle file
  - Final ML code in train_classifier.py

## Files in the Repository
```
- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py   # py file to do the cleaning
|- DisasterResponse.db   # database, created once .py file is run

- models
|- train_classifier.py   # py file to build the model
|- classifier.pkl  # saved model, created once .py file is run
|- ETL Pipeline Preparation solution.ipynb  # not required, went deeper into the data to decide how to clean
|- ML Pipeline Preparation solution.ipynb  # not required, used to check train_classifier.py

- README.md
```

## Instructions
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Go to http://0.0.0.0:3000/