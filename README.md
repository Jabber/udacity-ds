# Kickstarter Project Analysis

## Libraries Used
- pandas==1.5.3
- matplotlib==3.7.1
- scikit-learn==1.2.2
- seaborn==0.12.2

Can be installed via requirements.txt

## Motivation for the Project
This project aims to analyze Kickstarter projects and understand the factors that contribute to the success or failure of a project. 
The primary questions:
- Which countries are the most successful ones when running a project?
- Understand how categories are distributed
- See any connected features
- And model a potential prediction about success based on priori data points

## Files in the Repository
- `KickStarter.ipynb`: Jupyter notebook containing the data analysis and visualizations
- `ks-projects-201801.csv`: Dataset containing information about Kickstarter projects
- `requirements.txt`: Dependencies to install before running the notebook

## Summary of Results
The analysis revealed that  
- The US has the highest number of successful projects, followed by the UK and Canada
- Film & Video, Music and Publishing are the biggest major categories represented
- Product Design, Documentary and Tabletop Games are the biggest minor categories represented
- There are multiple columns which can be disregarded as they contain redundant information
- It's possible to slightly outperform the pure statistical prediction with a binary logistic regression built on the `minor category`, `launched`, `country`, `usd_goal_real` features

## Acknowledgements
The data used is available on [Kaggle under MICKAËL MOUILLÉ](https://www.kaggle.com/datasets/kemical/kickstarter-projects?select=ks-projects-201801.csv), 
collected from the Kickstarter Platform. The USD conversion (`usd_pledged_real` and `usd_goal_real` columns) were generated from the [convert ks pledges to usd script](https://www.kaggle.com/tonyplaysguitar/convert-ks-pledges-to-usd) done by tonyplaysguitar. Special thanks to Mickaël Mouillé (the data creator) for the valuable dataset and inspiration.

## About the Dataset
The dataset contains information about Kickstarter projects.

- `ID`: Unique identifier for each project
- `name`: Name of the project
- `category`: Specific category of the project
- `main_category`: Major category of the project
- `currency`: Currency of goal and pledges
- `deadline`: Date by which the project hopes to achieve its funding goal
- `goal`: Funding goal
- `launched`: Date and time when the project was launched for crowdfunding
- `pledged`: Amount pledged by funders
- `state`: State of the project (e.g., 'failed', 'canceled')
- `backers`: Number of backers
- `country`: Country where the project is based
- `usd_pledged`: Conversion in US dollars of the pledged column (conversion done by Kickstarter)
- `usd_pledged_real`: Conversion in US dollars of the pledged column (conversion from Fixer.io API)
- `usd_goal_real`: Conversion in US dollars of the goal column (conversion from Fixer.io API)

Note: The value `"N,0""` in the `country` column seems to be an error in the Kickstarter data

## License
This project is licensed under the terms of the CC BY-NC-SA 4.0 license.