# DataScience

This is a repository where I store all my Data Science projects: data analysis, artificial intelligence etc.

# Table of Contents

-   [Fifa](https://github.com/Fournierp/DataScience/blob/master/Fifa/Fifa%2018%20Value-Wage%20Model/Fifa%2018%20Value-Wage%20Model.md)

## Fifa

Using the Fifa 18 Player Statistics [dataset](https://www.kaggle.com/thec03u5/fifa-18-demo-player-dataset). <br/>
Find the Jupyter Kernel [here]\(<https://www.kaggle.com/fournierp/fifa-18-transfer-value-wage-model/>. <br/>
I did a data analysis looking at the correlation between the characteristics of a player and its value on the transfer market and its salary. Through this Exploratory Data Analysis, I have used graphing libraries (natplotlib, seaborn and plotly) to show Player value per country, Player value versus his age etc. <br/>
Then I did a model to predict its value and wage based on the analysed characteristics. I used a Binary Tree. To determine the optimal hyperparameters for the library XGBoost, I used a Grid Search. After training the model on the training set, I measured its accuracy on the testing set and compared it with a Baseline Mean Average Error.
