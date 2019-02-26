## Fifa 18 Transfer Value and Wage ML Model

![](https://media.playstation.com/is/image/SCEA/ea-sports-fifa-18-llisting-thumb-01-us-02jun17?$Icon$)

This kernel uses Electronic Arts' *Fifa 18* Player Statistics [dataset](https://www.kaggle.com/thec03u5/fifa-18-demo-player-dataset) to predict any player's Transfer Market Value as well as his Wage.

The first part of the kernel is an Exploratory Data Analysis looking at the correlation between the characteristics of a player and its value on the transfer market and its salary. Through this Exploratory Data Analysis, I have used graphing libraries (matplotlib, seaborn and plotly) to vizualise Player value per country, Player value versus his age etc. Head over to [Kaggle](https://www.kaggle.com/fournierp/fifa-18-transfer-value-wage-model/) to see the Plotly graphs within the Notebook.

The second part is a Machine Learning Model designed to predict player value and wage based on the analysed characteristics. I used a Binary Tree. To determine the optimal hyperparameters for the library XGBoost, I used a Grid Search. After training the model on the training set, I measured its accuracy on the testing set and compared it with a Baseline Mean Average Error.