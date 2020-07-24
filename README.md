# News Clustering

A simple class project to cluster a sample data-set of 1000 tweets to one of the four classes which are: Sports, Economics, Politics and Culture using **K-Means** & **Mixture Models** in Python. The data-set is crawled from Tasnimnews.com and the crawler used for this task is written in Julia language.


#### Path to training files:
All data files, including trained models are located in the `data` directory.<br>
Raw data are pre-processed once but can be processed again in case there is new data available.<br>


## Instructions:
First of all do a `pip install -r requirements.txt` to install the required modules. You may need to install modules manually if this does not work as expected.
1. To cluster (train) news using K-Means algorithm, uncomment `kMeans.fit()` line in `main.py` module and run it. If you want to predict new data given trained model, uncomment `kMeans.predict()` line in the same file and run the module.
2. To cluster (train) news using Mixture Models algorithm, uncomment `mix.fit()` line in `main.py` module and run it. For prediction of new data, uncomment `mix.predict()` line in the same file and run the module.