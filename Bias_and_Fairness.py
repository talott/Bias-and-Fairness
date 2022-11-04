import pandas as pd
import numpy as np
import tensorflow as tf
import tempeh
import fairlearn
import matplotlib.pyplot as plt

import features
from main import get_Y_x_df
import model
import util

from sklearn.model_selection import train_test_split
#Fairness Metrics
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric
#Explainers
from aif360.explainers import MetricTextExplainer
#Scalers
from sklearn.preprocessing import StandardScaler
#Classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
#Bias Mitigation
from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.inprocessing import PrejudiceRemover

#Importing in the Health Data Set
data = pd.read_csv('data_new.csv')

#Splitting between the two populations (Data only includes black and white)
  #Patients can appear multiple times, one row is a year of data for any given patient 
white = data[data.race == 'white'] #88% of representative data
black = data[data.race == 'black']

#Showing the mean cost by risk score for black and white
  # The algo uses cost in place of health needs, ie higher health costs = higher health needs
  #The risk scores are assessed based on health cost, there is no evidence of bias at this point
bins = np.arange(1, 100, 2)
white_mean_cost = white.groupby(pd.qcut(white.risk_score_t, q=50)).cost_t.mean() / 1e3
black_mean_cost = black.groupby(pd.qcut(black.risk_score_t, q=50)).cost_t.mean() / 1e3

fig, [ax, ax1] = plt.subplots(2,1)
ax.scatter(bins, white_mean_cost, label = 'White')
ax.scatter(bins, black_mean_cost, label = 'Black')
ax.set_title('Mean Medical Cost per Risk Score Percentile')
plt.xlabel('Risk Score Percentile')
ax.set(ylabel='Mean Medical Costs')


#Showing the health outcome (# of chronic illnesses) vs risk score
  #Showing patients health outcome is a better indicator of effectiveness of the algorithm
  #This shows a clear bias as most black patient should be ranked at higher risk due to poor health outcomes compared to white patients in the same percentile
  #gagne is the # of chronic illnesses in the data set
  #Black patients only make 17.7% of population based on medical cost metric, if instead the algo were trained to achieve parity in mean number of chronic illnesses, then 46.5% of patients id' for enrollment would be Black
white_chronic_mean = white.groupby(pd.qcut(white.risk_score_t, q=50)).gagne_sum_t.mean()
black_chronic_mean = black.groupby(pd.qcut(black.risk_score_t, q=50)).gagne_sum_t.mean()

ax1.scatter(bins, white_chronic_mean, label = 'White')
ax1.scatter(bins, black_chronic_mean, label = 'Black')
ax1.set_title('Mean Health Outcomes per Risk Score Percentile')
#plt.xlabel('Risk Score Percentile')
ax1.set(ylabel='Mean Number of Chronic Illnesses')
plt.legend(loc='upper left')
plt.show()

#Graph medical expenditure vs Chronic illnesses

#Get full dataframe, list of all features names, list of predictor names
full_set, x_col_names, y_predictors = get_Y_x_df(data, verbose=True)
full_set = model.split_by_id(full_set, id_field='index',frac_train=.67)

# define train, holdout
    # reset_index for pd.concat() along column
#train_df = full_set[full_set['split'] == 'train'].reset_index(drop=True)
#holdout_df = full_set[full_set['split'] == 'holdout'].reset_index(drop=True)
