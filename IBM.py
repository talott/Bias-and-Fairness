import sys
from collections import defaultdict
from wsgiref import validate
#sys.path.insert(0, '../')

#%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
#from IPython.display import Markdown, display

# Datasets
import pandas as pd
from aif360.datasets import BinaryLabelDataset

# Fairness metrics
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric

# Explainers
from aif360.explainers import MetricTextExplainer

# Scalers
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

# Classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# Bias mitigation techniques
from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.inprocessing import PrejudiceRemover

# LIME
#from aif360.datasets.lime_encoder import LimeEncoder
#import lime
#from lime.lime_tabular import LimeTabularExplainer

np.random.seed(1)


def describe(train=None, val=None, test=None):
    if train is not None:
        print("#### Training Dataset shape")
        print(train.features.shape)
    if val is not None:
        print("#### Validation Dataset shape")
        print(val.features.shape)
    print("#### Test Dataset shape")
    print(test.features.shape)
    print("#### Favorable and unfavorable labels")
    print(test.favorable_label, test.unfavorable_label)
    print("#### Protected attribute names")
    print(test.protected_attribute_names)
    print("#### Privileged and unprivileged protected attribute values")
    print(test.privileged_protected_attributes, 
          test.unprivileged_protected_attributes)
    print("#### Dataset feature names")
    print(test.feature_names)
    print()

def describe_metrics(metrics, thresh_arr):
    best_ind = np.argmax(metrics['bal_acc'])
    print("Threshold corresponding to Best balanced accuracy: {:6.4f}".format(thresh_arr[best_ind]))
    print("Best balanced accuracy: {:6.4f}".format(metrics['bal_acc'][best_ind]))
#     disp_imp_at_best_ind = np.abs(1 - np.array(metrics['disp_imp']))[best_ind]
    disp_imp_at_best_ind = 1 - min(metrics['disp_imp'][best_ind], 1/metrics['disp_imp'][best_ind])
    print("Corresponding 1-min(DI, 1/DI) value: {:6.4f}".format(disp_imp_at_best_ind))
    print("Corresponding average odds difference value: {:6.4f}".format(metrics['avg_odds_diff'][best_ind]))
    print("Corresponding statistical parity difference value: {:6.4f}".format(metrics['stat_par_diff'][best_ind]))
    print("Corresponding equal opportunity difference value: {:6.4f}".format(metrics['eq_opp_diff'][best_ind]))
    print("Corresponding Theil index value: {:6.4f}".format(metrics['theil_ind'][best_ind]))
    print()

#Returns metrics for thresholds used
def test_val(dataset, model, thresh_arr):
    try:
        # sklearn classifier
        y_val_pred_prob = model.predict_proba(dataset.features)
        pos_ind = np.where(model.classes_ == dataset.favorable_label)[0][0]
    except AttributeError:
        # aif360 inprocessing algorithm
        y_val_pred_prob = model.predict(dataset).scores
        pos_ind = 0
    
    metric_arrs = defaultdict(list)
    for thresh in thresh_arr:
        y_val_pred = (y_val_pred_prob[:, pos_ind] > thresh).astype(np.float64)

        dataset_pred = dataset.copy()
        dataset_pred.labels = y_val_pred
        metric = ClassificationMetric(
                dataset, dataset_pred,
                privileged_groups=[{'gendera':2}],
                unprivileged_groups=[{'gendera':1}])

        metric_arrs['bal_acc'].append((metric.true_positive_rate()
                                     + metric.true_negative_rate()) / 2)
        metric_arrs['avg_odds_diff'].append(metric.average_odds_difference())
        metric_arrs['disp_imp'].append(metric.disparate_impact())
        metric_arrs['stat_par_diff'].append(metric.statistical_parity_difference())
        metric_arrs['eq_opp_diff'].append(metric.equal_opportunity_difference())
        metric_arrs['theil_ind'].append(metric.theil_index())
    
    return metric_arrs


# Get the dataset, LabelEncode, and split into train (60%), validate (20%), and test (20%)
data = pd.read_csv('HospitalMortality.csv')

# fill mean to all NaN
numeric_columns = data.select_dtypes(include=['number']).columns
data[numeric_columns] = data[numeric_columns].fillna(data.mean())
data = data.dropna(0)

#data = data.apply(LabelEncoder().fit_transform)
train, validation, test = \
              np.split(data.sample(frac=1, random_state=42), 
                       [int(.6*len(data)), int(.8*len(data))])

# Binary Label Dataset for Training
train_bld = (BinaryLabelDataset(df=train,
                                  label_names=["outcome"],
                                  protected_attribute_names=['gendera'],
                                  favorable_label=0,
                                  unfavorable_label=1))

val_bld = (BinaryLabelDataset(df=validation,
                                    label_names = ["outcome"],
                                    protected_attribute_names=["gendera"],
                                    favorable_label=0,
                                    unfavorable_label=1))

test_bld = (BinaryLabelDataset(df=test,
                                    label_names = ["outcome"],
                                    protected_attribute_names=["gendera"],
                                    favorable_label=0,
                                    unfavorable_label=1))


describe(train_bld, val_bld, test_bld)

# Metrics for original data
train_metric = BinaryLabelDatasetMetric(
    train_bld,
    privileged_groups=[{'gendera':2}],
    unprivileged_groups=[{'gendera':1}])

train_explainer = MetricTextExplainer(train_metric)
print(train_explainer.disparate_impact())

print("Logistic Regression")
#Training Logisitic Regression Model
train_model = train_bld
model = make_pipeline(StandardScaler(),
                      LogisticRegression(solver='liblinear', random_state=1))
fit_params = {'logisticregression__sample_weight': train_model.instance_weights}
lr_model = model.fit(train_model.features, train_model.labels.ravel(), **fit_params)

#Validate
thresh_arr = np.linspace(0.01, 0.5, 50)

val_metrics = test_val(dataset = val_bld,
                   model = lr_model,
                   thresh_arr = thresh_arr)
lr_best_ind = np.argmax(val_metrics['bal_acc'])

describe_metrics(val_metrics, thresh_arr)

#Testing LR model
lr_metrics = test_val(test_bld,lr_model,[thresh_arr[lr_best_ind]])
describe_metrics(lr_metrics, [thresh_arr[lr_best_ind]])

print("Random Forest")
#Random Forest Model
model = make_pipeline(StandardScaler(),
                      RandomForestClassifier(n_estimators=500, min_samples_leaf=25))
fit_params = {'randomforestclassifier__sample_weight': train_model.instance_weights}
rf_model = model.fit(train_model.features, train_model.labels.ravel(), **fit_params)

val_metrics = test_val(train_model, rf_model, thresh_arr)
rf_best_ind = np.argmax(val_metrics['bal_acc'])
describe_metrics(val_metrics, thresh_arr)

#Testing Forest
rf_metrics = test_val(test_bld, rf_model, [thresh_arr[rf_best_ind]])
describe_metrics(rf_metrics, [thresh_arr[rf_best_ind]])

print("Prejudice Remover")
#Bias Mitigation using in-Processing Technique
#training PR
model = PrejudiceRemover(sensitive_attr="gendera", eta=25.0)
pr_orig_scaler = StandardScaler()

dataset = train_bld.copy()
dataset.features = pr_orig_scaler.fit_transform(dataset.features)

pr_orig_panel19 = model.fit(dataset)

#validation PR
dataset = val_bld.copy()
dataset.features = pr_orig_scaler.transform(dataset.features)

val_metrics = test_val(dataset=dataset,
                   model=pr_orig_panel19,
                   thresh_arr=thresh_arr)
pr_orig_best_ind = np.argmax(val_metrics['bal_acc'])
describe_metrics(val_metrics, thresh_arr)

#testing PR
dataset = test_bld.copy()
dataset.features = pr_orig_scaler.transform(dataset.features)

pr_orig_metrics = test_val(dataset=dataset,
                       model=pr_orig_panel19,
                       thresh_arr=[thresh_arr[pr_orig_best_ind]])
describe_metrics(pr_orig_metrics, [thresh_arr[pr_orig_best_ind]])

#Shows table
pd.set_option('display.multi_sparse', False)
results = [lr_metrics, rf_metrics, lr_metrics,
           rf_metrics, pr_orig_metrics]
debias = pd.Series(['']*2 + ['Reweighing']*2
                 + ['Prejudice Remover'],
                   name='Bias Mitigator')
clf = pd.Series(['Logistic Regression', 'Random Forest']*2 + [''],
                name='Classifier')
pd.concat([pd.DataFrame(metrics) for metrics in results], axis=0).set_index([debias, clf])
