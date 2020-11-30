import scipy.sparse
import pandas as pd
import numpy as np
import sys
from data_configuration.data_conf import variable_setting, variable_normalize
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
from time import time
import os
from sys import argv
import itertools
import xgboost as xgb
from DatasetManager import DatasetManager
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV #For random search of hyperparameters
import itertools
import time
import multiprocessing

system_start = time.time()
dataset_name = argv[1] #Type in the name of the file that is generated from "extract_resource_features.py"
cls_method = argv[2]
results_dir = argv[3]
max_prefix = argv[4]

num_cpu_cores = 10

max_prefix = int(max_prefix)

dataset_address = "~/Resource/codes/index_encoded_data_" + dataset_name + "_" + str(max_prefix) + ".pkl"

# Reading data
# Be careful of "sep" parameter. "sep" parameter can be ";" or "," at times.
prefix_index_encoding = pd.read_pickle(dataset_address)
print(prefix_index_encoding.columns[0:10])

prefix_index_encoding = prefix_index_encoding.sample(frac=1).reset_index(drop=True)

data_dummy_prefix_no_resource, data_dummy_prefix_yes_resource = variable_setting(dataset_name, prefix_index_encoding)

# Deleting "case_len" feature
data_dummy_prefix_yes_resource = data_dummy_prefix_yes_resource.drop("case_len_prefixlen1",
                                                                     1)  # "case_len" feature only exists in data_dummy_prefix_yes_resource, not in data_dummy_no_resource

data_dummy_prefix_no_resource.to_csv('data_dummy_prefix_no_resource.csv', index=False)
data_dummy_prefix_yes_resource.to_csv('data_dummy_prefix_yes_resource.csv', index=False)

# Deleting NAs
data_dummy_prefix_no_resource = data_dummy_prefix_no_resource.dropna()
data_dummy_prefix_yes_resource = data_dummy_prefix_yes_resource.dropna()

# Splitting the data into X and y
y_dummy_no_resource = pd.DataFrame(
    [0 if i == "regular" else 1 for i in data_dummy_prefix_no_resource['label_prefixlen1']])
y_dummy_yes_resource = pd.DataFrame(
    [0 if i == "regular" else 1 for i in data_dummy_prefix_yes_resource['label_prefixlen1']])

X_dummy_no_resource = data_dummy_prefix_no_resource[data_dummy_prefix_no_resource.columns.drop('label_prefixlen1')]
X_dummy_yes_resource = data_dummy_prefix_yes_resource[data_dummy_prefix_yes_resource.columns.drop('label_prefixlen1')]

# Normalization for features only available in BPIC15_1_f2_exp_prefix
# If the feature is both case-level attribute and numeric, only one line will be written for normalization as below.
# If the feature is not a case-level attribute but numeric, five lines will be written (starting with for statement) for normalization as below.

X_dummy_no_resource = variable_normalize(X_dummy_no_resource)

print("X_dummy_no_resource after prefix shape is :" + str(X_dummy_no_resource.shape))
print(X_dummy_no_resource.columns)

print("X_dummy_yes_resource after prefix shape is :" + str(X_dummy_yes_resource.shape))
print(X_dummy_yes_resource.columns)
X_dummy_yes_resource_column_list = pd.DataFrame(X_dummy_yes_resource.columns)
X_dummy_yes_resource_column_list.to_csv('columns_X_dummy_yes_resource.csv', index=False)

X_dummy_no_resource.to_csv("X_dummy_no_resource_" + dataset_name + "_" + str(max_prefix) + ".csv")
y_dummy_no_resource.to_csv("y_dummy_no_resource_" + dataset_name + "_" + str(max_prefix) + ".csv")
X_dummy_yes_resource.to_csv("X_dummy_yes_resource_" + dataset_name + "_" + str(max_prefix) + ".csv")
y_dummy_yes_resource.to_csv("y_dummy_yes_resource_" + dataset_name + "_" + str(max_prefix) + ".csv")

## Selecting top20 important features only
if dataset_name == "BPIC17_O_Accepted_exp_prefix":
    # RF TOP 20 important features
    X_dummy_no_resource = X_dummy_no_resource[['open_cases_prefixlen1','timesincemidnight_prefixlen1','RequestedAmount_prefixlen1','hour_prefixlen1','month_prefixlen1','weekday_prefixlen1','CreditScore_prefixlen1','timesincecasestart_prefixlen2','timesincemidnight_prefixlen2','open_cases_prefixlen2','timesincelastevent_prefixlen2','RequestedAmount_prefixlen2','timesincecasestart_prefixlen1','hour_prefixlen2','timesincelastevent_prefixlen1','event_nr_prefixlen1','month_prefixlen2','CreditScore_prefixlen2','EventOrigin_Application_prefixlen1','weekday_prefixlen2']]
    X_dummy_yes_resource = X_dummy_yes_resource[['timesincemidnight_prefixlen1','n_tasks_prefixlen1','n_current_act_prefixlen1','n_tasks_recent_prefixlen1','n_cases_prefixlen1','n_current_handoff_prefixlen1','n_current_act_recent_prefixlen1','n_current_handoff_recent_prefixlen1','n_cases_recent_prefixlen1','open_cases_prefixlen1','RequestedAmount_prefixlen1','busyness_recent_prefixlen1','hour_prefixlen1','busyness_prefixlen1','CreditScore_prefixlen1','Selected_False_prefixlen1','Selected_True_prefixlen1','weekday_prefixlen1','n_handoffs_prefixlen1','timesincecasestart_prefixlen2']]

if dataset_name == "BPIC17_O_Cancelled_exp_prefix":
    # RF TOP 20 important features
    X_dummy_no_resource = X_dummy_no_resource[['timesincemidnight_prefixlen1','open_cases_prefixlen1','timesincecasestart_prefixlen2','timesincecasestart_prefixlen3','RequestedAmount_prefixlen1','Activity_A_Cancelled_prefixlen7','CreditScore_prefixlen7','CreditScore_prefixlen9','timesincecasestart_prefixlen4','hour_prefixlen1','timesincelastevent_prefixlen7','timesincemidnight_prefixlen2','timesincelastevent_prefixlen2','timesincecasestart_prefixlen7','open_cases_prefixlen2','timesincecasestart_prefixlen1','timesincelastevent_prefixlen1','CreditScore_prefixlen8','timesincelastevent_prefixlen3','month_prefixlen1']]
    X_dummy_yes_resource = X_dummy_yes_resource[['Selected_True_prefixlen7','Selected_False_prefixlen8','Selected_False_prefixlen7','Selected_False_prefixlen1','Selected_True_prefixlen2','Selected_True_prefixlen1','Selected_True_prefixlen8','Selected_True_prefixlen9','CreditScore_prefixlen7','CreditScore_prefixlen8','timesincecasestart_prefixlen2','timesincemidnight_prefixlen1','Selected_False_prefixlen4','polarity_current_handoff_prefixlen1','Selected_False_prefixlen9','n_current_act_recent_prefixlen1','n_cases_prefixlen1','n_current_act_prefixlen1','busyness_recent_prefixlen1','n_tasks_recent_prefixlen1']]

if dataset_name == "BPIC17_O_Refused_exp_prefix":
    # RF TOP 20 important features
    X_dummy_no_resource = X_dummy_no_resource[['timesincemidnight_prefixlen1','open_cases_prefixlen1','RequestedAmount_prefixlen1','hour_prefixlen1','timesincecasestart_prefixlen2','month_prefixlen1','timesincelastevent_prefixlen2','timesincemidnight_prefixlen2','open_cases_prefixlen2','weekday_prefixlen1','RequestedAmount_prefixlen2','timesincecasestart_prefixlen1','timesincecasestart_prefixlen3','hour_prefixlen2','timesincelastevent_prefixlen3','timesincelastevent_prefixlen1','timesincemidnight_prefixlen3','open_cases_prefixlen3','month_prefixlen2','weekday_prefixlen2']]
    X_dummy_yes_resource = X_dummy_yes_resource[['timesincemidnight_prefixlen1','polarity_current_handoff_prefixlen1','n_current_act_recent_prefixlen1','n_current_act_prefixlen1','n_cases_prefixlen1','n_tasks_prefixlen1','n_cases_recent_prefixlen1','n_tasks_recent_prefixlen1','n_current_handoff_recent_prefixlen1','n_current_handoff_prefixlen1','open_cases_prefixlen1','RequestedAmount_prefixlen1','busyness_recent_prefixlen1','Selected_False_prefixlen1','hour_prefixlen1','busyness_prefixlen1','timesincecasestart_prefixlen2','timesincemidnight_prefixlen2','timesincelastevent_prefixlen2','n_cases_recent_prefixlen2']]

if dataset_name == "BPIC11_f1_exp_prefix":
    # RF TOP 20 important features
    X_dummy_no_resource = X_dummy_no_resource[['Age_prefixlen1','open_cases_prefixlen1','group_General Lab Clinical Chemistry_prefixlen1','event_nr_prefixlen1','event_nr_prefixlen2','timesincecasestart_prefixlen1','open_cases_prefixlen2','timesincecasestart_prefixlen2','month_prefixlen1','weekday_prefixlen1','event_nr_prefixlen3','open_cases_prefixlen4','month_prefixlen2','open_cases_prefixlen3','event_nr_prefixlen4','open_cases_prefixlen5','weekday_prefixlen2','timesincecasestart_prefixlen4','Section_Section 2_prefixlen1','month_prefixlen3']]
    X_dummy_yes_resource = X_dummy_yes_resource[['Diagnosis_code_M11_prefixlen1','Age_prefixlen1','Specialism_code_b_SC86_prefixlen1','n_tasks_prefixlen1','ent_case_recent_prefixlen1','Diagnosis_code_M16_prefixlen1','n_cases_recent_prefixlen1','timesincecasestart_prefixlen2','n_current_act_prefixlen1','prev_resource_CHE2_prefixlen1','group_General Lab Clinical Chemistry_prefixlen1','n_current_act_recent_prefixlen1','n_acts_prefixlen1','n_cases_prefixlen1','n_current_handoff_prefixlen1','n_tasks_recent_prefixlen1','ratio_act_case_recent_prefixlen1','n_current_act_recent_prefixlen2','open_cases_prefixlen1','event_nr_prefixlen1']]

if dataset_name == "BPIC11_f2_exp_prefix":
    # RF TOP 20 important features
    X_dummy_no_resource = X_dummy_no_resource[['Age_prefixlen1','open_cases_prefixlen1','event_nr_prefixlen1','open_cases_prefixlen2','event_nr_prefixlen2','open_cases_prefixlen3','month_prefixlen1','timesincecasestart_prefixlen1','event_nr_prefixlen3','open_cases_prefixlen4','weekday_prefixlen1','event_nr_prefixlen4','timesincecasestart_prefixlen2','month_prefixlen2','timesincecasestart_prefixlen4','open_cases_prefixlen5','timesincecasestart_prefixlen5','timesincecasestart_prefixlen3','month_prefixlen3','event_nr_prefixlen5']]
    X_dummy_yes_resource = X_dummy_yes_resource[['Diagnosis_code_M16_prefixlen1','Diagnosis_code_M13_prefixlen1','Age_prefixlen1','Diagnosis_code_M16_prefixlen2','Diagnosis_code_M11_prefixlen1','n_current_act_prefixlen1','n_tasks_prefixlen1','n_current_handoff_prefixlen1','n_cases_recent_prefixlen1','n_tasks_recent_prefixlen1','n_cases_prefixlen1','open_cases_prefixlen1','n_current_act_recent_prefixlen1','n_current_handoff_recent_prefixlen1','timesincecasestart_prefixlen1','Diagnosis_code_M13_prefixlen2','n_current_handoff_prefixlen2','event_nr_prefixlen1','open_cases_prefixlen2','n_cases_prefixlen3']]

if dataset_name == "BPIC11_f3_exp_prefix":
    # RF TOP 20 important features
    X_dummy_no_resource = X_dummy_no_resource[['Age_prefixlen1','event_nr_prefixlen1','open_cases_prefixlen1','timesincecasestart_prefixlen1','event_nr_prefixlen2','open_cases_prefixlen2','month_prefixlen1','open_cases_prefixlen3','timesincecasestart_prefixlen2','event_nr_prefixlen3','weekday_prefixlen1','event_nr_prefixlen4','group_General Lab Clinical Chemistry_prefixlen1','open_cases_prefixlen4','month_prefixlen2','open_cases_prefixlen5','event_nr_prefixlen5','Treatment_code_TC101_prefixlen1','month_prefixlen3','month_prefixlen4']]
    X_dummy_yes_resource = X_dummy_yes_resource[['Diagnosis_code_M13_prefixlen1','Age_prefixlen1','Diagnosis_code_M13_prefixlen2','Diagnosis_code_M16_prefixlen1','n_current_handoff_prefixlen1','n_current_act_recent_prefixlen1','n_tasks_prefixlen1','n_current_handoff_recent_prefixlen1','ratio_act_case_recent_prefixlen1','event_nr_prefixlen1','timesincecasestart_prefixlen1','n_current_act_prefixlen1','n_tasks_recent_prefixlen1','open_cases_prefixlen1','n_cases_recent_prefixlen1','n_cases_prefixlen1','Specialism_code_a_SC61_prefixlen1','n_handoffs_recent_prefixlen1','prev_resource_CHE2_prefixlen1','n_current_handoff_recent_prefixlen2']]

if dataset_name == "BPIC11_f4_exp_prefix":
    # RF TOP 20 important features
    X_dummy_no_resource = X_dummy_no_resource[['Treatment_code_TC803_prefixlen1','open_cases_prefixlen1','Treatment_code_TC803_prefixlen2','Age_prefixlen1','event_nr_prefixlen1','Treatment_code_TC101_prefixlen1','timesincecasestart_prefixlen1','event_nr_prefixlen2','open_cases_prefixlen2','timesincecasestart_prefixlen2','month_prefixlen1','open_cases_prefixlen3','event_nr_prefixlen3','event_nr_prefixlen4','timesincecasestart_prefixlen4','open_cases_prefixlen4','month_prefixlen2','weekday_prefixlen1','Treatment_code_TC101_prefixlen2','event_nr_prefixlen5']]
    X_dummy_yes_resource = X_dummy_yes_resource[['Treatment_code_TC803_prefixlen1','Treatment_code_TC803_prefixlen2','Producer_BLOB_prefixlen1','Treatment_code_TC101_prefixlen1','n_tasks_recent_prefixlen1','n_current_act_prefixlen1','ratio_act_case_recent_prefixlen1','n_current_act_recent_prefixlen1','event_nr_prefixlen1','n_cases_recent_prefixlen1','n_cases_prefixlen1','n_tasks_prefixlen1','n_tasks_prefixlen2','Age_prefixlen1','busyness_recent_prefixlen1','n_current_handoff_recent_prefixlen1','Specialism_code_a_SC7_prefixlen1','n_current_handoff_prefixlen1','n_current_handoff_prefixlen2','n_acts_recent_prefixlen1']]

if dataset_name == "BPIC15_1_f2_exp_prefix":
    # RF TOP 20 important features
    X_dummy_no_resource = X_dummy_no_resource[['open_cases_prefixlen1','SUMleges_prefixlen1','weekday_prefixlen1','timesincecasestart_prefixlen3','timesincecasestart_prefixlen2','Milieu..vergunning._prefixlen1','month_prefixlen2','month_prefixlen1','open_cases_prefixlen2','timesincemidnight_prefixlen2','open_cases_prefixlen3','timesincelastevent_prefixlen2','timesincelastevent_prefixlen6','timesincecasestart_prefixlen4','timesincecasestart_prefixlen5','open_cases_prefixlen4','timesincemidnight_prefixlen4','timesincelastevent_prefixlen3','hour_prefixlen2','timesincelastevent_prefixlen4']]
    X_dummy_yes_resource = X_dummy_yes_resource[['SUMleges_prefixlen1','n_current_handoff_prefixlen1','n_cases_prefixlen1','n_current_act_recent_prefixlen1','open_cases_prefixlen1','ratio_act_case_recent_prefixlen1','n_tasks_recent_prefixlen1','Milieu..vergunning._prefixlen1','n_current_act_prefixlen3','timesincecasestart_prefixlen3','n_tasks_prefixlen1','weekday_prefixlen1','n_tasks_recent_prefixlen2','n_current_handoff_recent_prefixlen3','n_handoffs_recent_prefixlen3','n_current_handoff_recent_prefixlen1','timesincecasestart_prefixlen2','n_current_act_prefixlen1','n_cases_recent_prefixlen1','n_handoffs_recent_prefixlen1']]

if dataset_name == "BPIC15_2_f2_exp_prefix":
    # RF TOP 20 important features
    X_dummy_no_resource = X_dummy_no_resource[['Milieu..vergunning._prefixlen1','SUMleges_prefixlen1','timesincecasestart_prefixlen1','open_cases_prefixlen1','event_nr_prefixlen1','timesincecasestart_prefixlen2','timesincelastevent_prefixlen2','event_nr_prefixlen2','open_cases_prefixlen2','event_nr_prefixlen4','timesincecasestart_prefixlen3','timesincemidnight_prefixlen1','event_nr_prefixlen3','timesincelastevent_prefixlen1','timesincecasestart_prefixlen5','timesincelastevent_prefixlen3','timesincecasestart_prefixlen4','weekday_prefixlen1','open_cases_prefixlen3','month_prefixlen1']]
    X_dummy_yes_resource = X_dummy_yes_resource[['Milieu..vergunning._prefixlen1','SUMleges_prefixlen1','n_current_act_prefixlen1','n_current_act_prefixlen2','n_current_handoff_prefixlen1','timesincecasestart_prefixlen1','event_nr_prefixlen1','n_current_handoff_recent_prefixlen2','event_nr_prefixlen2','n_tasks_prefixlen1','n_current_act_recent_prefixlen1','open_cases_prefixlen1','n_current_handoff_prefixlen2','n_tasks_recent_prefixlen1','n_acts_prefixlen1','n_cases_prefixlen1','timesincemidnight_prefixlen1','event_nr_prefixlen4','n_current_handoff_recent_prefixlen1','timesincelastevent_prefixlen2']]

if dataset_name == "BPIC15_3_f2_exp_prefix":
    # RF TOP 20 important features
    X_dummy_no_resource = X_dummy_no_resource[['SUMleges_prefixlen1','open_cases_prefixlen1','event_nr_prefixlen1','event_nr_prefixlen2','event_nr_prefixlen3','timesincecasestart_prefixlen2','open_cases_prefixlen2','open_cases_prefixlen3','month_prefixlen1','timesincecasestart_prefixlen1','timesincecasestart_prefixlen3','open_cases_prefixlen4','event_nr_prefixlen4','timesincelastevent_prefixlen1','timesincecasestart_prefixlen4','weekday_prefixlen1','event_nr_prefixlen5','timesincelastevent_prefixlen2','timesincecasestart_prefixlen5','timesincelastevent_prefixlen3']]
    X_dummy_yes_resource = X_dummy_yes_resource[['event_nr_prefixlen2','SUMleges_prefixlen1','n_current_act_recent_prefixlen1','n_cases_prefixlen1','event_nr_prefixlen4','n_tasks_prefixlen1','event_nr_prefixlen3','event_nr_prefixlen1','n_tasks_recent_prefixlen1','n_current_act_recent_prefixlen2','n_current_act_prefixlen1','open_cases_prefixlen1','n_current_handoff_prefixlen2','n_cases_recent_prefixlen1','n_current_act_prefixlen2','timesincecasestart_prefixlen2','n_current_handoff_prefixlen1','n_tasks_prefixlen2','n_acts_recent_prefixlen1','n_current_handoff_recent_prefixlen2']]

if dataset_name == "BPIC15_4_f2_exp_prefix":
    # RF TOP 20 important features
    X_dummy_no_resource = X_dummy_no_resource[['SUMleges_prefixlen1','event_nr_prefixlen1','open_cases_prefixlen1','timesincecasestart_prefixlen1','timesincecasestart_prefixlen2','event_nr_prefixlen2','timesincecasestart_prefixlen3','open_cases_prefixlen2','timesincecasestart_prefixlen4','weekday_prefixlen1','month_prefixlen1','timesincemidnight_prefixlen2','event_nr_prefixlen3','event_nr_prefixlen4','timesincelastevent_prefixlen4','timesincecasestart_prefixlen5','weekday_prefixlen2','timesincemidnight_prefixlen3','timesincemidnight_prefixlen1','open_cases_prefixlen3']]
    X_dummy_yes_resource = X_dummy_yes_resource[['SUMleges_prefixlen1','event_nr_prefixlen1','timesincecasestart_prefixlen2','event_nr_prefixlen2','timesincecasestart_prefixlen1','polarity_current_handoff_prefixlen1','timesincecasestart_prefixlen3','n_cases_recent_prefixlen1','n_tasks_recent_prefixlen1','event_nr_prefixlen4','n_handoffs_recent_prefixlen1','event_nr_prefixlen3','n_current_act_prefixlen1','n_tasks_prefixlen1','n_current_handoff_recent_prefixlen1','open_cases_prefixlen1','n_cases_prefixlen1','n_current_act_recent_prefixlen1','n_current_handoff_prefixlen1','event_nr_prefixlen5']]

if dataset_name == "BPIC15_5_f2_exp_prefix":
    # RF TOP 20 important features
    X_dummy_no_resource = X_dummy_no_resource[['SUMleges_prefixlen1','event_nr_prefixlen1','timesincecasestart_prefixlen1','open_cases_prefixlen1','event_nr_prefixlen2','timesincecasestart_prefixlen2','month_prefixlen1','open_cases_prefixlen2','event_nr_prefixlen3','timesincelastevent_prefixlen1','timesincemidnight_prefixlen1','timesincecasestart_prefixlen3','weekday_prefixlen1','hour_prefixlen1','open_cases_prefixlen3','timesincelastevent_prefixlen2','month_prefixlen2','timesincemidnight_prefixlen2','timesincecasestart_prefixlen4','event_nr_prefixlen4']]
    X_dummy_yes_resource = X_dummy_yes_resource[['SUMleges_prefixlen1','event_nr_prefixlen1','timesincecasestart_prefixlen1','event_nr_prefixlen2','event_nr_prefixlen3','n_tasks_prefixlen1','n_current_handoff_prefixlen1','n_current_act_prefixlen2','n_tasks_recent_prefixlen1','n_handoffs_recent_prefixlen1','n_cases_prefixlen1','n_current_act_recent_prefixlen1','n_acts_prefixlen1','n_acts_recent_prefixlen1','timesincecasestart_prefixlen2','n_current_act_prefixlen1','open_cases_prefixlen1','ratio_act_case_recent_prefixlen1','n_cases_recent_prefixlen1','ratio_act_case_prefixlen1']]

if dataset_name == "traffic_fines_1_exp_prefix":
    # RF TOP 20 important features
    X_dummy_no_resource = X_dummy_no_resource[['open_cases_prefixlen1','amount_prefixlen1','month_prefixlen1','weekday_prefixlen1','timesincecasestart_prefixlen2','open_cases_prefixlen2','Activity_Payment_prefixlen2','timesincelastevent_prefixlen2','expense_prefixlen2','amount_prefixlen2','month_prefixlen2','month_prefixlen5','weekday_prefixlen2','timesincemidnight_prefixlen5','amount_prefixlen3','open_cases_prefixlen3','month_prefixlen3','article_7.0_prefixlen1','timesincecasestart_prefixlen3','timesincelastevent_prefixlen3']]
    X_dummy_yes_resource = X_dummy_yes_resource[['open_cases_prefixlen1','n_current_act_prefixlen1','n_tasks_recent_prefixlen1','n_tasks_prefixlen1','amount_prefixlen1','n_cases_recent_prefixlen1','n_current_act_recent_prefixlen1','n_current_handoff_prefixlen1','polarity_current_handoff_prefixlen1','n_current_handoff_recent_prefixlen1','n_cases_prefixlen1','Activity_Payment_prefixlen2','is_last_event_True_prefixlen2','n_current_act_prefixlen2','weekday_prefixlen1','open_cases_prefixlen5','month_prefixlen1','polarity_current_handoff_prefixlen2','expense_prefixlen2','n_current_handoff_recent_prefixlen2']]

##Pearson correlation coefficient calculation as a reference for feature selection
pearson_corr = pd.DataFrame(index=range(len(X_dummy_no_resource.columns)),
                            columns=["feature_name", "Pearson_correlation"])
pearson_corr.iloc[:, 0] = X_dummy_no_resource.columns
print(y_dummy_no_resource.columns)
for i in range(len(X_dummy_no_resource.columns) - 1):
    print(pearsonr(X_dummy_no_resource[X_dummy_no_resource.columns[i]],
                   pd.DataFrame(y_dummy_no_resource)[pd.DataFrame(y_dummy_no_resource).columns[0]]))
    pearson_corr.iloc[i, 1] = pearsonr(X_dummy_no_resource[X_dummy_no_resource.columns[i]],
                                       pd.DataFrame(y_dummy_no_resource)[pd.DataFrame(y_dummy_no_resource).columns[0]])[
        0]
pearson_name = "pearson_correlation_coefficient_no_resource_" + dataset_name + "_" + str(max_prefix) + ".csv"
pearson_corr.to_csv(pearson_name, index=False)

##Pearson correlation coefficient calculation as a reference for feature selection
pearson_corr = pd.DataFrame(index=range(len(X_dummy_yes_resource.columns)),
                            columns=["feature_name", "Pearson_correlation"])
pearson_corr.iloc[:, 0] = X_dummy_yes_resource.columns
print(y_dummy_yes_resource.columns)
for i in range(len(X_dummy_yes_resource.columns) - 1):
    print(pearsonr(X_dummy_yes_resource[X_dummy_yes_resource.columns[i]],
                   pd.DataFrame(y_dummy_yes_resource)[pd.DataFrame(y_dummy_yes_resource).columns[0]]))
    pearson_corr.iloc[i, 1] = pearsonr(X_dummy_yes_resource[X_dummy_yes_resource.columns[i]],
                                       pd.DataFrame(y_dummy_yes_resource)[
                                           pd.DataFrame(y_dummy_yes_resource).columns[0]])[0]
pearson_name = "pearson_correlation_coefficient_yes_resource_" + dataset_name + "_" + str(max_prefix) + ".csv"
pearson_corr.to_csv(pearson_name, index=False)

# [no resource] Setting k for k-fold cv
nsplits_no_resource = 2  # Set the number of k for cross validation
kf_no_resource = KFold(n_splits=nsplits_no_resource)
kf_no_resource.get_n_splits(X_dummy_no_resource)

# [yes resource] Setting k for k-fold cv
nsplits_yes_resource = 2  # Set the number of k for cross validation
kf_yes_resource = KFold(n_splits=nsplits_yes_resource)
kf_yes_resource.get_n_splits(X_dummy_yes_resource)

# [no resource] Creating an empty list to store each result of k-fold cv
dnn_auc_no_resource_arr = [None] * (nsplits_no_resource + 2)
rf_auc_no_resource_arr = [None] * (nsplits_no_resource + 2)
xgb_auc_no_resource_arr = [None] * (nsplits_no_resource + 2)

# [yes resource] Creating an empty list to store each result of k-fold cv
dnn_auc_yes_resource_arr = [None] * (nsplits_yes_resource + 2)
rf_auc_yes_resource_arr = [None] * (nsplits_yes_resource + 2)
xgb_auc_yes_resource_arr = [None] * (nsplits_yes_resource + 2)

# [no resource] Starting k-fold cv
index_no_resource = 0  # This index value is used to store auc into list created right above
for train_index, test_index in kf_no_resource.split(X_dummy_no_resource):
    # Splitting data into training and test set
    X_train_no_resource, X_test_no_resource = X_dummy_no_resource.iloc[train_index], X_dummy_no_resource.iloc[
        test_index]
    y_train_no_resource, y_test_no_resource = y_dummy_no_resource.iloc[train_index], y_dummy_no_resource.iloc[
        test_index]
    train_no_resource = pd.concat([X_train_no_resource, y_train_no_resource], axis=1)
    test_no_resource = pd.concat([X_test_no_resource, y_test_no_resource], axis=1)

    # Deleting columns with feature importances = 0 for no resource
    # These column indices are from Random Forest feature importances function
    cols_del_no_resource = []  # Indices with feature importances = 0 were not found yet. Please fill in the bracket with indices later.
    X_train_no_resource = X_train_no_resource.drop(X_train_no_resource.columns[cols_del_no_resource], axis=1)
    X_test_no_resource = X_test_no_resource.drop(X_test_no_resource.columns[cols_del_no_resource], axis=1)

    print("X_train_no_resource after prefix shape is :" + str(X_train_no_resource.shape))
    print(len(X_train_no_resource))
    print("X_test_no_resource after prefix shape is :" + str(X_test_no_resource.shape))
    print(len(X_test_no_resource))

    first_digit_parameters = [x for x in itertools.product((20, 40, 60, 80, 100), repeat=1)]
    second_digit_parameters = [x for x in itertools.product((1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                                                             18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
                                                             33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
                                                             48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
                                                             63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77,
                                                             78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92,
                                                             93, 94, 95, 96, 97, 98, 99, 100), repeat=2)]
    third_digit_parameters = [x for x in itertools.product((1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                                                            18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
                                                            33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
                                                            48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
                                                            63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77,
                                                            78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92,
                                                            93, 94, 95, 96, 97, 98, 99, 100), repeat=3)]
    all_digit_parameters = first_digit_parameters  # one hidden layer
    # all_digit_parameters = first_digit_parameters + second_digit_parameters # two hidden layers
    # all_digit_parameters = first_digit_parameters + second_digit_parameters + third_digit_parameters # three hidden layers
    learning_rate_init_parameters = [0.1, 0.01, 0.001]

    parameters = {'hidden_layer_sizes': all_digit_parameters,
                  'learning_rate_init': learning_rate_init_parameters}

    print("DNN started")
    start = time.time()

    # Making dnn classifier
    dnn_classifier_no_resource = MLPClassifier(max_iter=10000, activation='relu')
    dnn_classifier_no_resource_clf = RandomizedSearchCV(dnn_classifier_no_resource, parameters, n_jobs=num_cpu_cores,
                                                        cv=2)

    # Fitting the dnn classifier with training set
    dnn_classifier_no_resource_clf.fit(X_train_no_resource, y_train_no_resource.values.ravel())
    print("DNN best parameter is:")
    print(dnn_classifier_no_resource_clf.best_params_)

    print("DNN time: ", time.time() - start)

    n_tree = [100, 300, 500]
    max_depth = [20, 40, 60]
    min_samples_split = [5, 10, 20]

    parameters = {'n_estimators': n_tree,
                  'max_depth': max_depth,
                  'min_samples_split': min_samples_split}

    print("RF started")
    start = time.time()

    # Making rf classifier
    rf_classifier_no_resource = RandomForestClassifier(n_jobs=num_cpu_cores)
    rf_classifier_no_resource_clf = RandomizedSearchCV(rf_classifier_no_resource, parameters, n_jobs=num_cpu_cores,
                                                       cv=2)

    # Fitting the rf classifier with training set
    rf_classifier_no_resource_clf.fit(X_train_no_resource, y_train_no_resource.values.ravel())
    print("RF best parameter is:")
    print(rf_classifier_no_resource_clf.best_params_)

    print("RF time: ", time.time() - start)

    parameters = {'n_estimators': n_tree,
                  'learning_rate': learning_rate_init_parameters,
                  'max_depth': max_depth}

    print("XGB started")
    start = time.time()

    # Making xgb classifier
    xgb_classifier_no_resource = XGBClassifier(n_jobs=num_cpu_cores)
    # xgb_classifier_no_resource_clf = xgb_classifier_no_resource
    xgb_classifier_no_resource_clf = RandomizedSearchCV(xgb_classifier_no_resource, parameters, n_jobs=num_cpu_cores,
                                                        cv=2)

    # Fitting the xgb classifier with training set
    xgb_classifier_no_resource_clf.fit(X_train_no_resource, y_train_no_resource.values.ravel())
    print("XGB best parameter is:")
    print(xgb_classifier_no_resource_clf.best_params_)

    print("XGB time: ", time.time() - start)

    # Prediction using dnn classifier
    dnn_pred_no_resource = dnn_classifier_no_resource_clf.predict(X_test_no_resource)

    # Prediction using rf classifier
    rf_pred_no_resource = rf_classifier_no_resource_clf.predict(X_test_no_resource)

    # Prediction using xgb classifier
    xgb_pred_no_resource = xgb_classifier_no_resource_clf.predict(X_test_no_resource)

    # Calculating auc
    dnn_auc_no_resource = roc_auc_score(y_test_no_resource, dnn_pred_no_resource)
    rf_auc_no_resource = roc_auc_score(y_test_no_resource, rf_pred_no_resource)
    xgb_auc_no_resource = roc_auc_score(y_test_no_resource, xgb_pred_no_resource)
    print("AUC of DNN classifier of" + " fold " + str(index_no_resource + 1) + " no resource is: " + str(
        dnn_auc_no_resource))
    print("AUC of RF classifier of" + " fold " + str(index_no_resource + 1) + " no resource is: " + str(
        rf_auc_no_resource))
    print("AUC of XGB classifier of" + " fold " + str(index_no_resource + 1) + " no resource is: " + str(
        xgb_auc_no_resource))

    # Storing prediction results
    dnn_auc_no_resource_arr[index_no_resource] = dnn_auc_no_resource
    rf_auc_no_resource_arr[index_no_resource] = rf_auc_no_resource
    xgb_auc_no_resource_arr[index_no_resource] = xgb_auc_no_resource

    index_no_resource += 1
dnn_auc_no_resource_arr[2] = "Average"  # To let users figure out that dnn_auc_no_resource[3] value is the average when seeing the csv file
dnn_auc_no_resource_arr[3] = (dnn_auc_no_resource_arr[0] + dnn_auc_no_resource_arr[1]) / 2  # Averaging the auc values from kfold cv
rf_auc_no_resource_arr[2] = "Average"
rf_auc_no_resource_arr[3] = (rf_auc_no_resource_arr[0] + rf_auc_no_resource_arr[1]) / 2
xgb_auc_no_resource_arr[2] = "Average"
xgb_auc_no_resource_arr[3] = (xgb_auc_no_resource_arr[0] + xgb_auc_no_resource_arr[1]) / 2
print("Average AUC of DNN classifier no resource is: " + str(dnn_auc_no_resource_arr[3]))
print("Average AUC of RF classifier no resource is: " + str(rf_auc_no_resource_arr[3]))
print("Average AUC of XGB classifier no resource is: " + str(xgb_auc_no_resource_arr[3]))

no_dnn_save_name = "no_resource_dnn_result_" + dataset_name + "_" + str(max_prefix) + ".csv"
no_rf_save_name = "no_resource_rf_result_" + dataset_name + "_" + str(max_prefix) + ".csv"
no_xgb_save_name = "no_resource_xgb_result_" + dataset_name + "_" + str(max_prefix) + ".csv"
pd.DataFrame(dnn_auc_no_resource_arr).to_csv(no_dnn_save_name, index=False)
pd.DataFrame(rf_auc_no_resource_arr).to_csv(no_rf_save_name, index=False)
pd.DataFrame(xgb_auc_no_resource_arr).to_csv(no_xgb_save_name, index=False)


# Feature importances
pd.DataFrame(X_train_no_resource.columns.tolist()).to_csv(
    'column_names_no_resource_' + dataset_name + '_' + str(max_prefix) + '.csv')

##RF feature importances
no_resource_rf_feature_imp_name = "rf_feature_importances_no_resource_" + dataset_name + "_" + str(max_prefix) + ".csv"
np.savetxt(no_resource_rf_feature_imp_name,
           rf_classifier_no_resource.fit(X_train_no_resource, y_train_no_resource.values.ravel()).feature_importances_)
# np.savetxt("rf_feature_importances_index_no_resource.csv", np.argsort(rf_classifier_no_resource_clf.feature_importances_))

##XGB feature importances
no_resource_xgb_feature_imp_name = "xgb_feature_importances_no_resource_" + dataset_name + "_" + str(
    max_prefix) + ".csv"
np.savetxt(no_resource_xgb_feature_imp_name,
           xgb_classifier_no_resource.fit(X_train_no_resource, y_train_no_resource.values.ravel()).feature_importances_)
# np.savetxt("xgb_feature_importances_index_no_resource.csv", np.argsort(xgb_classifier_no_resource_clf.feature_importances_))


# [yes resource] Starting k-fold cv
index_yes_resource = 0  # This index value is used to store auc into list created right above
for train_index, test_index in kf_yes_resource.split(X_dummy_yes_resource):
    # Splitting data into training and test set
    X_train_yes_resource, X_test_yes_resource = X_dummy_yes_resource.iloc[train_index], X_dummy_yes_resource.iloc[
        test_index]
    y_train_yes_resource, y_test_yes_resource = y_dummy_yes_resource.iloc[train_index], y_dummy_yes_resource.iloc[
        test_index]
    train_yes_resource = pd.concat([X_train_yes_resource, y_train_yes_resource], axis=1)
    test_yes_resource = pd.concat([X_test_yes_resource, y_test_yes_resource], axis=1)

    print("X_train_yes_resource after prefix shape is :" + str(X_train_yes_resource.shape))
    print(len(X_train_yes_resource))
    print("X_test_yes_resource after prefix shape is :" + str(X_test_yes_resource.shape))
    print(len(X_test_yes_resource))

    first_digit_parameters = [x for x in itertools.product((20, 40, 60, 80, 100), repeat=1)]
    second_digit_parameters = [x for x in itertools.product((1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                                                             18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
                                                             33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
                                                             48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
                                                             63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77,
                                                             78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92,
                                                             93, 94, 95, 96, 97, 98, 99, 100), repeat=2)]
    third_digit_parameters = [x for x in itertools.product((1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                                                            18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
                                                            33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
                                                            48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
                                                            63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77,
                                                            78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92,
                                                            93, 94, 95, 96, 97, 98, 99, 100), repeat=3)]
    all_digit_parameters = first_digit_parameters  # one hidden layer
    # all_digit_parameters = first_digit_parameters + second_digit_parameters # two hidden layers
    # all_digit_parameters = first_digit_parameters + second_digit_parameters + third_digit_parameters # three hidden layers

    learning_rate_init_parameters = [0.1, 0.01, 0.001]

    parameters = {'hidden_layer_sizes': all_digit_parameters,
                  'learning_rate_init': learning_rate_init_parameters}

    print("DNN started")
    start = time.time()

    # Making dnn classifier
    dnn_classifier_yes_resource = MLPClassifier(max_iter=10000, activation='relu')
    dnn_classifier_yes_resource_clf = RandomizedSearchCV(dnn_classifier_yes_resource, parameters, n_jobs=num_cpu_cores,
                                                         cv=2)

    # Fitting the dnn classifier with training set
    dnn_classifier_yes_resource_clf.fit(X_train_yes_resource, y_train_yes_resource.values.ravel())
    print("DNN best parameter is:")
    print(dnn_classifier_yes_resource_clf.best_params_)

    print("DNN time: ", time.time() - start)

    n_tree = [100, 300, 500]
    max_depth = [20, 40, 60]
    min_samples_split = [5, 10, 20]

    parameters = {'n_estimators': n_tree,
                  'max_depth': max_depth,
                  'min_samples_split': min_samples_split}

    print("RF started")
    start = time.time()

    # Making rf classifier
    rf_classifier_yes_resource = RandomForestClassifier(n_jobs=num_cpu_cores)
    rf_classifier_yes_resource_clf = RandomizedSearchCV(rf_classifier_yes_resource, parameters, n_jobs=num_cpu_cores,
                                                        cv=2)

    # Fitting the rf classifier with training set
    rf_classifier_yes_resource_clf.fit(X_train_yes_resource, y_train_yes_resource.values.ravel())
    print("RF best parameter is:")
    print(rf_classifier_yes_resource_clf.best_params_)

    print("RF time: ", time.time() - start)

    parameters = {'n_estimators': n_tree,
                  'learning_rate': learning_rate_init_parameters,
                  'max_depth': max_depth}

    print("XGB started")
    start = time.time()

    # Making xgb classifier
    xgb_classifier_yes_resource = XGBClassifier(n_jobs=num_cpu_cores)
    xgb_classifier_yes_resource_clf = RandomizedSearchCV(xgb_classifier_yes_resource, parameters, n_jobs=num_cpu_cores,
                                                         cv=2)

    # Fitting the xgb classifier with training set
    xgb_classifier_yes_resource_clf.fit(X_train_yes_resource, y_train_yes_resource.values.ravel())
    print("XGB best parameter is:")
    print(xgb_classifier_yes_resource_clf.best_params_)

    print("XGB time: ", time.time() - start)

    # Prediction using dnn classifier
    dnn_pred_yes_resource = dnn_classifier_yes_resource_clf.predict(X_test_yes_resource)

    # Prediction using rf classifier
    rf_pred_yes_resource = rf_classifier_yes_resource_clf.predict(X_test_yes_resource)

    # Prediction using xgb classifier
    xgb_pred_yes_resource = xgb_classifier_yes_resource_clf.predict(X_test_yes_resource)

    # Calculating auc
    dnn_auc_yes_resource = roc_auc_score(y_test_yes_resource, dnn_pred_yes_resource)
    rf_auc_yes_resource = roc_auc_score(y_test_yes_resource, rf_pred_yes_resource)
    xgb_auc_yes_resource = roc_auc_score(y_test_yes_resource, xgb_pred_yes_resource)
    print("AUC of DNN classifier of" + " fold " + str(index_yes_resource + 1) + " yes resource is: " + str(
        dnn_auc_yes_resource))
    print("AUC of RF classifier of" + " fold " + str(index_yes_resource + 1) + " yes resource is: " + str(
        rf_auc_yes_resource))
    print("AUC of XGB classifier of" + " fold " + str(index_yes_resource + 1) + " yes resource is: " + str(
        xgb_auc_yes_resource))

    # Storing prediction results
    dnn_auc_yes_resource_arr[index_yes_resource] = dnn_auc_yes_resource
    rf_auc_yes_resource_arr[index_yes_resource] = rf_auc_yes_resource
    xgb_auc_yes_resource_arr[index_yes_resource] = xgb_auc_yes_resource

    index_yes_resource += 1
dnn_auc_yes_resource_arr[
    2] = "Average"  # To let users figure out that dnn_auc_yes_resource[3] value is the average when seeing the csv file
dnn_auc_yes_resource_arr[3] = (dnn_auc_yes_resource_arr[0] + dnn_auc_yes_resource_arr[1]) / 2  # Averaging the auc values from kfold cv
rf_auc_yes_resource_arr[2] = "Average"
rf_auc_yes_resource_arr[3] = (rf_auc_yes_resource_arr[0] + rf_auc_yes_resource_arr[1]) / 2
xgb_auc_yes_resource_arr[2] = "Average"
xgb_auc_yes_resource_arr[3] = (xgb_auc_yes_resource_arr[0] + xgb_auc_yes_resource_arr[1]) / 2
print("Average AUC of DNN classifier yes resource is: " + str(dnn_auc_yes_resource_arr[3]))
print("Average AUC of RF classifier yes resource is: " + str(rf_auc_yes_resource_arr[3]))
print("Average AUC of xgb classifier yes resource is: " + str(xgb_auc_yes_resource_arr[3]))

yes_dnn_save_name = "yes_resource_dnn_result_" + dataset_name + "_" + str(max_prefix) + ".csv"
yes_rf_save_name = "yes_resource_rf_result_" + dataset_name + "_" + str(max_prefix) + ".csv"
yes_xgb_save_name = "yes_resource_xgb_result_" + dataset_name + "_" + str(max_prefix) + ".csv"
pd.DataFrame(dnn_auc_yes_resource_arr).to_csv(yes_dnn_save_name, index=False)
pd.DataFrame(rf_auc_yes_resource_arr).to_csv(yes_rf_save_name, index=False)
pd.DataFrame(xgb_auc_yes_resource_arr).to_csv(yes_xgb_save_name, index=False)

# Feature importances
pd.DataFrame(X_train_yes_resource.columns.tolist()).to_csv(
    'column_names_yes_resource_' + dataset_name + '_' + str(max_prefix) + '.csv')

##RF feature importances
yes_resource_rf_feature_imp_name = "rf_feature_importances_yes_resource_" + dataset_name + "_" + str(
    max_prefix) + ".csv"
np.savetxt(yes_resource_rf_feature_imp_name, rf_classifier_yes_resource.fit(X_train_yes_resource,
                                                                            y_train_yes_resource.values.ravel()).feature_importances_)
# np.savetxt("rf_feature_importances_index_yes_resource.csv", np.argsort(rf_classifier_yes_resource_clf.feature_importances_))

##XGB feature importances
yes_resource_xgb_feature_imp_name = "xgb_feature_importances_yes_resource_" + dataset_name + "_" + str(
    max_prefix) + ".csv"
np.savetxt(yes_resource_xgb_feature_imp_name, xgb_classifier_yes_resource.fit(X_train_yes_resource,
                                                                              y_train_yes_resource.values.ravel()).feature_importances_)
# np.savetxt("xgb_feature_importances_index_yes_resource.csv", np.argsort(xgb_classifier_yes_resource_clf.feature_importances_))
system_end = time.time()
print("Total time:")
print(system_end - system_start)