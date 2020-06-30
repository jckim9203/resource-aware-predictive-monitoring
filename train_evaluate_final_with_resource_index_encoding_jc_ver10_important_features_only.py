import scipy.sparse
import pandas as pd
import numpy as np
import sys
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
data_with_exp_dir = "/home/jongchan/Resource/data_with_exp/data_with_exp/"

dataset_address = data_with_exp_dir + dataset_name + ".csv"

# Reading data
# Be careful of "sep" parameter. "sep" parameter can be ";" or "," at times.
data = pd.read_csv(dataset_address, sep=",")

if dataset_name == "BPIC11_f1_exp_prefix":
    data = data.rename(columns = {"Case.ID": "case_id"}) #For BPIC11_f1
if dataset_name == "BPIC11_f2_exp_prefix":    
    data = data.rename(columns = {"Case.ID": "case_id"}) #For BPIC11_f2
if dataset_name == "BPIC11_f3_exp_prefix":    
    data = data.rename(columns = {"Case.ID": "case_id"}) #For BPIC11_f3
if dataset_name == "BPIC11_f4_exp_prefix":    
    data = data.rename(columns = {"Case.ID": "case_id"}) #For BPIC11_f4
if dataset_name == "BPIC15_1_f2_exp_prefix":    
    data = data.rename(columns = {"Case.ID": "case_id"}) #For BPIC15_1_f2
if dataset_name == "BPIC15_2_f2_exp_prefix":    
    data = data.rename(columns = {"Case.ID": "case_id"}) #For BPIC15_2_f2
if dataset_name == "BPIC15_3_f2_exp_prefix":    
    data = data.rename(columns = {"Case.ID": "case_id"}) #For BPIC15_3_f2
if dataset_name == "BPIC15_4_f2_exp_prefix":    
    data = data.rename(columns = {"Case.ID": "case_id"}) #For BPIC15_4_f2
if dataset_name == "BPIC15_5_f2_exp_prefix":    
    data = data.rename(columns = {"Case.ID": "case_id"}) #For BPIC15_5_f2
if dataset_name == "traffic_fines_1_exp_prefix":    
    data = data.rename(columns = {"Case.ID": "case_id"}) #For traffic_fines_1
if dataset_name == "bpic2012_O_CANCELLED-COMPLETE_exp_prefix":    
    data = data.rename(columns = {"Case.ID": "case_id"}) #For BPIC2012_O_CANCELLED-COMPLETE
if dataset_name == "bpic2012_O_ACCEPTED-COMPLETE_exp_prefix":    
    data = data.rename(columns = {"Case.ID": "case_id"}) #For BPIC2012_O_ACCEPTED-COMPLETE
if dataset_name == "bpic2012_O_DECLINED-COMPLETE_exp_prefix":    
    data = data.rename(columns = {"Case.ID": "case_id"}) #For BPIC2012_O_DECLINED-COMPLETE
if dataset_name == "BPIC17_O_Cancelled_exp_prefix":    
    data = data.rename(columns = {"Case.ID": "case_id"}) #For BPIC17_O_Cancelled
if dataset_name == "BPIC17_O_Accepted_exp_prefix":    
    data = data.rename(columns = {"Case.ID": "case_id"}) #For BPIC17_O_Accepted
if dataset_name == "BPIC17_O_Refused_exp_prefix":    
    data = data.rename(columns = {"Case.ID": "case_id"}) #For BPIC17_O_Refused


print(data.columns)
print("Data size is: " + str(len(data)))

# Deleting "index" column
if data.columns[0] == "index":
    data = data.drop([data.columns[0]], axis = 1)

# Getting the index of "label" column
label_index = data.columns.get_loc("label")


data.to_csv('data_initial_max_prefix.csv', index=False)

# Calculating the length of each case
data['case_len'] = 0
for i in pd.unique(data['case_id']):
#    print(max(data['event_nr'][data['case_id'] == i]))
    data['case_len'][data['case_id'] == i] = max(data['event_nr'][data['case_id'] == i])
data.to_csv('data_prefix_adjusted.csv', index=False)



# Since 'Responsible_actor' feature has more than 2 different data types, data type conversion is conducted
def bpic15_1_f2_responsible_actor_dtype_conversion(iter):
    for i in range(len(data['Responsible_actor'])):
        if data['Responsible_actor'][i] == "other":
            continue
        else:
            if data['Responsible_actor'][i] == "missing":
                continue
            else:
                data['Responsible_actor'][i] = float(data['Responsible_actor'][i])
    return data
if __name__ == '__main__':
    if dataset_name == "BPIC15_1_f2_exp_prefix":
        pool = multiprocessing.Pool(processes=num_cpu_cores)
        data = pool.map(bpic15_1_f2_responsible_actor_dtype_conversion, [0])[0]
        pool.close()
        pool.join()


def bpic15_2_f2_responsible_actor_dtype_conversion(iter):
    for i in range(len(data['Responsible_actor'])):
        if data['Responsible_actor'][i] == "other":
            continue
        else:
            if data['Responsible_actor'][i] == "missing":
                continue
            else:
                data['Responsible_actor'][i] = float(data['Responsible_actor'][i])
    return data
if __name__ == '__main__':
    if dataset_name == "BPIC15_2_f2_exp_prefix":
        pool = multiprocessing.Pool(processes=num_cpu_cores)
        data = pool.map(bpic15_2_f2_responsible_actor_dtype_conversion, [0])[0]
        pool.close()
        pool.join()


def bpic15_3_f2_responsible_actor_dtype_conversion(iter):
    for i in range(len(data['Responsible_actor'])):
        if data['Responsible_actor'][i] == "other":
            continue
        else:
            if data['Responsible_actor'][i] == "missing":
                continue
            else:
                data['Responsible_actor'][i] = float(data['Responsible_actor'][i])
    return data
if __name__ == '__main__':
    if dataset_name == "BPIC15_3_f2_exp_prefix":
        pool = multiprocessing.Pool(processes=num_cpu_cores)
        data = pool.map(bpic15_3_f2_responsible_actor_dtype_conversion, [0])[0]
        pool.close()
        pool.join()

def bpic15_4_f2_responsible_actor_dtype_conversion(iter):
    for i in range(len(data['Responsible_actor'])):
        if data['Responsible_actor'][i] == "other":
            continue
        else:
            if data['Responsible_actor'][i] == "missing":
                continue
            else:
                data['Responsible_actor'][i] = float(data['Responsible_actor'][i])
    return data
if __name__ == '__main__':
    if dataset_name == "BPIC15_4_f2_exp_prefix":
        pool = multiprocessing.Pool(processes=num_cpu_cores)
        data = pool.map(bpic15_4_f2_responsible_actor_dtype_conversion, [0])[0]
        pool.close()
        pool.join()

def bpic15_5_f2_responsible_actor_dtype_conversion(iter):
    for i in range(len(data['Responsible_actor'])):
        if data['Responsible_actor'][i] == "other":
            continue
        else:
            if data['Responsible_actor'][i] == "missing":
                continue
            else:
                data['Responsible_actor'][i] = float(data['Responsible_actor'][i])
    return data
if __name__ == '__main__':
    if dataset_name == "BPIC15_5_f2_exp_prefix":
        pool = multiprocessing.Pool(processes=num_cpu_cores)
        data = pool.map(bpic15_5_f2_responsible_actor_dtype_conversion, [0])[0]
        pool.close()
        pool.join()


# Since 'monitoringResource' feature has more than 2 different data types, data type conversion is conducted
def bpic15_1_f2_monitoringResource_dtype_conversion(iter):
    for i in range(len(data['monitoringResource'])):
        if data['monitoringResource'][i] == "other":
            continue
        else:
            data['monitoringResource'][i] = float(data['monitoringResource'][i])
    return data
if __name__ == '__main__':
    if dataset_name == "BPIC15_1_f2_exp_prefix":
        pool = multiprocessing.Pool(processes=num_cpu_cores)
        data = pool.map(bpic15_1_f2_monitoringResource_dtype_conversion, [0])[0]
        pool.close()
        pool.join()


# Since 'monitoringResource' feature has more than 2 different data types, data type conversion is conducted
def bpic15_2_f2_monitoringResource_dtype_conversion(iter):
    for i in range(len(data['monitoringResource'])):
        if data['monitoringResource'][i] == "other":
            continue
        else:
            data['monitoringResource'][i] = float(data['monitoringResource'][i])
    return data
if __name__ == '__main__':
    if dataset_name == "BPIC15_2_f2_exp_prefix":
        pool = multiprocessing.Pool(processes=num_cpu_cores)
        data = pool.map(bpic15_2_f2_monitoringResource_dtype_conversion, [0])[0]
        pool.close()
        pool.join()


# Since 'monitoringResource' feature has more than 2 different data types, data type conversion is conducted
def bpic15_3_f2_monitoringResource_dtype_conversion(iter):
    for i in range(len(data['monitoringResource'])):
        if data['monitoringResource'][i] == "other":
            continue
        else:
            data['monitoringResource'][i] = float(data['monitoringResource'][i])
    return data
if __name__ == '__main__':
    if dataset_name == "BPIC15_3_f2_exp_prefix":
        pool = multiprocessing.Pool(processes=num_cpu_cores)
        data = pool.map(bpic15_3_f2_monitoringResource_dtype_conversion, [0])[0]
        pool.close()
        pool.join()


# Since 'monitoringResource' feature has more than 2 different data types, data type conversion is conducted
def bpic15_4_f2_monitoringResource_dtype_conversion(iter):
    for i in range(len(data['monitoringResource'])):
        if data['monitoringResource'][i] == "other":
            continue
        else:
            data['monitoringResource'][i] = float(data['monitoringResource'][i])
    return data
if __name__ == '__main__':
    if dataset_name == "BPIC15_4_f2_exp_prefix":
        pool = multiprocessing.Pool(processes=num_cpu_cores)
        data = pool.map(bpic15_4_f2_monitoringResource_dtype_conversion, [0])[0]
        pool.close()
        pool.join()

# Since 'monitoringResource' feature has more than 2 different data types, data type conversion is conducted
def bpic15_5_f2_monitoringResource_dtype_conversion(iter):
    for i in range(len(data['monitoringResource'])):
        if data['monitoringResource'][i] == "other":
            continue
        else:
            data['monitoringResource'][i] = float(data['monitoringResource'][i])
    return data
if __name__ == '__main__':
    if dataset_name == "BPIC15_5_f2_exp_prefix":
        pool = multiprocessing.Pool(processes=num_cpu_cores)
        data = pool.map(bpic15_5_f2_monitoringResource_dtype_conversion, [0])[0]
        pool.close()
        pool.join()

# Since 'Resource' feature has more than 2 different data types, data type conversion is conducted
def traffic_fines_1_Resource_dtype_conversion(iter):
    for i in range(len(data['Resource'])):
        if data['Resource'][i] == "other":
            continue
        else:
            if data['Resource'][i] == "missing":
                continue
            else:
                data['Resource'][i] = float(data['Resource'][i])
    return data
if __name__ == '__main__':
    if dataset_name == "traffic_fines_1_exp_prefix":
        pool = multiprocessing.Pool(processes=num_cpu_cores)
        data = pool.map(traffic_fines_1_Resource_dtype_conversion, [0])[0]
        pool.close()
        pool.join()

# Since 'Resource' feature has more than 2 different data types, data type conversion is conducted
def bpic2012_O_CANCELLED_COMPLETE_Resource_dtype_conversion(iter):
    for i in range(len(data['Resource'])):
        if data['Resource'][i] == "other":
            continue
        else:
            if data['Resource'][i] == "missing":
                continue
            else:
                data['Resource'][i] = float(data['Resource'][i])
    return data
if __name__ == '__main__':
    if dataset_name == "bpic2012_O_CANCELLED-COMPLETE_exp_prefix":
        pool = multiprocessing.Pool(processes=num_cpu_cores)
        data = pool.map(bpic2012_O_CANCELLED_COMPLETE_Resource_dtype_conversion, [0])[0]
        pool.close()
        pool.join()


# Since 'Resource' feature has more than 2 different data types, data type conversion is conducted
def bpic2012_O_ACCEPTED_COMPLETE_Resource_dtype_conversion(iter):
    for i in range(len(data['Resource'])):
        if data['Resource'][i] == "other":
            continue
        else:
            if data['Resource'][i] == "missing":
                continue
            else:
                data['Resource'][i] = float(data['Resource'][i])
    return data
if __name__ == '__main__':
    if dataset_name == "bpic2012_O_ACCEPTED-COMPLETE_exp_prefix":
        pool = multiprocessing.Pool(processes=num_cpu_cores)
        data = pool.map(bpic2012_O_ACCEPTED_COMPLETE_Resource_dtype_conversion, [0])[0]
        pool.close()
        pool.join()


# Since 'Resource' feature has more than 2 different data types, data type conversion is conducted
def bpic2012_O_DECLINED_COMPLETE_Resource_dtype_conversion(iter):
    for i in range(len(data['Resource'])):
        if data['Resource'][i] == "other":
            continue
        else:
            if data['Resource'][i] == "missing":
                continue
            else:
                data['Resource'][i] = float(data['Resource'][i])
    return data
if __name__ == '__main__':
    if dataset_name == "bpic2012_O_DECLINED-COMPLETE_exp_prefix":
        pool = multiprocessing.Pool(processes=num_cpu_cores)
        data = pool.map(bpic2012_O_DECLINED_COMPLETE_Resource_dtype_conversion, [0])[0]
        pool.close()
        pool.join()


#print(list(data.columns.values))

# Getting the indices of variables in data that will not be used
if dataset_name == "BPIC11_f1_exp_prefix":
    timestamp_index = data.columns.get_loc("time.timestamp") #Time
    data = data.drop(data.columns[[timestamp_index]], axis = 1)

if dataset_name == "BPIC11_f2_exp_prefix":
    timestamp_index = data.columns.get_loc("time.timestamp") #Time
    data = data.drop(data.columns[[timestamp_index]], axis = 1)

if dataset_name == "BPIC11_f3_exp_prefix":
    timestamp_index = data.columns.get_loc("time.timestamp") #Time
    data = data.drop(data.columns[[timestamp_index]], axis = 1)

if dataset_name == "BPIC11_f4_exp_prefix":
    timestamp_index = data.columns.get_loc("time.timestamp") #Time
    data = data.drop(data.columns[[timestamp_index]], axis = 1)

if dataset_name == "BPIC15_1_f2_exp_prefix":
    timestamp_index = data.columns.get_loc("time.timestamp") #Time
    data = data.drop(data.columns[[timestamp_index]], axis = 1)

if dataset_name == "BPIC15_2_f2_exp_prefix":
    timestamp_index = data.columns.get_loc("time.timestamp") #Time
    data = data.drop(data.columns[[timestamp_index]], axis = 1)

if dataset_name == "BPIC15_3_f2_exp_prefix":
    timestamp_index = data.columns.get_loc("time.timestamp") #Time
    data = data.drop(data.columns[[timestamp_index]], axis = 1)

if dataset_name == "BPIC15_4_f2_exp_prefix":
    timestamp_index = data.columns.get_loc("time.timestamp") #Time
    data = data.drop(data.columns[[timestamp_index]], axis = 1)

if dataset_name == "BPIC15_5_f2_exp_prefix":
    timestamp_index = data.columns.get_loc("time.timestamp") #Time
    data = data.drop(data.columns[[timestamp_index]], axis = 1)

# Getting the indices of variables in data that will not be used
if dataset_name == "traffic_fines_1_exp_prefix":
    timestamp_index = data.columns.get_loc("Complete.Timestamp") #Time
    data = data.drop(data.columns[[timestamp_index]], axis = 1)

# Getting the indices of variables in data that will not be used
if dataset_name == "bpic2012_O_CANCELLED-COMPLETE_exp_prefix":
    timestamp_index = data.columns.get_loc("Complete.Timestamp") #Time
    data = data.drop(data.columns[[timestamp_index]], axis = 1)

# Getting the indices of variables in data that will not be used
if dataset_name == "bpic2012_O_ACCEPTED-COMPLETE_exp_prefix":
    timestamp_index = data.columns.get_loc("Complete.Timestamp") #Time
    data = data.drop(data.columns[[timestamp_index]], axis = 1)

# Getting the indices of variables in data that will not be used
if dataset_name == "bpic2012_O_DECLINED-COMPLETE_exp_prefix":
    timestamp_index = data.columns.get_loc("Complete.Timestamp") #Time
    data = data.drop(data.columns[[timestamp_index]], axis = 1)

# Getting the indices of variables in data that will not be used
if dataset_name == "BPIC17_O_Cancelled_exp_prefix":
    timestamp_index = data.columns.get_loc("time.timestamp") #Time
    data = data.drop(data.columns[[timestamp_index]], axis = 1)

# Getting the indices of variables in data that will not be used
if dataset_name == "BPIC17_O_Accepted_exp_prefix":
    timestamp_index = data.columns.get_loc("time.timestamp") #Time
    data = data.drop(data.columns[[timestamp_index]], axis = 1)

# Getting the indices of variables in data that will not be used
if dataset_name == "BPIC17_O_Refused_exp_prefix":
    timestamp_index = data.columns.get_loc("time.timestamp") #Time
    data = data.drop(data.columns[[timestamp_index]], axis = 1)

# Dummification of data
if dataset_name == "BPIC11_f1_exp_prefix":
    data_dummy = pd.get_dummies(data, prefix="Activity", columns=['Activity.code'])
    data_dummy = pd.get_dummies(data_dummy, prefix="Producer", columns=['Producer.code'])
    data_dummy = pd.get_dummies(data_dummy, prefix="Treatment_code", columns=['Treatment.code'])
    data_dummy = pd.get_dummies(data_dummy, prefix="Diagnosis_code", columns=['Diagnosis.code'])
    data_dummy = pd.get_dummies(data_dummy, prefix="Specialism_code", columns=['Specialism.code.1'])
    data_dummy = pd.get_dummies(data_dummy, prefix="Specialism_code1", columns=['Specialism.code.2'])
    data_dummy = pd.get_dummies(data_dummy, prefix="prev_resource", columns=['prev_resource'])
    data_dummy = pd.get_dummies(data_dummy, prefix="is_last_event", columns=['is_last_event'])
    data_dummy = pd.get_dummies(data_dummy, prefix="Section", columns=['Section'])
    data_dummy = pd.get_dummies(data_dummy, prefix="group", columns=['group'])
    data_dummy = data_dummy.fillna(0)

if dataset_name == "BPIC11_f2_exp_prefix":
    data_dummy = pd.get_dummies(data, prefix="Activity", columns=['Activity.code'])
    data_dummy = pd.get_dummies(data_dummy, prefix="Producer", columns=['Producer.code'])
    data_dummy = pd.get_dummies(data_dummy, prefix="Treatment_code", columns=['Treatment.code'])
    data_dummy = pd.get_dummies(data_dummy, prefix="Diagnosis_code", columns=['Diagnosis.code'])
    data_dummy = pd.get_dummies(data_dummy, prefix="Specialism_code", columns=['Specialism.code.1'])
    data_dummy = pd.get_dummies(data_dummy, prefix="Specialism_code1", columns=['Specialism.code.2'])
    data_dummy = pd.get_dummies(data_dummy, prefix="prev_resource", columns=['prev_resource'])
    data_dummy = pd.get_dummies(data_dummy, prefix="is_last_event", columns=['is_last_event'])
    data_dummy = pd.get_dummies(data_dummy, prefix="Section", columns=['Section'])
    data_dummy = pd.get_dummies(data_dummy, prefix="group", columns=['group'])
    data_dummy = data_dummy.fillna(0)

if dataset_name == "BPIC11_f3_exp_prefix":
    data_dummy = pd.get_dummies(data, prefix="Activity", columns=['Activity.code'])
    data_dummy = pd.get_dummies(data_dummy, prefix="Producer", columns=['Producer.code'])
    data_dummy = pd.get_dummies(data_dummy, prefix="Treatment_code", columns=['Treatment.code'])
    data_dummy = pd.get_dummies(data_dummy, prefix="Diagnosis_code", columns=['Diagnosis.code'])
    data_dummy = pd.get_dummies(data_dummy, prefix="Specialism_code", columns=['Specialism.code.1'])
    data_dummy = pd.get_dummies(data_dummy, prefix="Specialism_code1", columns=['Specialism.code.2'])
    data_dummy = pd.get_dummies(data_dummy, prefix="prev_resource", columns=['prev_resource'])
    data_dummy = pd.get_dummies(data_dummy, prefix="is_last_event", columns=['is_last_event'])
    data_dummy = pd.get_dummies(data_dummy, prefix="Section", columns=['Section'])
    data_dummy = pd.get_dummies(data_dummy, prefix="group", columns=['group'])
    data_dummy = data_dummy.fillna(0)

if dataset_name == "BPIC11_f4_exp_prefix":
    data_dummy = pd.get_dummies(data, prefix="Activity", columns=['Activity.code'])
    data_dummy = pd.get_dummies(data_dummy, prefix="Producer", columns=['Producer.code'])
    data_dummy = pd.get_dummies(data_dummy, prefix="Treatment_code", columns=['Treatment.code'])
    data_dummy = pd.get_dummies(data_dummy, prefix="Diagnosis_code", columns=['Diagnosis.code'])
    data_dummy = pd.get_dummies(data_dummy, prefix="Specialism_code", columns=['Specialism.code.1'])
    data_dummy = pd.get_dummies(data_dummy, prefix="Specialism_code1", columns=['Specialism.code.2'])
    data_dummy = pd.get_dummies(data_dummy, prefix="prev_resource", columns=['prev_resource'])
    data_dummy = pd.get_dummies(data_dummy, prefix="is_last_event", columns=['is_last_event'])
    data_dummy = pd.get_dummies(data_dummy, prefix="Section", columns=['Section'])
    data_dummy = pd.get_dummies(data_dummy, prefix="group", columns=['group'])
    data_dummy = data_dummy.fillna(0)

if dataset_name == "BPIC15_1_f2_exp_prefix":
    data_dummy = pd.get_dummies(data, prefix="Activity", columns=['Activity'])
    data_dummy = pd.get_dummies(data_dummy, prefix="monitoringResource", columns=['monitoringResource'])
    data_dummy = pd.get_dummies(data_dummy, prefix="question", columns=['question'])
    data_dummy = pd.get_dummies(data_dummy, prefix="org_resource", columns=['org.resource'])
    data_dummy = pd.get_dummies(data_dummy, prefix="Responsible_actor", columns=['Responsible_actor'])
    data_dummy = pd.get_dummies(data_dummy, prefix="prev_resource", columns=['prev_resource'])
    data_dummy = pd.get_dummies(data_dummy, prefix="is_last_event", columns=['is_last_event'])
    data_dummy = data_dummy.fillna(0)

if dataset_name == "BPIC15_2_f2_exp_prefix":
    data_dummy = pd.get_dummies(data, prefix="Activity", columns=['Activity'])
    data_dummy = pd.get_dummies(data_dummy, prefix="monitoringResource", columns=['monitoringResource'])
    data_dummy = pd.get_dummies(data_dummy, prefix="question", columns=['question'])
    data_dummy = pd.get_dummies(data_dummy, prefix="org_resource", columns=['org.resource'])
    data_dummy = pd.get_dummies(data_dummy, prefix="Responsible_actor", columns=['Responsible_actor'])
    data_dummy = pd.get_dummies(data_dummy, prefix="prev_resource", columns=['prev_resource'])
    data_dummy = pd.get_dummies(data_dummy, prefix="is_last_event", columns=['is_last_event'])
    data_dummy = data_dummy.fillna(0)

if dataset_name == "BPIC15_3_f2_exp_prefix":
    data_dummy = pd.get_dummies(data, prefix="Activity", columns=['Activity'])
    data_dummy = pd.get_dummies(data_dummy, prefix="monitoringResource", columns=['monitoringResource'])
    data_dummy = pd.get_dummies(data_dummy, prefix="question", columns=['question'])
    data_dummy = pd.get_dummies(data_dummy, prefix="org_resource", columns=['org.resource'])
    data_dummy = pd.get_dummies(data_dummy, prefix="Responsible_actor", columns=['Responsible_actor'])
    data_dummy = pd.get_dummies(data_dummy, prefix="prev_resource", columns=['prev_resource'])
    data_dummy = pd.get_dummies(data_dummy, prefix="is_last_event", columns=['is_last_event'])
    data_dummy = data_dummy.fillna(0)

if dataset_name == "BPIC15_4_f2_exp_prefix":
    data_dummy = pd.get_dummies(data, prefix="Activity", columns=['Activity'])
    data_dummy = pd.get_dummies(data_dummy, prefix="monitoringResource", columns=['monitoringResource'])
    data_dummy = pd.get_dummies(data_dummy, prefix="question", columns=['question'])
    data_dummy = pd.get_dummies(data_dummy, prefix="org_resource", columns=['org.resource'])
    data_dummy = pd.get_dummies(data_dummy, prefix="Responsible_actor", columns=['Responsible_actor'])
    data_dummy = pd.get_dummies(data_dummy, prefix="prev_resource", columns=['prev_resource'])
    data_dummy = pd.get_dummies(data_dummy, prefix="is_last_event", columns=['is_last_event'])
    data_dummy = data_dummy.fillna(0)

if dataset_name == "BPIC15_5_f2_exp_prefix":
    data_dummy = pd.get_dummies(data, prefix="Activity", columns=['Activity'])
    data_dummy = pd.get_dummies(data_dummy, prefix="monitoringResource", columns=['monitoringResource'])
    data_dummy = pd.get_dummies(data_dummy, prefix="question", columns=['question'])
    data_dummy = pd.get_dummies(data_dummy, prefix="org_resource", columns=['org.resource'])
    data_dummy = pd.get_dummies(data_dummy, prefix="Responsible_actor", columns=['Responsible_actor'])
    data_dummy = pd.get_dummies(data_dummy, prefix="prev_resource", columns=['prev_resource'])
    data_dummy = pd.get_dummies(data_dummy, prefix="is_last_event", columns=['is_last_event'])
    data_dummy = data_dummy.fillna(0)

if dataset_name == "traffic_fines_1_exp_prefix":
    data_dummy = pd.get_dummies(data, prefix="Activity", columns=['Activity'])
    data_dummy = pd.get_dummies(data_dummy, prefix="article", columns=['article'])
    data_dummy = pd.get_dummies(data_dummy, prefix="lastSent", columns=['lastSent'])
    data_dummy = pd.get_dummies(data_dummy, prefix="notificationType", columns=['notificationType'])
    data_dummy = pd.get_dummies(data_dummy, prefix="dismissal", columns=['dismissal'])
    data_dummy = pd.get_dummies(data_dummy, prefix="vehicleClass", columns=['vehicleClass'])
    data_dummy = pd.get_dummies(data_dummy, prefix="prev_resource", columns=['prev_resource'])
    data_dummy = pd.get_dummies(data_dummy, prefix="Resource", columns=['Resource'])
    data_dummy = pd.get_dummies(data_dummy, prefix="is_last_event", columns=['is_last_event'])
    data_dummy = data_dummy.fillna(0)

if dataset_name == "bpic2012_O_CANCELLED-COMPLETE_exp_prefix":
    data_dummy = pd.get_dummies(data, prefix="Activity", columns=['Activity'])
    data_dummy = pd.get_dummies(data_dummy, prefix="lifecycle.transition", columns=['lifecycle.transition'])
    data_dummy = pd.get_dummies(data_dummy, prefix="prev_resource", columns=['prev_resource'])
    data_dummy = pd.get_dummies(data_dummy, prefix="Resource", columns=['Resource'])
    data_dummy = pd.get_dummies(data_dummy, prefix="is_last_event", columns=['is_last_event'])
    data_dummy = data_dummy.fillna(0)

if dataset_name == "bpic2012_O_ACCEPTED-COMPLETE_exp_prefix":
    data_dummy = pd.get_dummies(data, prefix="Activity", columns=['Activity'])
    data_dummy = pd.get_dummies(data_dummy, prefix="lifecycle.transition", columns=['lifecycle.transition'])
    data_dummy = pd.get_dummies(data_dummy, prefix="prev_resource", columns=['prev_resource'])
    data_dummy = pd.get_dummies(data_dummy, prefix="Resource", columns=['Resource'])
    data_dummy = pd.get_dummies(data_dummy, prefix="is_last_event", columns=['is_last_event'])
    data_dummy = data_dummy.fillna(0)

if dataset_name == "bpic2012_O_DECLINED-COMPLETE_exp_prefix":
    data_dummy = pd.get_dummies(data, prefix="Activity", columns=['Activity'])
    data_dummy = pd.get_dummies(data_dummy, prefix="lifecycle.transition", columns=['lifecycle.transition'])
    data_dummy = pd.get_dummies(data_dummy, prefix="prev_resource", columns=['prev_resource'])
    data_dummy = pd.get_dummies(data_dummy, prefix="Resource", columns=['Resource'])
    data_dummy = pd.get_dummies(data_dummy, prefix="is_last_event", columns=['is_last_event'])
    data_dummy = data_dummy.fillna(0)

if dataset_name == "BPIC17_O_Cancelled_exp_prefix":
    data_dummy = pd.get_dummies(data, prefix="Activity", columns=['Activity'])
    data_dummy = pd.get_dummies(data_dummy, prefix="ApplicationType", columns=['ApplicationType'])
    data_dummy = pd.get_dummies(data_dummy, prefix="LoanGoal", columns=['LoanGoal'])
    data_dummy = pd.get_dummies(data_dummy, prefix="org_resource", columns=['org.resource'])
    data_dummy = pd.get_dummies(data_dummy, prefix="Action", columns=['Action'])
    data_dummy = pd.get_dummies(data_dummy, prefix="EventOrigin", columns=['EventOrigin'])
    data_dummy = pd.get_dummies(data_dummy, prefix="lifecycle.transition", columns=['lifecycle.transition'])
    data_dummy = pd.get_dummies(data_dummy, prefix="Accepted", columns=['Accepted'])
    data_dummy = pd.get_dummies(data_dummy, prefix="Selected", columns=['Selected'])
    data_dummy = pd.get_dummies(data_dummy, prefix="prev_resource", columns=['prev_resource'])
    data_dummy = pd.get_dummies(data_dummy, prefix="is_last_event", columns=['is_last_event'])
    data_dummy = data_dummy.fillna(0)

if dataset_name == "BPIC17_O_Accepted_exp_prefix":
    data_dummy = pd.get_dummies(data, prefix="Activity", columns=['Activity'])
    data_dummy = pd.get_dummies(data_dummy, prefix="ApplicationType", columns=['ApplicationType'])
    data_dummy = pd.get_dummies(data_dummy, prefix="LoanGoal", columns=['LoanGoal'])
    data_dummy = pd.get_dummies(data_dummy, prefix="org_resource", columns=['org.resource'])
    data_dummy = pd.get_dummies(data_dummy, prefix="Action", columns=['Action'])
    data_dummy = pd.get_dummies(data_dummy, prefix="EventOrigin", columns=['EventOrigin'])
    data_dummy = pd.get_dummies(data_dummy, prefix="lifecycle.transition", columns=['lifecycle.transition'])
    data_dummy = pd.get_dummies(data_dummy, prefix="Accepted", columns=['Accepted'])
    data_dummy = pd.get_dummies(data_dummy, prefix="Selected", columns=['Selected'])
    data_dummy = pd.get_dummies(data_dummy, prefix="prev_resource", columns=['prev_resource'])
    data_dummy = pd.get_dummies(data_dummy, prefix="is_last_event", columns=['is_last_event'])
    data_dummy = data_dummy.fillna(0)

if dataset_name == "BPIC17_O_Refused_exp_prefix":
    data_dummy = pd.get_dummies(data, prefix="Activity", columns=['Activity'])
    data_dummy = pd.get_dummies(data_dummy, prefix="ApplicationType", columns=['ApplicationType'])
    data_dummy = pd.get_dummies(data_dummy, prefix="LoanGoal", columns=['LoanGoal'])
    data_dummy = pd.get_dummies(data_dummy, prefix="org_resource", columns=['org.resource'])
    data_dummy = pd.get_dummies(data_dummy, prefix="Action", columns=['Action'])
    data_dummy = pd.get_dummies(data_dummy, prefix="EventOrigin", columns=['EventOrigin'])
    data_dummy = pd.get_dummies(data_dummy, prefix="lifecycle.transition", columns=['lifecycle.transition'])
    data_dummy = pd.get_dummies(data_dummy, prefix="Accepted", columns=['Accepted'])
    data_dummy = pd.get_dummies(data_dummy, prefix="Selected", columns=['Selected'])
    data_dummy = pd.get_dummies(data_dummy, prefix="prev_resource", columns=['prev_resource'])
    data_dummy = pd.get_dummies(data_dummy, prefix="is_last_event", columns=['is_last_event'])
    data_dummy = data_dummy.fillna(0)


# Index encoding
# Adding the suffix indicating the number of prefix to each feature
print("Index encoding started")
data_dummy_prefix = data_dummy.copy()
print(data_dummy_prefix.columns)

#Below code is written in order not to make duplicate features for case-level attributes
def bpic11_f1_index_encoding(iter):
    for col in data_dummy_prefix.columns:
        if col not in ("case_id","Diagnosis.code","label","case_len", "Treatment.code","Specialism.code.1","Specialism.code.2","Age"):
            for prefix in range(max_prefix):
                data_dummy_prefix[str(col) + '_prefixlen' + str(prefix+1)] = 0
        else:
            data_dummy_prefix[str(col) + '_prefixlen' + str(1)] = 0 #For case-level attributes, only "_prefixlen1" is attached as suffix.
    return data_dummy_prefix
if __name__ == '__main__':
    if dataset_name == "BPIC11_f1_exp_prefix":
        if max_prefix <= 20:
            pool = multiprocessing.Pool(processes=num_cpu_cores)
            data_dummy_prefix = pool.map(bpic11_f1_index_encoding, [0])[0]
            pool.close()
            pool.join()
        else:
            for col in data_dummy_prefix.columns:
                if col not in ("case_id","Diagnosis.code","label","case_len","Treatment.code", "Specialism.code.1","Specialism.code.2","Age"):
                    for prefix in range(max_prefix):
                        data_dummy_prefix[str(col) + '_prefixlen' + str(prefix+1)] = 0
                else:
                    data_dummy_prefix[str(col) + '_prefixlen' + str(1)] = 0 #For case-level attributes, only "_prefixlen1" is attached as suffix.



def bpic11_f2_index_encoding(iter):
    for col in data_dummy_prefix.columns:
        if col not in ("case_id","Diagnosis.code","label","case_len", "Treatment.code","Specialism.code.1","Specialism.code.2","Age"):
            for prefix in range(max_prefix):
                data_dummy_prefix[str(col) + '_prefixlen' + str(prefix+1)] = 0
        else:
            data_dummy_prefix[str(col) + '_prefixlen' + str(1)] = 0 #For case-level attributes, only "_prefixlen1" is attached as suffix.
    return data_dummy_prefix
if __name__ == '__main__':
    if dataset_name == "BPIC11_f2_exp_prefix":
        if max_prefix <= 20:
            pool = multiprocessing.Pool(processes=num_cpu_cores)
            data_dummy_prefix = pool.map(bpic11_f2_index_encoding, [0])[0]
            pool.close()
            pool.join()
        else:
            for col in data_dummy_prefix.columns:
                if col not in ("case_id","Diagnosis.code","label","case_len","Treatment.code", "Specialism.code.1","Specialism.code.2","Age"):
                    for prefix in range(max_prefix):
                        data_dummy_prefix[str(col) + '_prefixlen' + str(prefix+1)] = 0
                else:
                    data_dummy_prefix[str(col) + '_prefixlen' + str(1)] = 0 #For case-level attributes, only "_prefixlen1" is attached as suffix.



def bpic11_f3_index_encoding(iter):
    for col in data_dummy_prefix.columns:
        if col not in ("case_id","Diagnosis.code","label","case_len", "Treatment.code","Specialism.code.1","Specialism.code.2","Age"):
            for prefix in range(max_prefix):
                data_dummy_prefix[str(col) + '_prefixlen' + str(prefix+1)] = 0
        else:
            data_dummy_prefix[str(col) + '_prefixlen' + str(1)] = 0 #For case-level attributes, only "_prefixlen1" is attached as suffix.
    return data_dummy_prefix
if __name__ == '__main__':
    if dataset_name == "BPIC11_f3_exp_prefix":
        if max_prefix <= 20:
            pool = multiprocessing.Pool(processes=num_cpu_cores)
            data_dummy_prefix = pool.map(bpic11_f3_index_encoding, [0])[0]
            pool.close()
            pool.join()
        else:
            for col in data_dummy_prefix.columns:
                if col not in ("case_id","Diagnosis.code","label","case_len","Treatment.code", "Specialism.code.1","Specialism.code.2","Age"):
                    for prefix in range(max_prefix):
                        data_dummy_prefix[str(col) + '_prefixlen' + str(prefix+1)] = 0
                else:
                    data_dummy_prefix[str(col) + '_prefixlen' + str(1)] = 0 #For case-level attributes, only "_prefixlen1" is attached as suffix.



def bpic11_f4_index_encoding(iter):
    for col in data_dummy_prefix.columns:
        if col not in ("case_id","Diagnosis.code","label","case_len", "Treatment.code","Specialism.code.1","Specialism.code.2","Age"):
            for prefix in range(max_prefix):
                data_dummy_prefix[str(col) + '_prefixlen' + str(prefix+1)] = 0
        else:
            data_dummy_prefix[str(col) + '_prefixlen' + str(1)] = 0 #For case-level attributes, only "_prefixlen1" is attached as suffix.
    return data_dummy_prefix
if __name__ == '__main__':
    if dataset_name == "BPIC11_f4_exp_prefix":
        if max_prefix <= 20:
            pool = multiprocessing.Pool(processes=num_cpu_cores)
            data_dummy_prefix = pool.map(bpic11_f4_index_encoding, [0])[0]
            pool.close()
            pool.join()
        else:
            for col in data_dummy_prefix.columns:
                if col not in ("case_id","Diagnosis.code","label","case_len","Treatment.code", "Specialism.code.1","Specialism.code.2","Age"):
                    for prefix in range(max_prefix):
                        data_dummy_prefix[str(col) + '_prefixlen' + str(prefix+1)] = 0
                else:
                    data_dummy_prefix[str(col) + '_prefixlen' + str(1)] = 0 #For case-level attributes, only "_prefixlen1" is attached as suffix.



def bpic15_1_f2_index_encoding(iter):
    for col in data_dummy_prefix.columns:
        if col not in ("case_id","case_len","label","Aanleg..Uitvoeren.werk.of.werkzaamheid.","SUMleges","Bouw","Brandveilig.gebruik..melding.","Brandveilig.gebruik..vergunning.","Gebiedsbescherming","Handelen.in.strijd.met.regels.RO","Inrit.Uitweg","Kap","Milieu..melding.","Milieu..neutraal.wijziging.","Milieu..omgevingsvergunning.beperkte.milieutoets.","Milieu..vergunning.","Monument","Reclame","Sloop"):
            for prefix in range(max_prefix):
                data_dummy_prefix[str(col) + '_prefixlen' + str(prefix+1)] = 0
        else:
            data_dummy_prefix[str(col) + '_prefixlen' + str(1)] = 0 #For case-level attributes, only "_prefixlen1" is attached as suffix.
    return data_dummy_prefix
if __name__ == '__main__':
    if dataset_name == "BPIC15_1_f2_exp_prefix":
        if max_prefix <= 20:
            pool = multiprocessing.Pool(processes=num_cpu_cores)
            data_dummy_prefix = pool.map(bpic15_1_f2_index_encoding, [0])[0]
            pool.close()
            pool.join()
        else:
            for col in data_dummy_prefix.columns:
                if col not in ("case_id","case_len","label","Aanleg..Uitvoeren.werk.of.werkzaamheid.","SUMleges","Bouw","Brandveilig.gebruik..melding.","Brandveilig.gebruik..vergunning.","Gebiedsbescherming","Handelen.in.strijd.met.regels.RO","Inrit.Uitweg","Kap","Milieu..melding.","Milieu..neutraal.wijziging.","Milieu..omgevingsvergunning.beperkte.milieutoets.","Milieu..vergunning.","Monument","Reclame","Sloop"):
                    for prefix in range(max_prefix):
                        data_dummy_prefix[str(col) + '_prefixlen' + str(prefix+1)] = 0
                else:
                    data_dummy_prefix[str(col) + '_prefixlen' + str(1)] = 0 #For case-level attributes, only "_prefixlen1" is attached as suffix.



def bpic15_2_f2_index_encoding(iter):
    for col in data_dummy_prefix.columns:
        if col not in ("case_id","case_len","label","Aanleg..Uitvoeren.werk.of.werkzaamheid.","SUMleges","Bouw","Brandveilig.gebruik..melding.","Brandveilig.gebruik..vergunning.","Gebiedsbescherming","Handelen.in.strijd.met.regels.RO","Inrit.Uitweg","Kap","Milieu..melding.","Milieu..neutraal.wijziging.","Milieu..omgevingsvergunning.beperkte.milieutoets.","Milieu..vergunning.","Monument","Reclame","Sloop"):
            for prefix in range(max_prefix):
                data_dummy_prefix[str(col) + '_prefixlen' + str(prefix+1)] = 0
        else:
            data_dummy_prefix[str(col) + '_prefixlen' + str(1)] = 0 #For case-level attributes, only "_prefixlen1" is attached as suffix.
    return data_dummy_prefix
if __name__ == '__main__':
    if dataset_name == "BPIC15_2_f2_exp_prefix":
        if max_prefix <= 20:
            pool = multiprocessing.Pool(processes=num_cpu_cores)
            data_dummy_prefix = pool.map(bpic15_2_f2_index_encoding, [0])[0]
            pool.close()
            pool.join()
        else:
            for col in data_dummy_prefix.columns:
                if col not in ("case_id","case_len","label","Aanleg..Uitvoeren.werk.of.werkzaamheid.","SUMleges","Bouw","Brandveilig.gebruik..melding.","Brandveilig.gebruik..vergunning.","Gebiedsbescherming","Handelen.in.strijd.met.regels.RO","Inrit.Uitweg","Kap","Milieu..melding.","Milieu..neutraal.wijziging.","Milieu..omgevingsvergunning.beperkte.milieutoets.","Milieu..vergunning.","Monument","Reclame","Sloop"):
                    for prefix in range(max_prefix):
                        data_dummy_prefix[str(col) + '_prefixlen' + str(prefix+1)] = 0
                else:
                    data_dummy_prefix[str(col) + '_prefixlen' + str(1)] = 0 #For case-level attributes, only "_prefixlen1" is attached as suffix.


def bpic15_3_f2_index_encoding(iter):
    for col in data_dummy_prefix.columns:
        if col not in ("case_id","case_len","label","Aanleg..Uitvoeren.werk.of.werkzaamheid.","SUMleges","Bouw","Brandveilig.gebruik..melding.","Brandveilig.gebruik..vergunning.","Gebiedsbescherming","Handelen.in.strijd.met.regels.RO","Inrit.Uitweg","Kap","Milieu..melding.","Milieu..neutraal.wijziging.","Milieu..omgevingsvergunning.beperkte.milieutoets.","Milieu..vergunning.","Monument","Reclame","Sloop"):
            for prefix in range(max_prefix):
                data_dummy_prefix[str(col) + '_prefixlen' + str(prefix+1)] = 0
        else:
            data_dummy_prefix[str(col) + '_prefixlen' + str(1)] = 0 #For case-level attributes, only "_prefixlen1" is attached as suffix.
    return data_dummy_prefix
if __name__ == '__main__':
    if dataset_name == "BPIC15_3_f2_exp_prefix":
        if max_prefix <= 20:
            pool = multiprocessing.Pool(processes=num_cpu_cores)
            data_dummy_prefix = pool.map(bpic15_3_f2_index_encoding, [0])[0]
            pool.close()
            pool.join()
        else:
            for col in data_dummy_prefix.columns:
                if col not in ("case_id","case_len","label","Aanleg..Uitvoeren.werk.of.werkzaamheid.","SUMleges","Bouw","Brandveilig.gebruik..melding.","Brandveilig.gebruik..vergunning.","Gebiedsbescherming","Handelen.in.strijd.met.regels.RO","Inrit.Uitweg","Kap","Milieu..melding.","Milieu..neutraal.wijziging.","Milieu..omgevingsvergunning.beperkte.milieutoets.","Milieu..vergunning.","Monument","Reclame","Sloop"):
                    for prefix in range(max_prefix):
                        data_dummy_prefix[str(col) + '_prefixlen' + str(prefix+1)] = 0
                else:
                    data_dummy_prefix[str(col) + '_prefixlen' + str(1)] = 0 #For case-level attributes, only "_prefixlen1" is attached as suffix.


def bpic15_4_f2_index_encoding(iter):
    for col in data_dummy_prefix.columns:
        if col not in ("case_id","case_len","label","Aanleg..Uitvoeren.werk.of.werkzaamheid.","SUMleges","Bouw","Brandveilig.gebruik..melding.","Brandveilig.gebruik..vergunning.","Gebiedsbescherming","Handelen.in.strijd.met.regels.RO","Inrit.Uitweg","Kap","Milieu..melding.","Milieu..neutraal.wijziging.","Milieu..omgevingsvergunning.beperkte.milieutoets.","Milieu..vergunning.","Monument","Reclame","Sloop"):
            for prefix in range(max_prefix):
                data_dummy_prefix[str(col) + '_prefixlen' + str(prefix+1)] = 0
        else:
            data_dummy_prefix[str(col) + '_prefixlen' + str(1)] = 0 #For case-level attributes, only "_prefixlen1" is attached as suffix.
    return data_dummy_prefix
if __name__ == '__main__':
    if dataset_name == "BPIC15_4_f2_exp_prefix":
        if max_prefix <= 20:
            pool = multiprocessing.Pool(processes=num_cpu_cores)
            data_dummy_prefix = pool.map(bpic15_4_f2_index_encoding, [0])[0]
            pool.close()
            pool.join()
        else:
            for col in data_dummy_prefix.columns:
                if col not in ("case_id","case_len","label","Aanleg..Uitvoeren.werk.of.werkzaamheid.","SUMleges","Bouw","Brandveilig.gebruik..melding.","Brandveilig.gebruik..vergunning.","Gebiedsbescherming","Handelen.in.strijd.met.regels.RO","Inrit.Uitweg","Kap","Milieu..melding.","Milieu..neutraal.wijziging.","Milieu..omgevingsvergunning.beperkte.milieutoets.","Milieu..vergunning.","Monument","Reclame","Sloop"):
                    for prefix in range(max_prefix):
                        data_dummy_prefix[str(col) + '_prefixlen' + str(prefix+1)] = 0
                else:
                    data_dummy_prefix[str(col) + '_prefixlen' + str(1)] = 0 #For case-level attributes, only "_prefixlen1" is attached as suffix.


def bpic15_5_f2_index_encoding(iter):
    for col in data_dummy_prefix.columns:
        if col not in ("case_id","case_len","label","Aanleg..Uitvoeren.werk.of.werkzaamheid.","SUMleges","Bouw","Brandveilig.gebruik..melding.","Brandveilig.gebruik..vergunning.","Gebiedsbescherming","Handelen.in.strijd.met.regels.RO","Inrit.Uitweg","Kap","Milieu..melding.","Milieu..neutraal.wijziging.","Milieu..omgevingsvergunning.beperkte.milieutoets.","Milieu..vergunning.","Monument","Reclame","Sloop"):
            for prefix in range(max_prefix):
                data_dummy_prefix[str(col) + '_prefixlen' + str(prefix+1)] = 0
        else:
            data_dummy_prefix[str(col) + '_prefixlen' + str(1)] = 0 #For case-level attributes, only "_prefixlen1" is attached as suffix.
    return data_dummy_prefix
if __name__ == '__main__':
    if dataset_name == "BPIC15_5_f2_exp_prefix":
        if max_prefix <= 20:
            pool = multiprocessing.Pool(processes=num_cpu_cores)
            data_dummy_prefix = pool.map(bpic15_5_f2_index_encoding, [0])[0]
            pool.close()
            pool.join()
        else:
            for col in data_dummy_prefix.columns:
                if col not in ("case_id","case_len","label","Aanleg..Uitvoeren.werk.of.werkzaamheid.","SUMleges","Bouw","Brandveilig.gebruik..melding.","Brandveilig.gebruik..vergunning.","Gebiedsbescherming","Handelen.in.strijd.met.regels.RO","Inrit.Uitweg","Kap","Milieu..melding.","Milieu..neutraal.wijziging.","Milieu..omgevingsvergunning.beperkte.milieutoets.","Milieu..vergunning.","Monument","Reclame","Sloop"):
                    for prefix in range(max_prefix):
                        data_dummy_prefix[str(col) + '_prefixlen' + str(prefix+1)] = 0
                else:
                    data_dummy_prefix[str(col) + '_prefixlen' + str(1)] = 0 #For case-level attributes, only "_prefixlen1" is attached as suffix.

def traffic_fines_1_index_encoding(iter):
    for col in data_dummy_prefix.columns:
        if col not in ("case_id","article","vehicleClass", "points","case_len","label"):
            for prefix in range(max_prefix):
                data_dummy_prefix[str(col) + '_prefixlen' + str(prefix+1)] = 0
        else:
            data_dummy_prefix[str(col) + '_prefixlen' + str(1)] = 0 #For case-level attributes, only "_prefixlen1" is attached as suffix.
    return data_dummy_prefix
if __name__ == '__main__':
    if dataset_name == "traffic_fines_1_exp_prefix":
        if max_prefix <= 20:
            pool = multiprocessing.Pool(processes=num_cpu_cores)
            data_dummy_prefix = pool.map(traffic_fines_1_index_encoding, [0])[0]
            pool.close()
            pool.join()
        else:
            for col in data_dummy_prefix.columns:
                if col not in ("case_id","article","vehicleClass", "points","case_len","label"):
                    for prefix in range(max_prefix):
                        data_dummy_prefix[str(col) + '_prefixlen' + str(prefix+1)] = 0
                else:
                    data_dummy_prefix[str(col) + '_prefixlen' + str(1)] = 0 #For case-level attributes, only "_prefixlen1" is attached as suffix.

def bpic2012_O_CANCELLED_COMPLETE_index_encoding(iter):
    for col in data_dummy_prefix.columns:
        if col not in ("case_id","AMOUNT_REQ", "case_len","label"):
            for prefix in range(max_prefix):
                data_dummy_prefix[str(col) + '_prefixlen' + str(prefix+1)] = 0
        else:
            data_dummy_prefix[str(col) + '_prefixlen' + str(1)] = 0 #For case-level attributes, only "_prefixlen1" is attached as suffix.
    return data_dummy_prefix
if __name__ == '__main__':
    if dataset_name == "bpic2012_O_CANCELLED-COMPLETE_exp_prefix":
        if max_prefix <= 20:
            pool = multiprocessing.Pool(processes=num_cpu_cores)
            data_dummy_prefix = pool.map(bpic2012_O_CANCELLED_COMPLETE_index_encoding, [0])[0]
            pool.close()
            pool.join()
        else:
            for col in data_dummy_prefix.columns:
                if col not in ("case_id","AMOUNT_REQ", "case_len","label"):
                    for prefix in range(max_prefix):
                        data_dummy_prefix[str(col) + '_prefixlen' + str(prefix+1)] = 0
                else:
                    data_dummy_prefix[str(col) + '_prefixlen' + str(1)] = 0 #For case-level attributes, only "_prefixlen1" is attached as suffix.

def bpic2012_O_ACCEPTED_COMPLETE_index_encoding(iter):
    for col in data_dummy_prefix.columns:
        if col not in ("case_id","AMOUNT_REQ", "case_len","label"):
            for prefix in range(max_prefix):
                data_dummy_prefix[str(col) + '_prefixlen' + str(prefix+1)] = 0
        else:
            data_dummy_prefix[str(col) + '_prefixlen' + str(1)] = 0 #For case-level attributes, only "_prefixlen1" is attached as suffix.
    return data_dummy_prefix
if __name__ == '__main__':
    if dataset_name == "bpic2012_O_ACCEPTED-COMPLETE_exp_prefix":
        if max_prefix <= 20:
            pool = multiprocessing.Pool(processes=num_cpu_cores)
            data_dummy_prefix = pool.map(bpic2012_O_ACCEPTED_COMPLETE_index_encoding, [0])[0]
            pool.close()
            pool.join()
        else:
            for col in data_dummy_prefix.columns:
                if col not in ("case_id","AMOUNT_REQ", "case_len","label"):
                    for prefix in range(max_prefix):
                        data_dummy_prefix[str(col) + '_prefixlen' + str(prefix+1)] = 0
                else:
                    data_dummy_prefix[str(col) + '_prefixlen' + str(1)] = 0 #For case-level attributes, only "_prefixlen1" is attached as suffix.

def bpic2012_O_DECLINED_COMPLETE_index_encoding(iter):
    for col in data_dummy_prefix.columns:
        if col not in ("case_id","AMOUNT_REQ", "case_len","label"):
            for prefix in range(max_prefix):
                data_dummy_prefix[str(col) + '_prefixlen' + str(prefix+1)] = 0
        else:
            data_dummy_prefix[str(col) + '_prefixlen' + str(1)] = 0 #For case-level attributes, only "_prefixlen1" is attached as suffix.
    return data_dummy_prefix
if __name__ == '__main__':
    if dataset_name == "bpic2012_O_DECLINED-COMPLETE_exp_prefix":
        if max_prefix <= 20:
            pool = multiprocessing.Pool(processes=num_cpu_cores)
            data_dummy_prefix = pool.map(bpic2012_O_DECLINED_COMPLETE_index_encoding, [0])[0]
            pool.close()
            pool.join()
        else:
            for col in data_dummy_prefix.columns:
                if col not in ("case_id","AMOUNT_REQ", "case_len","label"):
                    for prefix in range(max_prefix):
                        data_dummy_prefix[str(col) + '_prefixlen' + str(prefix+1)] = 0
                else:
                    data_dummy_prefix[str(col) + '_prefixlen' + str(1)] = 0 #For case-level attributes, only "_prefixlen1" is attached as suffix.

def BPIC17_O_Cancelled_index_encoding(iter):
    for col in data_dummy_prefix.columns:
        if col not in ("case_id","Action", "EventOrigin", "lifecycle.transition", "Accepted", "Selected", "AMOUNT_REQ", "case_len","label"):
            for prefix in range(max_prefix):
                data_dummy_prefix[str(col) + '_prefixlen' + str(prefix+1)] = 0
        else:
            data_dummy_prefix[str(col) + '_prefixlen' + str(1)] = 0 #For case-level attributes, only "_prefixlen1" is attached as suffix.
    return data_dummy_prefix
if __name__ == '__main__':
    if dataset_name == "BPIC17_O_Cancelled_exp_prefix":
        if max_prefix <= 20:
            pool = multiprocessing.Pool(processes=num_cpu_cores)
            data_dummy_prefix = pool.map(BPIC17_O_Cancelled_index_encoding, [0])[0]
            pool.close()
            pool.join()
        else:
            for col in data_dummy_prefix.columns:
                if col not in ("case_id","Action", "EventOrigin", "lifecycle.transition", "Accepted", "Selected", "AMOUNT_REQ", "case_len","label"):
                    for prefix in range(max_prefix):
                        data_dummy_prefix[str(col) + '_prefixlen' + str(prefix+1)] = 0
                else:
                    data_dummy_prefix[str(col) + '_prefixlen' + str(1)] = 0 #For case-level attributes, only "_prefixlen1" is attached as suffix.


def BPIC17_O_Accepted_index_encoding(iter):
    for col in data_dummy_prefix.columns:
        if col not in ("case_id","Action", "EventOrigin", "lifecycle.transition", "Accepted", "Selected", "AMOUNT_REQ", "case_len","label"):
            for prefix in range(max_prefix):
                data_dummy_prefix[str(col) + '_prefixlen' + str(prefix+1)] = 0
        else:
            data_dummy_prefix[str(col) + '_prefixlen' + str(1)] = 0 #For case-level attributes, only "_prefixlen1" is attached as suffix.
    return data_dummy_prefix
if __name__ == '__main__':
    if dataset_name == "BPIC17_O_Accepted_exp_prefix":
        if max_prefix <= 20:
            pool = multiprocessing.Pool(processes=num_cpu_cores)
            data_dummy_prefix = pool.map(BPIC17_O_Accepted_index_encoding, [0])[0]
            pool.close()
            pool.join()
        else:
            for col in data_dummy_prefix.columns:
                if col not in ("case_id","Action", "EventOrigin", "lifecycle.transition", "Accepted", "Selected", "AMOUNT_REQ", "case_len","label"):
                    for prefix in range(max_prefix):
                        data_dummy_prefix[str(col) + '_prefixlen' + str(prefix+1)] = 0
                else:
                    data_dummy_prefix[str(col) + '_prefixlen' + str(1)] = 0 #For case-level attributes, only "_prefixlen1" is attached as suffix.


def BPIC17_O_Refused_index_encoding(iter):
    for col in data_dummy_prefix.columns:
        if col not in ("case_id","Action", "EventOrigin", "lifecycle.transition", "Accepted", "Selected", "AMOUNT_REQ", "case_len","label"):
            for prefix in range(max_prefix):
                data_dummy_prefix[str(col) + '_prefixlen' + str(prefix+1)] = 0
        else:
            data_dummy_prefix[str(col) + '_prefixlen' + str(1)] = 0 #For case-level attributes, only "_prefixlen1" is attached as suffix.
    return data_dummy_prefix
if __name__ == '__main__':
    if dataset_name == "BPIC17_O_Refused_exp_prefix":
        if max_prefix <= 20:
            pool = multiprocessing.Pool(processes=num_cpu_cores)
            data_dummy_prefix = pool.map(BPIC17_O_Refused_index_encoding, [0])[0]
            pool.close()
            pool.join()
        else:
            for col in data_dummy_prefix.columns:
                if col not in ("case_id","Action", "EventOrigin", "lifecycle.transition", "Accepted", "Selected", "AMOUNT_REQ", "case_len","label"):
                    for prefix in range(max_prefix):
                        data_dummy_prefix[str(col) + '_prefixlen' + str(prefix+1)] = 0
                else:
                    data_dummy_prefix[str(col) + '_prefixlen' + str(1)] = 0 #For case-level attributes, only "_prefixlen1" is attached as suffix.

print(data_dummy_prefix.columns)
new_col_name = pd.DataFrame(data_dummy_prefix.columns)
new_col_name.to_csv('new_col_name_list.csv')
data_dummy_prefix.to_csv('data_dummy_prefix.csv', index = False)
print("Index encoding finished")
    
# (OPTION 1. This works well) Filling in the newly-made index features with the values of original features
if max_prefix == 2:
    prefix1 = data_dummy_prefix[data_dummy_prefix['prefix']==1]
    prefix1 = prefix1.reset_index()
    prefix1 = prefix1.drop("index",1)
    print(len(prefix1))
    print("Processing columns with prefix 1...")
    for col in prefix1:
        if col[len(col)-10:len(col)] == "prefixlen1":
            prefix1[col] = prefix1[col[0:len(col)-11]]

    prefix1_ref = prefix1[prefix1['case_len']>=2]
    prefix1_ref = prefix1_ref.reset_index()
    prefix1_ref = prefix1_ref.drop("index",1)
    prefix2 = data_dummy_prefix[data_dummy_prefix['prefix']==2]
    prefix2 = prefix2.reset_index()
    prefix2 = prefix2.drop("index",1)
    print(len(prefix2))
    print("Processing columns with prefix 2...")
    for col in prefix2:
        if col[len(col)-10:len(col)] == "prefixlen1":
            prefix2[col] = prefix1_ref[col[0:len(col)-11]]
        if col[len(col)-10:len(col)] == "prefixlen2":
            prefix2[col] = prefix2[col[0:len(col)-11]]
    prefix1 = prefix1.reset_index()
    prefix1 = prefix1.drop("index",1)
    prefix2 = prefix2.reset_index()
    prefix2 = prefix2.drop("index",1)
    prefix_index_encoding = prefix1.append(prefix2)
    prefix_index_encoding = prefix_index_encoding.reset_index()
    prefix_index_encoding = prefix_index_encoding.drop("index",1)
    prefix_index_encoding = prefix_index_encoding.filter(regex="prefixlen")
    prefix_index_encoding.to_csv("index_encoded_data_" + dataset_name + "_" + str(max_prefix) + ".csv", index = False)

if max_prefix == 3:
    prefix1 = data_dummy_prefix[data_dummy_prefix['prefix']==1]
    prefix1 = prefix1.reset_index()
    prefix1 = prefix1.drop("index",1)
    print(len(prefix1))
    print("Processing columns with prefix 1...")
    for col in prefix1:
        if col[len(col)-10:len(col)] == "prefixlen1":
            prefix1[col] = prefix1[col[0:len(col)-11]]

    prefix1_ref = prefix1[prefix1['case_len']>=2]
    prefix1_ref = prefix1_ref.reset_index()
    prefix1_ref = prefix1_ref.drop("index",1)
    prefix2 = data_dummy_prefix[data_dummy_prefix['prefix']==2]
    prefix2 = prefix2.reset_index()
    prefix2 = prefix2.drop("index",1)
    print(len(prefix2))
    print("Processing columns with prefix 2...")
    for col in prefix2:
        if col[len(col)-10:len(col)] == "prefixlen1":
            prefix2[col] = prefix1_ref[col[0:len(col)-11]]
        if col[len(col)-10:len(col)] == "prefixlen2":
            prefix2[col] = prefix2[col[0:len(col)-11]]

    prefix1_ref = prefix1[prefix1['case_len']>=3]
    prefix1_ref = prefix1_ref.reset_index()
    prefix1_ref = prefix1_ref.drop("index",1)
    prefix2_ref = prefix2[prefix2['case_len']>=3]
    prefix2_ref = prefix2_ref.reset_index()
    prefix2_ref = prefix2_ref.drop("index",1)
    prefix3 = data_dummy_prefix[data_dummy_prefix['prefix']==3]
    prefix3 = prefix3.reset_index()
    prefix3 = prefix3.drop("index",1)
    print(len(prefix3))
    print("Processing columns with prefix 3...")
    for col in prefix3:
        if col[len(col)-10:len(col)] == "prefixlen1":
            prefix3[col] = prefix1_ref[col[0:len(col)-11]]
        if col[len(col)-10:len(col)] == "prefixlen2":
            prefix3[col] = prefix2_ref[col[0:len(col)-11]]
        if col[len(col)-10:len(col)] == "prefixlen3":
            prefix3[col] = prefix3[col[0:len(col)-11]]

    prefix1 = prefix1.reset_index()
    prefix1 = prefix1.drop("index",1)
    prefix2 = prefix2.reset_index()
    prefix2 = prefix2.drop("index",1)
    prefix3 = prefix3.reset_index()
    prefix3 = prefix3.drop("index",1)
    prefix_index_encoding = prefix1.append(prefix2)
    prefix_index_encoding = prefix_index_encoding.append(prefix3)
    prefix_index_encoding = prefix_index_encoding.reset_index()
    prefix_index_encoding = prefix_index_encoding.drop("index",1)
    prefix_index_encoding = prefix_index_encoding.filter(regex="prefixlen")
    prefix_index_encoding.to_csv("index_encoded_data_" + dataset_name + "_" + str(max_prefix) + ".csv", index = False)

if max_prefix == 10:
    prefix1 = data_dummy_prefix[data_dummy_prefix['prefix']==1]
    prefix1 = prefix1.reset_index()
    prefix1 = prefix1.drop("index",1)
    print(len(prefix1))
    print("Processing columns with prefix 1...")
    for col in prefix1:
        if col[len(col)-10:len(col)] == "prefixlen1":
            prefix1[col] = prefix1[col[0:len(col)-11]]

    prefix1_ref = prefix1[prefix1['case_len']>=2]
    prefix1_ref = prefix1_ref.reset_index()
    prefix1_ref = prefix1_ref.drop("index",1)
    prefix2 = data_dummy_prefix[data_dummy_prefix['prefix']==2]
    prefix2 = prefix2.reset_index()
    prefix2 = prefix2.drop("index",1)
    print(len(prefix2))
    print("Processing columns with prefix 2...")
    for col in prefix2:
        if col[len(col)-10:len(col)] == "prefixlen1":
            prefix2[col] = prefix1_ref[col[0:len(col)-11]]
        if col[len(col)-10:len(col)] == "prefixlen2":
            prefix2[col] = prefix2[col[0:len(col)-11]]

    prefix1_ref = prefix1[prefix1['case_len']>=3]
    prefix1_ref = prefix1_ref.reset_index()
    prefix1_ref = prefix1_ref.drop("index",1)
    prefix2_ref = prefix2[prefix2['case_len']>=3]
    prefix2_ref = prefix2_ref.reset_index()
    prefix2_ref = prefix2_ref.drop("index",1)
    prefix3 = data_dummy_prefix[data_dummy_prefix['prefix']==3]
    prefix3 = prefix3.reset_index()
    prefix3 = prefix3.drop("index",1)
    print(len(prefix3))
    print("Processing columns with prefix 3...")
    for col in prefix3:
        if col[len(col)-10:len(col)] == "prefixlen1":
            prefix3[col] = prefix1_ref[col[0:len(col)-11]]
        if col[len(col)-10:len(col)] == "prefixlen2":
            prefix3[col] = prefix2_ref[col[0:len(col)-11]]
        if col[len(col)-10:len(col)] == "prefixlen3":
            prefix3[col] = prefix3[col[0:len(col)-11]]

    prefix1_ref = prefix1[prefix1['case_len']>=4]
    prefix1_ref = prefix1_ref.reset_index()
    prefix1_ref = prefix1_ref.drop("index",1)
    prefix2_ref = prefix2[prefix2['case_len']>=4]
    prefix2_ref = prefix2_ref.reset_index()
    prefix2_ref = prefix2_ref.drop("index",1)
    prefix3_ref = prefix3[prefix3['case_len']>=4]
    prefix3_ref = prefix3_ref.reset_index()
    prefix3_ref = prefix3_ref.drop("index",1)
    prefix4 = data_dummy_prefix[data_dummy_prefix['prefix']==4]
    prefix4 = prefix4.reset_index()
    prefix4 = prefix4.drop("index",1)
    print(len(prefix4))
    print("Processing columns with prefix 4...")
    for col in prefix4:
        if col[len(col)-10:len(col)] == "prefixlen1":
            prefix4[col] = prefix1_ref[col[0:len(col)-11]]
        if col[len(col)-10:len(col)] == "prefixlen2":
            prefix4[col] = prefix2_ref[col[0:len(col)-11]]
        if col[len(col)-10:len(col)] == "prefixlen3":
            prefix4[col] = prefix3_ref[col[0:len(col)-11]]
        if col[len(col)-10:len(col)] == "prefixlen4":
            prefix4[col] = prefix4[col[0:len(col)-11]]

    prefix1_ref = prefix1[prefix1['case_len']>=5]
    prefix1_ref = prefix1_ref.reset_index()
    prefix1_ref = prefix1_ref.drop("index",1)
    prefix2_ref = prefix2[prefix2['case_len']>=5]
    prefix2_ref = prefix2_ref.reset_index()
    prefix2_ref = prefix2_ref.drop("index",1)
    prefix3_ref = prefix3[prefix3['case_len']>=5]
    prefix3_ref = prefix3_ref.reset_index()
    prefix3_ref = prefix3_ref.drop("index",1)
    prefix4_ref = prefix4[prefix4['case_len']>=5]
    prefix4_ref = prefix4_ref.reset_index()
    prefix4_ref = prefix4_ref.drop("index",1)
    prefix5 = data_dummy_prefix[data_dummy_prefix['prefix']==5]
    prefix5 = prefix5.reset_index()
    prefix5 = prefix5.drop("index",1)
    print(len(prefix5))
    print("Processing columns with prefix 5...")
    for col in prefix5:
        if col[len(col)-10:len(col)] == "prefixlen1":
            prefix5[col] = prefix1_ref[col[0:len(col)-11]]
        if col[len(col)-10:len(col)] == "prefixlen2":
            prefix5[col] = prefix2_ref[col[0:len(col)-11]]
        if col[len(col)-10:len(col)] == "prefixlen3":
            prefix5[col] = prefix3_ref[col[0:len(col)-11]]
        if col[len(col)-10:len(col)] == "prefixlen4":
            prefix5[col] = prefix4_ref[col[0:len(col)-11]]
        if col[len(col)-10:len(col)] == "prefixlen5":
            prefix5[col] = prefix5[col[0:len(col)-11]]

    prefix1_ref = prefix1[prefix1['case_len']>=6]
    prefix1_ref = prefix1_ref.reset_index()
    prefix1_ref = prefix1_ref.drop("index",1)
    prefix2_ref = prefix2[prefix2['case_len']>=6]
    prefix2_ref = prefix2_ref.reset_index()
    prefix2_ref = prefix2_ref.drop("index",1)
    prefix3_ref = prefix3[prefix3['case_len']>=6]
    prefix3_ref = prefix3_ref.reset_index()
    prefix3_ref = prefix3_ref.drop("index",1)
    prefix4_ref = prefix4[prefix4['case_len']>=6]
    prefix4_ref = prefix4_ref.reset_index()
    prefix4_ref = prefix4_ref.drop("index",1)
    prefix5_ref = prefix5[prefix5['case_len']>=6]
    prefix5_ref = prefix5_ref.reset_index()
    prefix5_ref = prefix5_ref.drop("index",1)
    prefix6 = data_dummy_prefix[data_dummy_prefix['prefix']==6]
    prefix6 = prefix6.reset_index()
    prefix6 = prefix6.drop("index",1)
    print(len(prefix6))
    print("Processing columns with prefix 6...")
    for col in prefix6:
        if col[len(col)-10:len(col)] == "prefixlen1":
            prefix6[col] = prefix1_ref[col[0:len(col)-11]]
        if col[len(col)-10:len(col)] == "prefixlen2":
            prefix6[col] = prefix2_ref[col[0:len(col)-11]]
        if col[len(col)-10:len(col)] == "prefixlen3":
            prefix6[col] = prefix3_ref[col[0:len(col)-11]]
        if col[len(col)-10:len(col)] == "prefixlen4":
            prefix6[col] = prefix4_ref[col[0:len(col)-11]]
        if col[len(col)-10:len(col)] == "prefixlen5":
            prefix6[col] = prefix5_ref[col[0:len(col)-11]]
        if col[len(col)-10:len(col)] == "prefixlen6":
            prefix6[col] = prefix6[col[0:len(col)-11]]

    prefix1_ref = prefix1[prefix1['case_len']>=7]
    prefix1_ref = prefix1_ref.reset_index()
    prefix1_ref = prefix1_ref.drop("index",1)
    prefix2_ref = prefix2[prefix2['case_len']>=7]
    prefix2_ref = prefix2_ref.reset_index()
    prefix2_ref = prefix2_ref.drop("index",1)
    prefix3_ref = prefix3[prefix3['case_len']>=7]
    prefix3_ref = prefix3_ref.reset_index()
    prefix3_ref = prefix3_ref.drop("index",1)
    prefix4_ref = prefix4[prefix4['case_len']>=7]
    prefix4_ref = prefix4_ref.reset_index()
    prefix4_ref = prefix4_ref.drop("index",1)
    prefix5_ref = prefix5[prefix5['case_len']>=7]
    prefix5_ref = prefix5_ref.reset_index()
    prefix5_ref = prefix5_ref.drop("index",1)
    prefix6_ref = prefix6[prefix6['case_len']>=7]
    prefix6_ref = prefix6_ref.reset_index()
    prefix6_ref = prefix6_ref.drop("index",1)
    prefix7 = data_dummy_prefix[data_dummy_prefix['prefix']==7]
    prefix7 = prefix7.reset_index()
    prefix7 = prefix7.drop("index",1)
    print(len(prefix7))
    print("Processing columns with prefix 7...")
    for col in prefix7:
        if col[len(col)-10:len(col)] == "prefixlen1":
            prefix7[col] = prefix1_ref[col[0:len(col)-11]]
        if col[len(col)-10:len(col)] == "prefixlen2":
            prefix7[col] = prefix2_ref[col[0:len(col)-11]]
        if col[len(col)-10:len(col)] == "prefixlen3":
            prefix7[col] = prefix3_ref[col[0:len(col)-11]]
        if col[len(col)-10:len(col)] == "prefixlen4":
            prefix7[col] = prefix4_ref[col[0:len(col)-11]]
        if col[len(col)-10:len(col)] == "prefixlen5":
            prefix7[col] = prefix5_ref[col[0:len(col)-11]]
        if col[len(col)-10:len(col)] == "prefixlen6":
            prefix7[col] = prefix6_ref[col[0:len(col)-11]]
        if col[len(col)-10:len(col)] == "prefixlen7":
            prefix7[col] = prefix7[col[0:len(col)-11]]

    prefix1_ref = prefix1[prefix1['case_len']>=8]
    prefix1_ref = prefix1_ref.reset_index()
    prefix1_ref = prefix1_ref.drop("index",1)
    prefix2_ref = prefix2[prefix2['case_len']>=8]
    prefix2_ref = prefix2_ref.reset_index()
    prefix2_ref = prefix2_ref.drop("index",1)
    prefix3_ref = prefix3[prefix3['case_len']>=8]
    prefix3_ref = prefix3_ref.reset_index()
    prefix3_ref = prefix3_ref.drop("index",1)
    prefix4_ref = prefix4[prefix4['case_len']>=8]
    prefix4_ref = prefix4_ref.reset_index()
    prefix4_ref = prefix4_ref.drop("index",1)
    prefix5_ref = prefix5[prefix5['case_len']>=8]
    prefix5_ref = prefix5_ref.reset_index()
    prefix5_ref = prefix5_ref.drop("index",1)
    prefix6_ref = prefix6[prefix6['case_len']>=8]
    prefix6_ref = prefix6_ref.reset_index()
    prefix6_ref = prefix6_ref.drop("index",1)
    prefix7_ref = prefix7[prefix7['case_len']>=8]
    prefix7_ref = prefix7_ref.reset_index()
    prefix7_ref = prefix7_ref.drop("index",1)
    prefix8 = data_dummy_prefix[data_dummy_prefix['prefix']==8]
    prefix8 = prefix8.reset_index()
    prefix8 = prefix8.drop("index",1)
    print(len(prefix8))
    print("Processing columns with prefix 8...")
    for col in prefix8:
        if col[len(col)-10:len(col)] == "prefixlen1":
            prefix8[col] = prefix1_ref[col[0:len(col)-11]]
        if col[len(col)-10:len(col)] == "prefixlen2":
            prefix8[col] = prefix2_ref[col[0:len(col)-11]]
        if col[len(col)-10:len(col)] == "prefixlen3":
            prefix8[col] = prefix3_ref[col[0:len(col)-11]]
        if col[len(col)-10:len(col)] == "prefixlen4":
            prefix8[col] = prefix4_ref[col[0:len(col)-11]]
        if col[len(col)-10:len(col)] == "prefixlen5":
            prefix8[col] = prefix5_ref[col[0:len(col)-11]]
        if col[len(col)-10:len(col)] == "prefixlen6":
            prefix8[col] = prefix6_ref[col[0:len(col)-11]]
        if col[len(col)-10:len(col)] == "prefixlen7":
            prefix8[col] = prefix7_ref[col[0:len(col)-11]]
        if col[len(col)-10:len(col)] == "prefixlen8":
            prefix8[col] = prefix8[col[0:len(col)-11]]

    prefix1_ref = prefix1[prefix1['case_len']>=9]
    prefix1_ref = prefix1_ref.reset_index()
    prefix1_ref = prefix1_ref.drop("index",1)
    prefix2_ref = prefix2[prefix2['case_len']>=9]
    prefix2_ref = prefix2_ref.reset_index()
    prefix2_ref = prefix2_ref.drop("index",1)
    prefix3_ref = prefix3[prefix3['case_len']>=9]
    prefix3_ref = prefix3_ref.reset_index()
    prefix3_ref = prefix3_ref.drop("index",1)
    prefix4_ref = prefix4[prefix4['case_len']>=9]
    prefix4_ref = prefix4_ref.reset_index()
    prefix4_ref = prefix4_ref.drop("index",1)
    prefix5_ref = prefix5[prefix5['case_len']>=9]
    prefix5_ref = prefix5_ref.reset_index()
    prefix5_ref = prefix5_ref.drop("index",1)
    prefix6_ref = prefix6[prefix6['case_len']>=9]
    prefix6_ref = prefix6_ref.reset_index()
    prefix6_ref = prefix6_ref.drop("index",1)
    prefix7_ref = prefix7[prefix7['case_len']>=9]
    prefix7_ref = prefix7_ref.reset_index()
    prefix7_ref = prefix7_ref.drop("index",1)
    prefix8_ref = prefix8[prefix8['case_len']>=9]
    prefix8_ref = prefix8_ref.reset_index()
    prefix8_ref = prefix8_ref.drop("index",1)
    prefix9 = data_dummy_prefix[data_dummy_prefix['prefix']==9]
    prefix9 = prefix9.reset_index()
    prefix9 = prefix9.drop("index",1)
    print(len(prefix9))
    print("Processing columns with prefix 9...")
    for col in prefix9:
        if col[len(col)-10:len(col)] == "prefixlen1":
            prefix9[col] = prefix1_ref[col[0:len(col)-11]]
        if col[len(col)-10:len(col)] == "prefixlen2":
            prefix9[col] = prefix2_ref[col[0:len(col)-11]]
        if col[len(col)-10:len(col)] == "prefixlen3":
            prefix9[col] = prefix3_ref[col[0:len(col)-11]]
        if col[len(col)-10:len(col)] == "prefixlen4":
            prefix9[col] = prefix4_ref[col[0:len(col)-11]]
        if col[len(col)-10:len(col)] == "prefixlen5":
            prefix9[col] = prefix5_ref[col[0:len(col)-11]]
        if col[len(col)-10:len(col)] == "prefixlen6":
            prefix9[col] = prefix6_ref[col[0:len(col)-11]]
        if col[len(col)-10:len(col)] == "prefixlen7":
            prefix9[col] = prefix7_ref[col[0:len(col)-11]]
        if col[len(col)-10:len(col)] == "prefixlen8":
            prefix9[col] = prefix8_ref[col[0:len(col)-11]]
        if col[len(col)-10:len(col)] == "prefixlen9":
            prefix9[col] = prefix9[col[0:len(col)-11]]

    prefix1_ref = prefix1[prefix1['case_len']>=10]
    prefix1_ref = prefix1_ref.reset_index()
    prefix1_ref = prefix1_ref.drop("index",1)
    prefix2_ref = prefix2[prefix2['case_len']>=10]
    prefix2_ref = prefix2_ref.reset_index()
    prefix2_ref = prefix2_ref.drop("index",1)
    prefix3_ref = prefix3[prefix3['case_len']>=10]
    prefix3_ref = prefix3_ref.reset_index()
    prefix3_ref = prefix3_ref.drop("index",1)
    prefix4_ref = prefix4[prefix4['case_len']>=10]
    prefix4_ref = prefix4_ref.reset_index()
    prefix4_ref = prefix4_ref.drop("index",1)
    prefix5_ref = prefix5[prefix5['case_len']>=10]
    prefix5_ref = prefix5_ref.reset_index()
    prefix5_ref = prefix5_ref.drop("index",1)
    prefix6_ref = prefix6[prefix6['case_len']>=10]
    prefix6_ref = prefix6_ref.reset_index()
    prefix6_ref = prefix6_ref.drop("index",1)
    prefix7_ref = prefix7[prefix7['case_len']>=10]
    prefix7_ref = prefix7_ref.reset_index()
    prefix7_ref = prefix7_ref.drop("index",1)
    prefix8_ref = prefix8[prefix8['case_len']>=10]
    prefix8_ref = prefix8_ref.reset_index()
    prefix8_ref = prefix8_ref.drop("index",1)
    prefix9_ref = prefix9[prefix9['case_len']>=10]
    prefix9_ref = prefix9_ref.reset_index()
    prefix9_ref = prefix9_ref.drop("index",1)
    prefix10 = data_dummy_prefix[data_dummy_prefix['prefix']==10]
    prefix10 = prefix10.reset_index()
    prefix10 = prefix10.drop("index",1)
    print(len(prefix10))
    print("Processing columns with prefix 10...")
    for col in prefix10:
        if col[len(col)-10:len(col)] == "prefixlen1":
            prefix10[col] = prefix1_ref[col[0:len(col)-11]]
        if col[len(col)-10:len(col)] == "prefixlen2":
            prefix10[col] = prefix2_ref[col[0:len(col)-11]]
        if col[len(col)-10:len(col)] == "prefixlen3":
            prefix10[col] = prefix3_ref[col[0:len(col)-11]]
        if col[len(col)-10:len(col)] == "prefixlen4":
            prefix10[col] = prefix4_ref[col[0:len(col)-11]]
        if col[len(col)-10:len(col)] == "prefixlen5":
            prefix10[col] = prefix5_ref[col[0:len(col)-11]]
        if col[len(col)-10:len(col)] == "prefixlen6":
            prefix10[col] = prefix6_ref[col[0:len(col)-11]]
        if col[len(col)-10:len(col)] == "prefixlen7":
            prefix10[col] = prefix7_ref[col[0:len(col)-11]]
        if col[len(col)-10:len(col)] == "prefixlen8":
            prefix10[col] = prefix8_ref[col[0:len(col)-11]]
        if col[len(col)-10:len(col)] == "prefixlen9":
            prefix10[col] = prefix9_ref[col[0:len(col)-11]]
        if col[len(col)-11:len(col)] == "prefixlen10":
            prefix10[col] = prefix10[col[0:len(col)-12]]

    prefix1 = prefix1.reset_index()
    prefix1 = prefix1.drop("index",1)
    prefix2 = prefix2.reset_index()
    prefix2 = prefix2.drop("index",1)
    prefix3 = prefix3.reset_index()
    prefix3 = prefix3.drop("index",1)
    prefix4 = prefix4.reset_index()
    prefix4 = prefix4.drop("index",1)
    prefix5 = prefix5.reset_index()
    prefix5 = prefix5.drop("index",1)
    prefix6 = prefix6.reset_index()
    prefix6 = prefix6.drop("index",1)
    prefix7 = prefix7.reset_index()
    prefix7 = prefix7.drop("index",1)
    prefix8 = prefix8.reset_index()
    prefix8 = prefix8.drop("index",1)
    prefix9 = prefix9.reset_index()
    prefix9 = prefix9.drop("index",1)
    prefix10 = prefix10.reset_index()
    prefix10 = prefix10.drop("index",1)
    prefix_index_encoding = prefix1.append(prefix2)
    prefix_index_encoding = prefix_index_encoding.append(prefix3)
    prefix_index_encoding = prefix_index_encoding.append(prefix4)
    prefix_index_encoding = prefix_index_encoding.append(prefix5)
    prefix_index_encoding = prefix_index_encoding.append(prefix6)
    prefix_index_encoding = prefix_index_encoding.append(prefix7)
    prefix_index_encoding = prefix_index_encoding.append(prefix8)
    prefix_index_encoding = prefix_index_encoding.append(prefix9)
    prefix_index_encoding = prefix_index_encoding.append(prefix10)
    prefix_index_encoding = prefix_index_encoding.reset_index()
    prefix_index_encoding = prefix_index_encoding.drop("index",1)
    prefix_index_encoding = prefix_index_encoding.filter(regex="prefixlen")
    prefix_index_encoding.to_csv("index_encoded_data_" + dataset_name + "_" + str(max_prefix) + ".csv", index = False)


if max_prefix == 40:
    prefix1 = data_dummy_prefix[data_dummy_prefix['prefix']==1]
    prefix1 = prefix1.reset_index()
    prefix1 = prefix1.drop("index",1)
    print(len(prefix1))
    print("Processing columns with prefix 1...")
    def max_prefix_40_step1(iter):
        for col in prefix1:
            if col[len(col)-10:len(col)] == "prefixlen1":
                prefix1[col] = prefix1[col[0:len(col)-11]]
        return prefix1
    if __name__ == '__main__':
        pool = multiprocessing.Pool(processes=num_cpu_cores)
        prefix1 = pool.map(max_prefix_40_step1, [0])[0]
        pool.close()
        pool.join()

    prefix1_ref = prefix1[prefix1['case_len']>=2]
    prefix1_ref = prefix1_ref.reset_index()
    prefix1_ref = prefix1_ref.drop("index",1)
    prefix2 = data_dummy_prefix[data_dummy_prefix['prefix']==2]
    prefix2 = prefix2.reset_index()
    prefix2 = prefix2.drop("index",1)
    print(len(prefix2))
    print("Processing columns with prefix 2...")
    def max_prefix_40_step2(iter):
        for col in prefix2:
            if col[len(col)-10:len(col)] == "prefixlen1":
                prefix2[col] = prefix1_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen2":
                prefix2[col] = prefix2[col[0:len(col)-11]]
        return prefix2
    if __name__ == '__main__':
        pool = multiprocessing.Pool(processes=num_cpu_cores)
        prefix2 = pool.map(max_prefix_40_step2, [0])[0]
        pool.close()
        pool.join()

    prefix1_ref = prefix1[prefix1['case_len']>=3]
    prefix1_ref = prefix1_ref.reset_index()
    prefix1_ref = prefix1_ref.drop("index",1)
    prefix2_ref = prefix2[prefix2['case_len']>=3]
    prefix2_ref = prefix2_ref.reset_index()
    prefix2_ref = prefix2_ref.drop("index",1)
    prefix3 = data_dummy_prefix[data_dummy_prefix['prefix']==3]
    prefix3 = prefix3.reset_index()
    prefix3 = prefix3.drop("index",1)
    print(len(prefix3))
    print("Processing columns with prefix 3...")
    def max_prefix_40_step3(iter):
        for col in prefix3:
            if col[len(col)-10:len(col)] == "prefixlen1":
                prefix3[col] = prefix1_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen2":
                prefix3[col] = prefix2_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen3":
                prefix3[col] = prefix3[col[0:len(col)-11]]
        return prefix3
    if __name__ == '__main__':
        pool = multiprocessing.Pool(processes=num_cpu_cores)
        prefix3 = pool.map(max_prefix_40_step3, [0])[0]
        pool.close()
        pool.join()

    prefix1_ref = prefix1[prefix1['case_len']>=4]
    prefix1_ref = prefix1_ref.reset_index()
    prefix1_ref = prefix1_ref.drop("index",1)
    prefix2_ref = prefix2[prefix2['case_len']>=4]
    prefix2_ref = prefix2_ref.reset_index()
    prefix2_ref = prefix2_ref.drop("index",1)
    prefix3_ref = prefix3[prefix3['case_len']>=4]
    prefix3_ref = prefix3_ref.reset_index()
    prefix3_ref = prefix3_ref.drop("index",1)
    prefix4 = data_dummy_prefix[data_dummy_prefix['prefix']==4]
    prefix4 = prefix4.reset_index()
    prefix4 = prefix4.drop("index",1)
    print(len(prefix4))
    print("Processing columns with prefix 4...")
    def max_prefix_40_step4(iter):
        for col in prefix4:
            if col[len(col)-10:len(col)] == "prefixlen1":
                prefix4[col] = prefix1_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen2":
                prefix4[col] = prefix2_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen3":
                prefix4[col] = prefix3_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen4":
                prefix4[col] = prefix4[col[0:len(col)-11]]
        return prefix4
    if __name__ == '__main__':
        pool = multiprocessing.Pool(processes=num_cpu_cores)
        prefix4 = pool.map(max_prefix_40_step4, [0])[0]
        pool.close()
        pool.join()

    prefix1_ref = prefix1[prefix1['case_len']>=5]
    prefix1_ref = prefix1_ref.reset_index()
    prefix1_ref = prefix1_ref.drop("index",1)
    prefix2_ref = prefix2[prefix2['case_len']>=5]
    prefix2_ref = prefix2_ref.reset_index()
    prefix2_ref = prefix2_ref.drop("index",1)
    prefix3_ref = prefix3[prefix3['case_len']>=5]
    prefix3_ref = prefix3_ref.reset_index()
    prefix3_ref = prefix3_ref.drop("index",1)
    prefix4_ref = prefix4[prefix4['case_len']>=5]
    prefix4_ref = prefix4_ref.reset_index()
    prefix4_ref = prefix4_ref.drop("index",1)
    prefix5 = data_dummy_prefix[data_dummy_prefix['prefix']==5]
    prefix5 = prefix5.reset_index()
    prefix5 = prefix5.drop("index",1)
    print(len(prefix5))
    print("Processing columns with prefix 5...")
    def max_prefix_40_step5(iter):
        for col in prefix5:
            if col[len(col)-10:len(col)] == "prefixlen1":
                prefix5[col] = prefix1_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen2":
                prefix5[col] = prefix2_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen3":
                prefix5[col] = prefix3_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen4":
                prefix5[col] = prefix4_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen5":
                prefix5[col] = prefix5[col[0:len(col)-11]]
        return prefix5
    if __name__ == '__main__':
        pool = multiprocessing.Pool(processes=num_cpu_cores)
        prefix5 = pool.map(max_prefix_40_step5, [0])[0]
        pool.close()
        pool.join()

    prefix1_ref = prefix1[prefix1['case_len']>=6]
    prefix1_ref = prefix1_ref.reset_index()
    prefix1_ref = prefix1_ref.drop("index",1)
    prefix2_ref = prefix2[prefix2['case_len']>=6]
    prefix2_ref = prefix2_ref.reset_index()
    prefix2_ref = prefix2_ref.drop("index",1)
    prefix3_ref = prefix3[prefix3['case_len']>=6]
    prefix3_ref = prefix3_ref.reset_index()
    prefix3_ref = prefix3_ref.drop("index",1)
    prefix4_ref = prefix4[prefix4['case_len']>=6]
    prefix4_ref = prefix4_ref.reset_index()
    prefix4_ref = prefix4_ref.drop("index",1)
    prefix5_ref = prefix5[prefix5['case_len']>=6]
    prefix5_ref = prefix5_ref.reset_index()
    prefix5_ref = prefix5_ref.drop("index",1)
    prefix6 = data_dummy_prefix[data_dummy_prefix['prefix']==6]
    prefix6 = prefix6.reset_index()
    prefix6 = prefix6.drop("index",1)
    print(len(prefix6))
    print("Processing columns with prefix 6...")
    def max_prefix_40_step6(iter):
        for col in prefix6:
            if col[len(col)-10:len(col)] == "prefixlen1":
                prefix6[col] = prefix1_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen2":
                prefix6[col] = prefix2_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen3":
                prefix6[col] = prefix3_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen4":
                prefix6[col] = prefix4_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen5":
                prefix6[col] = prefix5_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen6":
                prefix6[col] = prefix6[col[0:len(col)-11]]
        return prefix6
    if __name__ == '__main__':
        pool = multiprocessing.Pool(processes=num_cpu_cores)
        prefix6 = pool.map(max_prefix_40_step6, [0])[0]
        pool.close()
        pool.join()

    prefix1_ref = prefix1[prefix1['case_len']>=7]
    prefix1_ref = prefix1_ref.reset_index()
    prefix1_ref = prefix1_ref.drop("index",1)
    prefix2_ref = prefix2[prefix2['case_len']>=7]
    prefix2_ref = prefix2_ref.reset_index()
    prefix2_ref = prefix2_ref.drop("index",1)
    prefix3_ref = prefix3[prefix3['case_len']>=7]
    prefix3_ref = prefix3_ref.reset_index()
    prefix3_ref = prefix3_ref.drop("index",1)
    prefix4_ref = prefix4[prefix4['case_len']>=7]
    prefix4_ref = prefix4_ref.reset_index()
    prefix4_ref = prefix4_ref.drop("index",1)
    prefix5_ref = prefix5[prefix5['case_len']>=7]
    prefix5_ref = prefix5_ref.reset_index()
    prefix5_ref = prefix5_ref.drop("index",1)
    prefix6_ref = prefix6[prefix6['case_len']>=7]
    prefix6_ref = prefix6_ref.reset_index()
    prefix6_ref = prefix6_ref.drop("index",1)
    prefix7 = data_dummy_prefix[data_dummy_prefix['prefix']==7]
    prefix7 = prefix7.reset_index()
    prefix7 = prefix7.drop("index",1)
    print(len(prefix7))
    print("Processing columns with prefix 7...")
    def max_prefix_40_step7(iter):
        for col in prefix7:
            if col[len(col)-10:len(col)] == "prefixlen1":
                prefix7[col] = prefix1_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen2":
                prefix7[col] = prefix2_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen3":
                prefix7[col] = prefix3_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen4":
                prefix7[col] = prefix4_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen5":
                prefix7[col] = prefix5_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen6":
                prefix7[col] = prefix6_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen7":
                prefix7[col] = prefix7[col[0:len(col)-11]]
        return prefix7
    if __name__ == '__main__':
        pool = multiprocessing.Pool(processes=num_cpu_cores)
        prefix7 = pool.map(max_prefix_40_step7, [0])[0]
        pool.close()
        pool.join()

    prefix1_ref = prefix1[prefix1['case_len']>=8]
    prefix1_ref = prefix1_ref.reset_index()
    prefix1_ref = prefix1_ref.drop("index",1)
    prefix2_ref = prefix2[prefix2['case_len']>=8]
    prefix2_ref = prefix2_ref.reset_index()
    prefix2_ref = prefix2_ref.drop("index",1)
    prefix3_ref = prefix3[prefix3['case_len']>=8]
    prefix3_ref = prefix3_ref.reset_index()
    prefix3_ref = prefix3_ref.drop("index",1)
    prefix4_ref = prefix4[prefix4['case_len']>=8]
    prefix4_ref = prefix4_ref.reset_index()
    prefix4_ref = prefix4_ref.drop("index",1)
    prefix5_ref = prefix5[prefix5['case_len']>=8]
    prefix5_ref = prefix5_ref.reset_index()
    prefix5_ref = prefix5_ref.drop("index",1)
    prefix6_ref = prefix6[prefix6['case_len']>=8]
    prefix6_ref = prefix6_ref.reset_index()
    prefix6_ref = prefix6_ref.drop("index",1)
    prefix7_ref = prefix7[prefix7['case_len']>=8]
    prefix7_ref = prefix7_ref.reset_index()
    prefix7_ref = prefix7_ref.drop("index",1)
    prefix8 = data_dummy_prefix[data_dummy_prefix['prefix']==8]
    prefix8 = prefix8.reset_index()
    prefix8 = prefix8.drop("index",1)
    print(len(prefix8))
    print("Processing columns with prefix 8...")
    def max_prefix_40_step8(iter):
        for col in prefix8:
            if col[len(col)-10:len(col)] == "prefixlen1":
                prefix8[col] = prefix1_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen2":
                prefix8[col] = prefix2_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen3":
                prefix8[col] = prefix3_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen4":
                prefix8[col] = prefix4_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen5":
                prefix8[col] = prefix5_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen6":
                prefix8[col] = prefix6_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen7":
                prefix8[col] = prefix7_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen8":
                prefix8[col] = prefix8[col[0:len(col)-11]]
        return prefix8
    if __name__ == '__main__':
        pool = multiprocessing.Pool(processes=num_cpu_cores)
        prefix8 = pool.map(max_prefix_40_step8, [0])[0]
        pool.close()
        pool.join()

    prefix1_ref = prefix1[prefix1['case_len']>=9]
    prefix1_ref = prefix1_ref.reset_index()
    prefix1_ref = prefix1_ref.drop("index",1)
    prefix2_ref = prefix2[prefix2['case_len']>=9]
    prefix2_ref = prefix2_ref.reset_index()
    prefix2_ref = prefix2_ref.drop("index",1)
    prefix3_ref = prefix3[prefix3['case_len']>=9]
    prefix3_ref = prefix3_ref.reset_index()
    prefix3_ref = prefix3_ref.drop("index",1)
    prefix4_ref = prefix4[prefix4['case_len']>=9]
    prefix4_ref = prefix4_ref.reset_index()
    prefix4_ref = prefix4_ref.drop("index",1)
    prefix5_ref = prefix5[prefix5['case_len']>=9]
    prefix5_ref = prefix5_ref.reset_index()
    prefix5_ref = prefix5_ref.drop("index",1)
    prefix6_ref = prefix6[prefix6['case_len']>=9]
    prefix6_ref = prefix6_ref.reset_index()
    prefix6_ref = prefix6_ref.drop("index",1)
    prefix7_ref = prefix7[prefix7['case_len']>=9]
    prefix7_ref = prefix7_ref.reset_index()
    prefix7_ref = prefix7_ref.drop("index",1)
    prefix8_ref = prefix8[prefix8['case_len']>=9]
    prefix8_ref = prefix8_ref.reset_index()
    prefix8_ref = prefix8_ref.drop("index",1)
    prefix9 = data_dummy_prefix[data_dummy_prefix['prefix']==9]
    prefix9 = prefix9.reset_index()
    prefix9 = prefix9.drop("index",1)
    print(len(prefix9))
    print("Processing columns with prefix 9...")
    def max_prefix_40_step9(iter):
        for col in prefix9:
            if col[len(col)-10:len(col)] == "prefixlen1":
                prefix9[col] = prefix1_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen2":
                prefix9[col] = prefix2_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen3":
                prefix9[col] = prefix3_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen4":
                prefix9[col] = prefix4_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen5":
                prefix9[col] = prefix5_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen6":
                prefix9[col] = prefix6_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen7":
                prefix9[col] = prefix7_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen8":
                prefix9[col] = prefix8_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen9":
                prefix9[col] = prefix9[col[0:len(col)-11]]
        return prefix9
    if __name__ == '__main__':
        pool = multiprocessing.Pool(processes=num_cpu_cores)
        prefix9 = pool.map(max_prefix_40_step9, [0])[0]
        pool.close()
        pool.join()

    prefix1_ref = prefix1[prefix1['case_len']>=10]
    prefix1_ref = prefix1_ref.reset_index()
    prefix1_ref = prefix1_ref.drop("index",1)
    prefix2_ref = prefix2[prefix2['case_len']>=10]
    prefix2_ref = prefix2_ref.reset_index()
    prefix2_ref = prefix2_ref.drop("index",1)
    prefix3_ref = prefix3[prefix3['case_len']>=10]
    prefix3_ref = prefix3_ref.reset_index()
    prefix3_ref = prefix3_ref.drop("index",1)
    prefix4_ref = prefix4[prefix4['case_len']>=10]
    prefix4_ref = prefix4_ref.reset_index()
    prefix4_ref = prefix4_ref.drop("index",1)
    prefix5_ref = prefix5[prefix5['case_len']>=10]
    prefix5_ref = prefix5_ref.reset_index()
    prefix5_ref = prefix5_ref.drop("index",1)
    prefix6_ref = prefix6[prefix6['case_len']>=10]
    prefix6_ref = prefix6_ref.reset_index()
    prefix6_ref = prefix6_ref.drop("index",1)
    prefix7_ref = prefix7[prefix7['case_len']>=10]
    prefix7_ref = prefix7_ref.reset_index()
    prefix7_ref = prefix7_ref.drop("index",1)
    prefix8_ref = prefix8[prefix8['case_len']>=10]
    prefix8_ref = prefix8_ref.reset_index()
    prefix8_ref = prefix8_ref.drop("index",1)
    prefix9_ref = prefix9[prefix9['case_len']>=10]
    prefix9_ref = prefix9_ref.reset_index()
    prefix9_ref = prefix9_ref.drop("index",1)
    prefix10 = data_dummy_prefix[data_dummy_prefix['prefix']==10]
    prefix10 = prefix10.reset_index()
    prefix10 = prefix10.drop("index",1)
    print(len(prefix10))
    print("Processing columns with prefix 10...")
    def max_prefix_40_step10(iter):
        for col in prefix10:
            if col[len(col)-10:len(col)] == "prefixlen1":
                prefix10[col] = prefix1_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen2":
                prefix10[col] = prefix2_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen3":
                prefix10[col] = prefix3_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen4":
                prefix10[col] = prefix4_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen5":
                prefix10[col] = prefix5_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen6":
                prefix10[col] = prefix6_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen7":
                prefix10[col] = prefix7_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen8":
                prefix10[col] = prefix8_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen9":
                prefix10[col] = prefix9_ref[col[0:len(col)-11]]
            if col[len(col)-11:len(col)] == "prefixlen10":
                prefix10[col] = prefix10[col[0:len(col)-12]]
        return prefix10
    if __name__ == '__main__':
        pool = multiprocessing.Pool(processes=num_cpu_cores)
        prefix10 = pool.map(max_prefix_40_step10, [0])[0]
        pool.close()
        pool.join()

    prefix1_ref = prefix1[prefix1['case_len']>=11]
    prefix1_ref = prefix1_ref.reset_index()
    prefix1_ref = prefix1_ref.drop("index",1)
    prefix2_ref = prefix2[prefix2['case_len']>=11]
    prefix2_ref = prefix2_ref.reset_index()
    prefix2_ref = prefix2_ref.drop("index",1)
    prefix3_ref = prefix3[prefix3['case_len']>=11]
    prefix3_ref = prefix3_ref.reset_index()
    prefix3_ref = prefix3_ref.drop("index",1)
    prefix4_ref = prefix4[prefix4['case_len']>=11]
    prefix4_ref = prefix4_ref.reset_index()
    prefix4_ref = prefix4_ref.drop("index",1)
    prefix5_ref = prefix5[prefix5['case_len']>=11]
    prefix5_ref = prefix5_ref.reset_index()
    prefix5_ref = prefix5_ref.drop("index",1)
    prefix6_ref = prefix6[prefix6['case_len']>=11]
    prefix6_ref = prefix6_ref.reset_index()
    prefix6_ref = prefix6_ref.drop("index",1)
    prefix7_ref = prefix7[prefix7['case_len']>=11]
    prefix7_ref = prefix7_ref.reset_index()
    prefix7_ref = prefix7_ref.drop("index",1)
    prefix8_ref = prefix8[prefix8['case_len']>=11]
    prefix8_ref = prefix8_ref.reset_index()
    prefix8_ref = prefix8_ref.drop("index",1)
    prefix9_ref = prefix9[prefix9['case_len']>=11]
    prefix9_ref = prefix9_ref.reset_index()
    prefix9_ref = prefix9_ref.drop("index",1)
    prefix10_ref = prefix10[prefix10['case_len']>=11]
    prefix10_ref = prefix10_ref.reset_index()
    prefix10_ref = prefix10_ref.drop("index",1)
    prefix11 = data_dummy_prefix[data_dummy_prefix['prefix']==11]
    prefix11 = prefix11.reset_index()
    prefix11 = prefix11.drop("index",1)
    print(len(prefix11))
    print("Processing columns with prefix 11...")
    def max_prefix_40_step11(iter):
        for col in prefix11:
            if col[len(col)-10:len(col)] == "prefixlen1":
                prefix11[col] = prefix1_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen2":
                prefix11[col] = prefix2_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen3":
                prefix11[col] = prefix3_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen4":
                prefix11[col] = prefix4_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen5":
                prefix11[col] = prefix5_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen6":
                prefix11[col] = prefix6_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen7":
                prefix11[col] = prefix7_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen8":
                prefix11[col] = prefix8_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen9":
                prefix11[col] = prefix9_ref[col[0:len(col)-11]]
            if col[len(col)-11:len(col)] == "prefixlen10":
                prefix11[col] = prefix10_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen11":
                prefix11[col] = prefix11[col[0:len(col)-12]]
        return prefix11
    if __name__ == '__main__':
        pool = multiprocessing.Pool(processes=num_cpu_cores)
        prefix11 = pool.map(max_prefix_40_step11, [0])[0]
        pool.close()
        pool.join()

    prefix1_ref = prefix1[prefix1['case_len']>=12]
    prefix1_ref = prefix1_ref.reset_index()
    prefix1_ref = prefix1_ref.drop("index",1)
    prefix2_ref = prefix2[prefix2['case_len']>=12]
    prefix2_ref = prefix2_ref.reset_index()
    prefix2_ref = prefix2_ref.drop("index",1)
    prefix3_ref = prefix3[prefix3['case_len']>=12]
    prefix3_ref = prefix3_ref.reset_index()
    prefix3_ref = prefix3_ref.drop("index",1)
    prefix4_ref = prefix4[prefix4['case_len']>=12]
    prefix4_ref = prefix4_ref.reset_index()
    prefix4_ref = prefix4_ref.drop("index",1)
    prefix5_ref = prefix5[prefix5['case_len']>=12]
    prefix5_ref = prefix5_ref.reset_index()
    prefix5_ref = prefix5_ref.drop("index",1)
    prefix6_ref = prefix6[prefix6['case_len']>=12]
    prefix6_ref = prefix6_ref.reset_index()
    prefix6_ref = prefix6_ref.drop("index",1)
    prefix7_ref = prefix7[prefix7['case_len']>=12]
    prefix7_ref = prefix7_ref.reset_index()
    prefix7_ref = prefix7_ref.drop("index",1)
    prefix8_ref = prefix8[prefix8['case_len']>=12]
    prefix8_ref = prefix8_ref.reset_index()
    prefix8_ref = prefix8_ref.drop("index",1)
    prefix9_ref = prefix9[prefix9['case_len']>=12]
    prefix9_ref = prefix9_ref.reset_index()
    prefix9_ref = prefix9_ref.drop("index",1)
    prefix10_ref = prefix10[prefix10['case_len']>=12]
    prefix10_ref = prefix10_ref.reset_index()
    prefix10_ref = prefix10_ref.drop("index",1)
    prefix11_ref = prefix11[prefix11['case_len']>=12]
    prefix11_ref = prefix11_ref.reset_index()
    prefix11_ref = prefix11_ref.drop("index",1)
    prefix12 = data_dummy_prefix[data_dummy_prefix['prefix']==12]
    prefix12 = prefix12.reset_index()
    prefix12 = prefix12.drop("index",1)
    print(len(prefix12))
    print("Processing columns with prefix 12...")
    def max_prefix_40_step12(iter):
        for col in prefix12:
            if col[len(col)-10:len(col)] == "prefixlen1":
                prefix12[col] = prefix1_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen2":
                prefix12[col] = prefix2_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen3":
                prefix12[col] = prefix3_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen4":
                prefix12[col] = prefix4_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen5":
                prefix12[col] = prefix5_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen6":
                prefix12[col] = prefix6_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen7":
                prefix12[col] = prefix7_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen8":
                prefix12[col] = prefix8_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen9":
                prefix12[col] = prefix9_ref[col[0:len(col)-11]]
            if col[len(col)-11:len(col)] == "prefixlen10":
                prefix12[col] = prefix10_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen11":
                prefix12[col] = prefix11_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen12":
                prefix12[col] = prefix12[col[0:len(col)-12]]
        return prefix12
    if __name__ == '__main__':
        pool = multiprocessing.Pool(processes=num_cpu_cores)
        prefix12 = pool.map(max_prefix_40_step12, [0])[0]
        pool.close()
        pool.join()

    prefix1_ref = prefix1[prefix1['case_len']>=13]
    prefix1_ref = prefix1_ref.reset_index()
    prefix1_ref = prefix1_ref.drop("index",1)
    prefix2_ref = prefix2[prefix2['case_len']>=13]
    prefix2_ref = prefix2_ref.reset_index()
    prefix2_ref = prefix2_ref.drop("index",1)
    prefix3_ref = prefix3[prefix3['case_len']>=13]
    prefix3_ref = prefix3_ref.reset_index()
    prefix3_ref = prefix3_ref.drop("index",1)
    prefix4_ref = prefix4[prefix4['case_len']>=13]
    prefix4_ref = prefix4_ref.reset_index()
    prefix4_ref = prefix4_ref.drop("index",1)
    prefix5_ref = prefix5[prefix5['case_len']>=13]
    prefix5_ref = prefix5_ref.reset_index()
    prefix5_ref = prefix5_ref.drop("index",1)
    prefix6_ref = prefix6[prefix6['case_len']>=13]
    prefix6_ref = prefix6_ref.reset_index()
    prefix6_ref = prefix6_ref.drop("index",1)
    prefix7_ref = prefix7[prefix7['case_len']>=13]
    prefix7_ref = prefix7_ref.reset_index()
    prefix7_ref = prefix7_ref.drop("index",1)
    prefix8_ref = prefix8[prefix8['case_len']>=13]
    prefix8_ref = prefix8_ref.reset_index()
    prefix8_ref = prefix8_ref.drop("index",1)
    prefix9_ref = prefix9[prefix9['case_len']>=13]
    prefix9_ref = prefix9_ref.reset_index()
    prefix9_ref = prefix9_ref.drop("index",1)
    prefix10_ref = prefix10[prefix10['case_len']>=13]
    prefix10_ref = prefix10_ref.reset_index()
    prefix10_ref = prefix10_ref.drop("index",1)
    prefix11_ref = prefix11[prefix11['case_len']>=13]
    prefix11_ref = prefix11_ref.reset_index()
    prefix11_ref = prefix11_ref.drop("index",1)
    prefix12_ref = prefix12[prefix12['case_len']>=13]
    prefix12_ref = prefix12_ref.reset_index()
    prefix12_ref = prefix12_ref.drop("index",1)
    prefix13 = data_dummy_prefix[data_dummy_prefix['prefix']==13]
    prefix13 = prefix13.reset_index()
    prefix13 = prefix13.drop("index",1)
    print(len(prefix13))
    print("Processing columns with prefix 13...")
    def max_prefix_40_step13(iter):
        for col in prefix13:
            if col[len(col)-10:len(col)] == "prefixlen1":
                prefix13[col] = prefix1_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen2":
                prefix13[col] = prefix2_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen3":
                prefix13[col] = prefix3_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen4":
                prefix13[col] = prefix4_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen5":
                prefix13[col] = prefix5_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen6":
                prefix13[col] = prefix6_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen7":
                prefix13[col] = prefix7_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen8":
                prefix13[col] = prefix8_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen9":
                prefix13[col] = prefix9_ref[col[0:len(col)-11]]
            if col[len(col)-11:len(col)] == "prefixlen10":
                prefix13[col] = prefix10_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen11":
                prefix13[col] = prefix11_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen12":
                prefix13[col] = prefix12_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen13":
                prefix13[col] = prefix13[col[0:len(col)-12]]
        return prefix13
    if __name__ == '__main__':
        pool = multiprocessing.Pool(processes=num_cpu_cores)
        prefix13 = pool.map(max_prefix_40_step13, [0])[0]
        pool.close()
        pool.join()

    prefix1_ref = prefix1[prefix1['case_len']>=14]
    prefix1_ref = prefix1_ref.reset_index()
    prefix1_ref = prefix1_ref.drop("index",1)
    prefix2_ref = prefix2[prefix2['case_len']>=14]
    prefix2_ref = prefix2_ref.reset_index()
    prefix2_ref = prefix2_ref.drop("index",1)
    prefix3_ref = prefix3[prefix3['case_len']>=14]
    prefix3_ref = prefix3_ref.reset_index()
    prefix3_ref = prefix3_ref.drop("index",1)
    prefix4_ref = prefix4[prefix4['case_len']>=14]
    prefix4_ref = prefix4_ref.reset_index()
    prefix4_ref = prefix4_ref.drop("index",1)
    prefix5_ref = prefix5[prefix5['case_len']>=14]
    prefix5_ref = prefix5_ref.reset_index()
    prefix5_ref = prefix5_ref.drop("index",1)
    prefix6_ref = prefix6[prefix6['case_len']>=14]
    prefix6_ref = prefix6_ref.reset_index()
    prefix6_ref = prefix6_ref.drop("index",1)
    prefix7_ref = prefix7[prefix7['case_len']>=14]
    prefix7_ref = prefix7_ref.reset_index()
    prefix7_ref = prefix7_ref.drop("index",1)
    prefix8_ref = prefix8[prefix8['case_len']>=14]
    prefix8_ref = prefix8_ref.reset_index()
    prefix8_ref = prefix8_ref.drop("index",1)
    prefix9_ref = prefix9[prefix9['case_len']>=14]
    prefix9_ref = prefix9_ref.reset_index()
    prefix9_ref = prefix9_ref.drop("index",1)
    prefix10_ref = prefix10[prefix10['case_len']>=14]
    prefix10_ref = prefix10_ref.reset_index()
    prefix10_ref = prefix10_ref.drop("index",1)
    prefix11_ref = prefix11[prefix11['case_len']>=14]
    prefix11_ref = prefix11_ref.reset_index()
    prefix11_ref = prefix11_ref.drop("index",1)
    prefix12_ref = prefix12[prefix12['case_len']>=14]
    prefix12_ref = prefix12_ref.reset_index()
    prefix12_ref = prefix12_ref.drop("index",1)
    prefix13_ref = prefix13[prefix13['case_len']>=14]
    prefix13_ref = prefix13_ref.reset_index()
    prefix13_ref = prefix13_ref.drop("index",1)
    prefix14 = data_dummy_prefix[data_dummy_prefix['prefix']==14]
    prefix14 = prefix14.reset_index()
    prefix14 = prefix14.drop("index",1)
    print(len(prefix14))
    print("Processing columns with prefix 14...")
    def max_prefix_40_step14(iter):
        for col in prefix14:
            if col[len(col)-10:len(col)] == "prefixlen1":
                prefix14[col] = prefix1_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen2":
                prefix14[col] = prefix2_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen3":
                prefix14[col] = prefix3_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen4":
                prefix14[col] = prefix4_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen5":
                prefix14[col] = prefix5_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen6":
                prefix14[col] = prefix6_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen7":
                prefix14[col] = prefix7_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen8":
                prefix14[col] = prefix8_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen9":
                prefix14[col] = prefix9_ref[col[0:len(col)-11]]
            if col[len(col)-11:len(col)] == "prefixlen10":
                prefix14[col] = prefix10_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen11":
                prefix14[col] = prefix11_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen12":
                prefix14[col] = prefix12_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen13":
                prefix14[col] = prefix13_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen14":
                prefix14[col] = prefix14[col[0:len(col)-12]]
        return prefix14
    if __name__ == '__main__':
        pool = multiprocessing.Pool(processes=num_cpu_cores)
        prefix14 = pool.map(max_prefix_40_step14, [0])[0]
        pool.close()
        pool.join()

    prefix1_ref = prefix1[prefix1['case_len']>=15]
    prefix1_ref = prefix1_ref.reset_index()
    prefix1_ref = prefix1_ref.drop("index",1)
    prefix2_ref = prefix2[prefix2['case_len']>=15]
    prefix2_ref = prefix2_ref.reset_index()
    prefix2_ref = prefix2_ref.drop("index",1)
    prefix3_ref = prefix3[prefix3['case_len']>=15]
    prefix3_ref = prefix3_ref.reset_index()
    prefix3_ref = prefix3_ref.drop("index",1)
    prefix4_ref = prefix4[prefix4['case_len']>=15]
    prefix4_ref = prefix4_ref.reset_index()
    prefix4_ref = prefix4_ref.drop("index",1)
    prefix5_ref = prefix5[prefix5['case_len']>=15]
    prefix5_ref = prefix5_ref.reset_index()
    prefix5_ref = prefix5_ref.drop("index",1)
    prefix6_ref = prefix6[prefix6['case_len']>=15]
    prefix6_ref = prefix6_ref.reset_index()
    prefix6_ref = prefix6_ref.drop("index",1)
    prefix7_ref = prefix7[prefix7['case_len']>=15]
    prefix7_ref = prefix7_ref.reset_index()
    prefix7_ref = prefix7_ref.drop("index",1)
    prefix8_ref = prefix8[prefix8['case_len']>=15]
    prefix8_ref = prefix8_ref.reset_index()
    prefix8_ref = prefix8_ref.drop("index",1)
    prefix9_ref = prefix9[prefix9['case_len']>=15]
    prefix9_ref = prefix9_ref.reset_index()
    prefix9_ref = prefix9_ref.drop("index",1)
    prefix10_ref = prefix10[prefix10['case_len']>=15]
    prefix10_ref = prefix10_ref.reset_index()
    prefix10_ref = prefix10_ref.drop("index",1)
    prefix11_ref = prefix11[prefix11['case_len']>=15]
    prefix11_ref = prefix11_ref.reset_index()
    prefix11_ref = prefix11_ref.drop("index",1)
    prefix12_ref = prefix12[prefix12['case_len']>=15]
    prefix12_ref = prefix12_ref.reset_index()
    prefix12_ref = prefix12_ref.drop("index",1)
    prefix13_ref = prefix13[prefix13['case_len']>=15]
    prefix13_ref = prefix13_ref.reset_index()
    prefix13_ref = prefix13_ref.drop("index",1)
    prefix14_ref = prefix14[prefix14['case_len']>=15]
    prefix14_ref = prefix14_ref.reset_index()
    prefix14_ref = prefix14_ref.drop("index",1)
    prefix15 = data_dummy_prefix[data_dummy_prefix['prefix']==15]
    prefix15 = prefix15.reset_index()
    prefix15 = prefix15.drop("index",1)
    print(len(prefix15))
    print("Processing columns with prefix 15...")
    def max_prefix_40_step15(iter):
        for col in prefix15:
            if col[len(col)-10:len(col)] == "prefixlen1":
                prefix15[col] = prefix1_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen2":
                prefix15[col] = prefix2_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen3":
                prefix15[col] = prefix3_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen4":
                prefix15[col] = prefix4_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen5":
                prefix15[col] = prefix5_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen6":
                prefix15[col] = prefix6_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen7":
                prefix15[col] = prefix7_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen8":
                prefix15[col] = prefix8_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen9":
                prefix15[col] = prefix9_ref[col[0:len(col)-11]]
            if col[len(col)-11:len(col)] == "prefixlen10":
                prefix15[col] = prefix10_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen11":
                prefix15[col] = prefix11_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen12":
                prefix15[col] = prefix12_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen13":
                prefix15[col] = prefix13_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen14":
                prefix15[col] = prefix14_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen15":
                prefix15[col] = prefix15[col[0:len(col)-12]]
        return prefix15
    if __name__ == '__main__':
        pool = multiprocessing.Pool(processes=num_cpu_cores)
        prefix15 = pool.map(max_prefix_40_step15, [0])[0]
        pool.close()
        pool.join()

    prefix1_ref = prefix1[prefix1['case_len']>=16]
    prefix1_ref = prefix1_ref.reset_index()
    prefix1_ref = prefix1_ref.drop("index",1)
    prefix2_ref = prefix2[prefix2['case_len']>=16]
    prefix2_ref = prefix2_ref.reset_index()
    prefix2_ref = prefix2_ref.drop("index",1)
    prefix3_ref = prefix3[prefix3['case_len']>=16]
    prefix3_ref = prefix3_ref.reset_index()
    prefix3_ref = prefix3_ref.drop("index",1)
    prefix4_ref = prefix4[prefix4['case_len']>=16]
    prefix4_ref = prefix4_ref.reset_index()
    prefix4_ref = prefix4_ref.drop("index",1)
    prefix5_ref = prefix5[prefix5['case_len']>=16]
    prefix5_ref = prefix5_ref.reset_index()
    prefix5_ref = prefix5_ref.drop("index",1)
    prefix6_ref = prefix6[prefix6['case_len']>=16]
    prefix6_ref = prefix6_ref.reset_index()
    prefix6_ref = prefix6_ref.drop("index",1)
    prefix7_ref = prefix7[prefix7['case_len']>=16]
    prefix7_ref = prefix7_ref.reset_index()
    prefix7_ref = prefix7_ref.drop("index",1)
    prefix8_ref = prefix8[prefix8['case_len']>=16]
    prefix8_ref = prefix8_ref.reset_index()
    prefix8_ref = prefix8_ref.drop("index",1)
    prefix9_ref = prefix9[prefix9['case_len']>=16]
    prefix9_ref = prefix9_ref.reset_index()
    prefix9_ref = prefix9_ref.drop("index",1)
    prefix10_ref = prefix10[prefix10['case_len']>=16]
    prefix10_ref = prefix10_ref.reset_index()
    prefix10_ref = prefix10_ref.drop("index",1)
    prefix11_ref = prefix11[prefix11['case_len']>=16]
    prefix11_ref = prefix11_ref.reset_index()
    prefix11_ref = prefix11_ref.drop("index",1)
    prefix12_ref = prefix12[prefix12['case_len']>=16]
    prefix12_ref = prefix12_ref.reset_index()
    prefix12_ref = prefix12_ref.drop("index",1)
    prefix13_ref = prefix13[prefix13['case_len']>=16]
    prefix13_ref = prefix13_ref.reset_index()
    prefix13_ref = prefix13_ref.drop("index",1)
    prefix14_ref = prefix14[prefix14['case_len']>=16]
    prefix14_ref = prefix14_ref.reset_index()
    prefix14_ref = prefix14_ref.drop("index",1)
    prefix15_ref = prefix15[prefix15['case_len']>=16]
    prefix15_ref = prefix15_ref.reset_index()
    prefix15_ref = prefix15_ref.drop("index",1)
    prefix16 = data_dummy_prefix[data_dummy_prefix['prefix']==16]
    prefix16 = prefix16.reset_index()
    prefix16 = prefix16.drop("index",1)
    print(len(prefix16))
    print("Processing columns with prefix 16...")
    def max_prefix_40_step16(iter):
        for col in prefix16:
            if col[len(col)-10:len(col)] == "prefixlen1":
                prefix16[col] = prefix1_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen2":
                prefix16[col] = prefix2_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen3":
                prefix16[col] = prefix3_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen4":
                prefix16[col] = prefix4_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen5":
                prefix16[col] = prefix5_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen6":
                prefix16[col] = prefix6_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen7":
                prefix16[col] = prefix7_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen8":
                prefix16[col] = prefix8_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen9":
                prefix16[col] = prefix9_ref[col[0:len(col)-11]]
            if col[len(col)-11:len(col)] == "prefixlen10":
                prefix16[col] = prefix10_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen11":
                prefix16[col] = prefix11_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen12":
                prefix16[col] = prefix12_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen13":
                prefix16[col] = prefix13_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen14":
                prefix16[col] = prefix14_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen15":
                prefix16[col] = prefix15_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen16":
                prefix16[col] = prefix16[col[0:len(col)-12]]
        return prefix16
    if __name__ == '__main__':
        pool = multiprocessing.Pool(processes=num_cpu_cores)
        prefix16 = pool.map(max_prefix_40_step16, [0])[0]
        pool.close()
        pool.join()

    prefix1_ref = prefix1[prefix1['case_len']>=17]
    prefix1_ref = prefix1_ref.reset_index()
    prefix1_ref = prefix1_ref.drop("index",1)
    prefix2_ref = prefix2[prefix2['case_len']>=17]
    prefix2_ref = prefix2_ref.reset_index()
    prefix2_ref = prefix2_ref.drop("index",1)
    prefix3_ref = prefix3[prefix3['case_len']>=17]
    prefix3_ref = prefix3_ref.reset_index()
    prefix3_ref = prefix3_ref.drop("index",1)
    prefix4_ref = prefix4[prefix4['case_len']>=17]
    prefix4_ref = prefix4_ref.reset_index()
    prefix4_ref = prefix4_ref.drop("index",1)
    prefix5_ref = prefix5[prefix5['case_len']>=17]
    prefix5_ref = prefix5_ref.reset_index()
    prefix5_ref = prefix5_ref.drop("index",1)
    prefix6_ref = prefix6[prefix6['case_len']>=17]
    prefix6_ref = prefix6_ref.reset_index()
    prefix6_ref = prefix6_ref.drop("index",1)
    prefix7_ref = prefix7[prefix7['case_len']>=17]
    prefix7_ref = prefix7_ref.reset_index()
    prefix7_ref = prefix7_ref.drop("index",1)
    prefix8_ref = prefix8[prefix8['case_len']>=17]
    prefix8_ref = prefix8_ref.reset_index()
    prefix8_ref = prefix8_ref.drop("index",1)
    prefix9_ref = prefix9[prefix9['case_len']>=17]
    prefix9_ref = prefix9_ref.reset_index()
    prefix9_ref = prefix9_ref.drop("index",1)
    prefix10_ref = prefix10[prefix10['case_len']>=17]
    prefix10_ref = prefix10_ref.reset_index()
    prefix10_ref = prefix10_ref.drop("index",1)
    prefix11_ref = prefix11[prefix11['case_len']>=17]
    prefix11_ref = prefix11_ref.reset_index()
    prefix11_ref = prefix11_ref.drop("index",1)
    prefix12_ref = prefix12[prefix12['case_len']>=17]
    prefix12_ref = prefix12_ref.reset_index()
    prefix12_ref = prefix12_ref.drop("index",1)
    prefix13_ref = prefix13[prefix13['case_len']>=17]
    prefix13_ref = prefix13_ref.reset_index()
    prefix13_ref = prefix13_ref.drop("index",1)
    prefix14_ref = prefix14[prefix14['case_len']>=17]
    prefix14_ref = prefix14_ref.reset_index()
    prefix14_ref = prefix14_ref.drop("index",1)
    prefix15_ref = prefix15[prefix15['case_len']>=17]
    prefix15_ref = prefix15_ref.reset_index()
    prefix15_ref = prefix15_ref.drop("index",1)
    prefix16_ref = prefix16[prefix16['case_len']>=17]
    prefix16_ref = prefix16_ref.reset_index()
    prefix16_ref = prefix16_ref.drop("index",1)
    prefix17 = data_dummy_prefix[data_dummy_prefix['prefix']==17]
    prefix17 = prefix17.reset_index()
    prefix17 = prefix17.drop("index",1)
    print(len(prefix17))
    print("Processing columns with prefix 17...")
    def max_prefix_40_step17(iter):
        for col in prefix17:
            if col[len(col)-10:len(col)] == "prefixlen1":
                prefix17[col] = prefix1_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen2":
                prefix17[col] = prefix2_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen3":
                prefix17[col] = prefix3_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen4":
                prefix17[col] = prefix4_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen5":
                prefix17[col] = prefix5_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen6":
                prefix17[col] = prefix6_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen7":
                prefix17[col] = prefix7_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen8":
                prefix17[col] = prefix8_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen9":
                prefix17[col] = prefix9_ref[col[0:len(col)-11]]
            if col[len(col)-11:len(col)] == "prefixlen10":
                prefix17[col] = prefix10_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen11":
                prefix17[col] = prefix11_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen12":
                prefix17[col] = prefix12_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen13":
                prefix17[col] = prefix13_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen14":
                prefix17[col] = prefix14_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen15":
                prefix17[col] = prefix15_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen16":
                prefix17[col] = prefix16_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen17":
                prefix17[col] = prefix17[col[0:len(col)-12]]
        return prefix17
    if __name__ == '__main__':
        pool = multiprocessing.Pool(processes=num_cpu_cores)
        prefix17 = pool.map(max_prefix_40_step17, [0])[0]
        pool.close()
        pool.join()

    prefix1_ref = prefix1[prefix1['case_len']>=18]
    prefix1_ref = prefix1_ref.reset_index()
    prefix1_ref = prefix1_ref.drop("index",1)
    prefix2_ref = prefix2[prefix2['case_len']>=18]
    prefix2_ref = prefix2_ref.reset_index()
    prefix2_ref = prefix2_ref.drop("index",1)
    prefix3_ref = prefix3[prefix3['case_len']>=18]
    prefix3_ref = prefix3_ref.reset_index()
    prefix3_ref = prefix3_ref.drop("index",1)
    prefix4_ref = prefix4[prefix4['case_len']>=18]
    prefix4_ref = prefix4_ref.reset_index()
    prefix4_ref = prefix4_ref.drop("index",1)
    prefix5_ref = prefix5[prefix5['case_len']>=18]
    prefix5_ref = prefix5_ref.reset_index()
    prefix5_ref = prefix5_ref.drop("index",1)
    prefix6_ref = prefix6[prefix6['case_len']>=18]
    prefix6_ref = prefix6_ref.reset_index()
    prefix6_ref = prefix6_ref.drop("index",1)
    prefix7_ref = prefix7[prefix7['case_len']>=18]
    prefix7_ref = prefix7_ref.reset_index()
    prefix7_ref = prefix7_ref.drop("index",1)
    prefix8_ref = prefix8[prefix8['case_len']>=18]
    prefix8_ref = prefix8_ref.reset_index()
    prefix8_ref = prefix8_ref.drop("index",1)
    prefix9_ref = prefix9[prefix9['case_len']>=18]
    prefix9_ref = prefix9_ref.reset_index()
    prefix9_ref = prefix9_ref.drop("index",1)
    prefix10_ref = prefix10[prefix10['case_len']>=18]
    prefix10_ref = prefix10_ref.reset_index()
    prefix10_ref = prefix10_ref.drop("index",1)
    prefix11_ref = prefix11[prefix11['case_len']>=18]
    prefix11_ref = prefix11_ref.reset_index()
    prefix11_ref = prefix11_ref.drop("index",1)
    prefix12_ref = prefix12[prefix12['case_len']>=18]
    prefix12_ref = prefix12_ref.reset_index()
    prefix12_ref = prefix12_ref.drop("index",1)
    prefix13_ref = prefix13[prefix13['case_len']>=18]
    prefix13_ref = prefix13_ref.reset_index()
    prefix13_ref = prefix13_ref.drop("index",1)
    prefix14_ref = prefix14[prefix14['case_len']>=18]
    prefix14_ref = prefix14_ref.reset_index()
    prefix14_ref = prefix14_ref.drop("index",1)
    prefix15_ref = prefix15[prefix15['case_len']>=18]
    prefix15_ref = prefix15_ref.reset_index()
    prefix15_ref = prefix15_ref.drop("index",1)
    prefix16_ref = prefix16[prefix16['case_len']>=18]
    prefix16_ref = prefix16_ref.reset_index()
    prefix16_ref = prefix16_ref.drop("index",1)
    prefix17_ref = prefix17[prefix17['case_len']>=18]
    prefix17_ref = prefix17_ref.reset_index()
    prefix17_ref = prefix17_ref.drop("index",1)
    prefix18 = data_dummy_prefix[data_dummy_prefix['prefix']==18]
    prefix18 = prefix18.reset_index()
    prefix18 = prefix18.drop("index",1)
    print(len(prefix18))
    print("Processing columns with prefix 18...")
    def max_prefix_40_step18(iter):
        for col in prefix18:
            if col[len(col)-10:len(col)] == "prefixlen1":
                prefix18[col] = prefix1_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen2":
                prefix18[col] = prefix2_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen3":
                prefix18[col] = prefix3_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen4":
                prefix18[col] = prefix4_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen5":
                prefix18[col] = prefix5_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen6":
                prefix18[col] = prefix6_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen7":
                prefix18[col] = prefix7_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen8":
                prefix18[col] = prefix8_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen9":
                prefix18[col] = prefix9_ref[col[0:len(col)-11]]
            if col[len(col)-11:len(col)] == "prefixlen10":
                prefix18[col] = prefix10_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen11":
                prefix18[col] = prefix11_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen12":
                prefix18[col] = prefix12_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen13":
                prefix18[col] = prefix13_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen14":
                prefix18[col] = prefix14_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen15":
                prefix18[col] = prefix15_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen16":
                prefix18[col] = prefix16_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen17":
                prefix18[col] = prefix17_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen18":
                prefix18[col] = prefix18[col[0:len(col)-12]]
        return prefix18
    if __name__ == '__main__':
        pool = multiprocessing.Pool(processes=num_cpu_cores)
        prefix18 = pool.map(max_prefix_40_step18, [0])[0]
        pool.close()
        pool.join()

    prefix1_ref = prefix1[prefix1['case_len']>=19]
    prefix1_ref = prefix1_ref.reset_index()
    prefix1_ref = prefix1_ref.drop("index",1)
    prefix2_ref = prefix2[prefix2['case_len']>=19]
    prefix2_ref = prefix2_ref.reset_index()
    prefix2_ref = prefix2_ref.drop("index",1)
    prefix3_ref = prefix3[prefix3['case_len']>=19]
    prefix3_ref = prefix3_ref.reset_index()
    prefix3_ref = prefix3_ref.drop("index",1)
    prefix4_ref = prefix4[prefix4['case_len']>=19]
    prefix4_ref = prefix4_ref.reset_index()
    prefix4_ref = prefix4_ref.drop("index",1)
    prefix5_ref = prefix5[prefix5['case_len']>=19]
    prefix5_ref = prefix5_ref.reset_index()
    prefix5_ref = prefix5_ref.drop("index",1)
    prefix6_ref = prefix6[prefix6['case_len']>=19]
    prefix6_ref = prefix6_ref.reset_index()
    prefix6_ref = prefix6_ref.drop("index",1)
    prefix7_ref = prefix7[prefix7['case_len']>=19]
    prefix7_ref = prefix7_ref.reset_index()
    prefix7_ref = prefix7_ref.drop("index",1)
    prefix8_ref = prefix8[prefix8['case_len']>=19]
    prefix8_ref = prefix8_ref.reset_index()
    prefix8_ref = prefix8_ref.drop("index",1)
    prefix9_ref = prefix9[prefix9['case_len']>=19]
    prefix9_ref = prefix9_ref.reset_index()
    prefix9_ref = prefix9_ref.drop("index",1)
    prefix10_ref = prefix10[prefix10['case_len']>=19]
    prefix10_ref = prefix10_ref.reset_index()
    prefix10_ref = prefix10_ref.drop("index",1)
    prefix11_ref = prefix11[prefix11['case_len']>=19]
    prefix11_ref = prefix11_ref.reset_index()
    prefix11_ref = prefix11_ref.drop("index",1)
    prefix12_ref = prefix12[prefix12['case_len']>=19]
    prefix12_ref = prefix12_ref.reset_index()
    prefix12_ref = prefix12_ref.drop("index",1)
    prefix13_ref = prefix13[prefix13['case_len']>=19]
    prefix13_ref = prefix13_ref.reset_index()
    prefix13_ref = prefix13_ref.drop("index",1)
    prefix14_ref = prefix14[prefix14['case_len']>=19]
    prefix14_ref = prefix14_ref.reset_index()
    prefix14_ref = prefix14_ref.drop("index",1)
    prefix15_ref = prefix15[prefix15['case_len']>=19]
    prefix15_ref = prefix15_ref.reset_index()
    prefix15_ref = prefix15_ref.drop("index",1)
    prefix16_ref = prefix16[prefix16['case_len']>=19]
    prefix16_ref = prefix16_ref.reset_index()
    prefix16_ref = prefix16_ref.drop("index",1)
    prefix17_ref = prefix17[prefix17['case_len']>=19]
    prefix17_ref = prefix17_ref.reset_index()
    prefix17_ref = prefix17_ref.drop("index",1)
    prefix18_ref = prefix18[prefix18['case_len']>=19]
    prefix18_ref = prefix18_ref.reset_index()
    prefix18_ref = prefix18_ref.drop("index",1)
    prefix19 = data_dummy_prefix[data_dummy_prefix['prefix']==19]
    prefix19 = prefix19.reset_index()
    prefix19 = prefix19.drop("index",1)
    print(len(prefix19))
    print("Processing columns with prefix 19...")
    def max_prefix_40_step19(iter):
        for col in prefix19:
            if col[len(col)-10:len(col)] == "prefixlen1":
                prefix19[col] = prefix1_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen2":
                prefix19[col] = prefix2_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen3":
                prefix19[col] = prefix3_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen4":
                prefix19[col] = prefix4_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen5":
                prefix19[col] = prefix5_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen6":
                prefix19[col] = prefix6_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen7":
                prefix19[col] = prefix7_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen8":
                prefix19[col] = prefix8_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen9":
                prefix19[col] = prefix9_ref[col[0:len(col)-11]]
            if col[len(col)-11:len(col)] == "prefixlen10":
                prefix19[col] = prefix10_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen11":
                prefix19[col] = prefix11_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen12":
                prefix19[col] = prefix12_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen13":
                prefix19[col] = prefix13_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen14":
                prefix19[col] = prefix14_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen15":
                prefix19[col] = prefix15_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen16":
                prefix19[col] = prefix16_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen17":
                prefix19[col] = prefix17_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen18":
                prefix19[col] = prefix18_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen19":
                prefix19[col] = prefix19[col[0:len(col)-12]]
        return prefix19
    if __name__ == '__main__':
        pool = multiprocessing.Pool(processes=num_cpu_cores)
        prefix19 = pool.map(max_prefix_40_step19, [0])[0]
        pool.close()
        pool.join()

    prefix1_ref = prefix1[prefix1['case_len']>=20]
    prefix1_ref = prefix1_ref.reset_index()
    prefix1_ref = prefix1_ref.drop("index",1)
    prefix2_ref = prefix2[prefix2['case_len']>=20]
    prefix2_ref = prefix2_ref.reset_index()
    prefix2_ref = prefix2_ref.drop("index",1)
    prefix3_ref = prefix3[prefix3['case_len']>=20]
    prefix3_ref = prefix3_ref.reset_index()
    prefix3_ref = prefix3_ref.drop("index",1)
    prefix4_ref = prefix4[prefix4['case_len']>=20]
    prefix4_ref = prefix4_ref.reset_index()
    prefix4_ref = prefix4_ref.drop("index",1)
    prefix5_ref = prefix5[prefix5['case_len']>=20]
    prefix5_ref = prefix5_ref.reset_index()
    prefix5_ref = prefix5_ref.drop("index",1)
    prefix6_ref = prefix6[prefix6['case_len']>=20]
    prefix6_ref = prefix6_ref.reset_index()
    prefix6_ref = prefix6_ref.drop("index",1)
    prefix7_ref = prefix7[prefix7['case_len']>=20]
    prefix7_ref = prefix7_ref.reset_index()
    prefix7_ref = prefix7_ref.drop("index",1)
    prefix8_ref = prefix8[prefix8['case_len']>=20]
    prefix8_ref = prefix8_ref.reset_index()
    prefix8_ref = prefix8_ref.drop("index",1)
    prefix9_ref = prefix9[prefix9['case_len']>=20]
    prefix9_ref = prefix9_ref.reset_index()
    prefix9_ref = prefix9_ref.drop("index",1)
    prefix10_ref = prefix10[prefix10['case_len']>=20]
    prefix10_ref = prefix10_ref.reset_index()
    prefix10_ref = prefix10_ref.drop("index",1)
    prefix11_ref = prefix11[prefix11['case_len']>=20]
    prefix11_ref = prefix11_ref.reset_index()
    prefix11_ref = prefix11_ref.drop("index",1)
    prefix12_ref = prefix12[prefix12['case_len']>=20]
    prefix12_ref = prefix12_ref.reset_index()
    prefix12_ref = prefix12_ref.drop("index",1)
    prefix13_ref = prefix13[prefix13['case_len']>=20]
    prefix13_ref = prefix13_ref.reset_index()
    prefix13_ref = prefix13_ref.drop("index",1)
    prefix14_ref = prefix14[prefix14['case_len']>=20]
    prefix14_ref = prefix14_ref.reset_index()
    prefix14_ref = prefix14_ref.drop("index",1)
    prefix15_ref = prefix15[prefix15['case_len']>=20]
    prefix15_ref = prefix15_ref.reset_index()
    prefix15_ref = prefix15_ref.drop("index",1)
    prefix16_ref = prefix16[prefix16['case_len']>=20]
    prefix16_ref = prefix16_ref.reset_index()
    prefix16_ref = prefix16_ref.drop("index",1)
    prefix17_ref = prefix17[prefix17['case_len']>=20]
    prefix17_ref = prefix17_ref.reset_index()
    prefix17_ref = prefix17_ref.drop("index",1)
    prefix18_ref = prefix18[prefix18['case_len']>=20]
    prefix18_ref = prefix18_ref.reset_index()
    prefix18_ref = prefix18_ref.drop("index",1)
    prefix19_ref = prefix19[prefix19['case_len']>=20]
    prefix19_ref = prefix19_ref.reset_index()
    prefix19_ref = prefix19_ref.drop("index",1)
    prefix20 = data_dummy_prefix[data_dummy_prefix['prefix']==20]
    prefix20 = prefix20.reset_index()
    prefix20 = prefix20.drop("index",1)
    print(len(prefix20))
    print("Processing columns with prefix 20...")
    def max_prefix_40_step20(iter):
        for col in prefix20:
            if col[len(col)-10:len(col)] == "prefixlen1":
                prefix20[col] = prefix1_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen2":
                prefix20[col] = prefix2_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen3":
                prefix20[col] = prefix3_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen4":
                prefix20[col] = prefix4_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen5":
                prefix20[col] = prefix5_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen6":
                prefix20[col] = prefix6_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen7":
                prefix20[col] = prefix7_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen8":
                prefix20[col] = prefix8_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen9":
                prefix20[col] = prefix9_ref[col[0:len(col)-11]]
            if col[len(col)-11:len(col)] == "prefixlen10":
                prefix20[col] = prefix10_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen11":
                prefix20[col] = prefix11_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen12":
                prefix20[col] = prefix12_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen13":
                prefix20[col] = prefix13_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen14":
                prefix20[col] = prefix14_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen15":
                prefix20[col] = prefix15_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen16":
                prefix20[col] = prefix16_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen17":
                prefix20[col] = prefix17_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen18":
                prefix20[col] = prefix18_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen19":
                prefix20[col] = prefix19_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen20":
                prefix20[col] = prefix20[col[0:len(col)-12]]
        return prefix20
    if __name__ == '__main__':
        pool = multiprocessing.Pool(processes=num_cpu_cores)
        prefix20 = pool.map(max_prefix_40_step20, [0])[0]
        pool.close()
        pool.join()

    prefix1_ref = prefix1[prefix1['case_len']>=21]
    prefix1_ref = prefix1_ref.reset_index()
    prefix1_ref = prefix1_ref.drop("index",1)
    prefix2_ref = prefix2[prefix2['case_len']>=21]
    prefix2_ref = prefix2_ref.reset_index()
    prefix2_ref = prefix2_ref.drop("index",1)
    prefix3_ref = prefix3[prefix3['case_len']>=21]
    prefix3_ref = prefix3_ref.reset_index()
    prefix3_ref = prefix3_ref.drop("index",1)
    prefix4_ref = prefix4[prefix4['case_len']>=21]
    prefix4_ref = prefix4_ref.reset_index()
    prefix4_ref = prefix4_ref.drop("index",1)
    prefix5_ref = prefix5[prefix5['case_len']>=21]
    prefix5_ref = prefix5_ref.reset_index()
    prefix5_ref = prefix5_ref.drop("index",1)
    prefix6_ref = prefix6[prefix6['case_len']>=21]
    prefix6_ref = prefix6_ref.reset_index()
    prefix6_ref = prefix6_ref.drop("index",1)
    prefix7_ref = prefix7[prefix7['case_len']>=21]
    prefix7_ref = prefix7_ref.reset_index()
    prefix7_ref = prefix7_ref.drop("index",1)
    prefix8_ref = prefix8[prefix8['case_len']>=21]
    prefix8_ref = prefix8_ref.reset_index()
    prefix8_ref = prefix8_ref.drop("index",1)
    prefix9_ref = prefix9[prefix9['case_len']>=21]
    prefix9_ref = prefix9_ref.reset_index()
    prefix9_ref = prefix9_ref.drop("index",1)
    prefix10_ref = prefix10[prefix10['case_len']>=21]
    prefix10_ref = prefix10_ref.reset_index()
    prefix10_ref = prefix10_ref.drop("index",1)
    prefix11_ref = prefix11[prefix11['case_len']>=21]
    prefix11_ref = prefix11_ref.reset_index()
    prefix11_ref = prefix11_ref.drop("index",1)
    prefix12_ref = prefix12[prefix12['case_len']>=21]
    prefix12_ref = prefix12_ref.reset_index()
    prefix12_ref = prefix12_ref.drop("index",1)
    prefix13_ref = prefix13[prefix13['case_len']>=21]
    prefix13_ref = prefix13_ref.reset_index()
    prefix13_ref = prefix13_ref.drop("index",1)
    prefix14_ref = prefix14[prefix14['case_len']>=21]
    prefix14_ref = prefix14_ref.reset_index()
    prefix14_ref = prefix14_ref.drop("index",1)
    prefix15_ref = prefix15[prefix15['case_len']>=21]
    prefix15_ref = prefix15_ref.reset_index()
    prefix15_ref = prefix15_ref.drop("index",1)
    prefix16_ref = prefix16[prefix16['case_len']>=21]
    prefix16_ref = prefix16_ref.reset_index()
    prefix16_ref = prefix16_ref.drop("index",1)
    prefix17_ref = prefix17[prefix17['case_len']>=21]
    prefix17_ref = prefix17_ref.reset_index()
    prefix17_ref = prefix17_ref.drop("index",1)
    prefix18_ref = prefix18[prefix18['case_len']>=21]
    prefix18_ref = prefix18_ref.reset_index()
    prefix18_ref = prefix18_ref.drop("index",1)
    prefix19_ref = prefix19[prefix19['case_len']>=21]
    prefix19_ref = prefix19_ref.reset_index()
    prefix19_ref = prefix19_ref.drop("index",1)
    prefix20_ref = prefix20[prefix20['case_len']>=21]
    prefix20_ref = prefix20_ref.reset_index()
    prefix20_ref = prefix20_ref.drop("index",1)
    prefix21 = data_dummy_prefix[data_dummy_prefix['prefix']==21]
    prefix21 = prefix21.reset_index()
    prefix21 = prefix21.drop("index",1)
    print(len(prefix21))
    print("Processing columns with prefix 21...")
    def max_prefix_40_step21(iter):
        for col in prefix21:
            if col[len(col)-10:len(col)] == "prefixlen1":
                prefix21[col] = prefix1_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen2":
                prefix21[col] = prefix2_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen3":
                prefix21[col] = prefix3_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen4":
                prefix21[col] = prefix4_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen5":
                prefix21[col] = prefix5_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen6":
                prefix21[col] = prefix6_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen7":
                prefix21[col] = prefix7_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen8":
                prefix21[col] = prefix8_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen9":
                prefix21[col] = prefix9_ref[col[0:len(col)-11]]
            if col[len(col)-11:len(col)] == "prefixlen10":
                prefix21[col] = prefix10_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen11":
                prefix21[col] = prefix11_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen12":
                prefix21[col] = prefix12_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen13":
                prefix21[col] = prefix13_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen14":
                prefix21[col] = prefix14_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen15":
                prefix21[col] = prefix15_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen16":
                prefix21[col] = prefix16_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen17":
                prefix21[col] = prefix17_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen18":
                prefix21[col] = prefix18_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen19":
                prefix21[col] = prefix19_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen20":
                prefix21[col] = prefix20_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen21":
                prefix21[col] = prefix21[col[0:len(col)-12]]
        return prefix21
    if __name__ == '__main__':
        pool = multiprocessing.Pool(processes=num_cpu_cores)
        prefix21 = pool.map(max_prefix_40_step21, [0])[0]
        pool.close()
        pool.join()

    prefix1_ref = prefix1[prefix1['case_len']>=22]
    prefix1_ref = prefix1_ref.reset_index()
    prefix1_ref = prefix1_ref.drop("index",1)
    prefix2_ref = prefix2[prefix2['case_len']>=22]
    prefix2_ref = prefix2_ref.reset_index()
    prefix2_ref = prefix2_ref.drop("index",1)
    prefix3_ref = prefix3[prefix3['case_len']>=22]
    prefix3_ref = prefix3_ref.reset_index()
    prefix3_ref = prefix3_ref.drop("index",1)
    prefix4_ref = prefix4[prefix4['case_len']>=22]
    prefix4_ref = prefix4_ref.reset_index()
    prefix4_ref = prefix4_ref.drop("index",1)
    prefix5_ref = prefix5[prefix5['case_len']>=22]
    prefix5_ref = prefix5_ref.reset_index()
    prefix5_ref = prefix5_ref.drop("index",1)
    prefix6_ref = prefix6[prefix6['case_len']>=22]
    prefix6_ref = prefix6_ref.reset_index()
    prefix6_ref = prefix6_ref.drop("index",1)
    prefix7_ref = prefix7[prefix7['case_len']>=22]
    prefix7_ref = prefix7_ref.reset_index()
    prefix7_ref = prefix7_ref.drop("index",1)
    prefix8_ref = prefix8[prefix8['case_len']>=22]
    prefix8_ref = prefix8_ref.reset_index()
    prefix8_ref = prefix8_ref.drop("index",1)
    prefix9_ref = prefix9[prefix9['case_len']>=22]
    prefix9_ref = prefix9_ref.reset_index()
    prefix9_ref = prefix9_ref.drop("index",1)
    prefix10_ref = prefix10[prefix10['case_len']>=22]
    prefix10_ref = prefix10_ref.reset_index()
    prefix10_ref = prefix10_ref.drop("index",1)
    prefix11_ref = prefix11[prefix11['case_len']>=22]
    prefix11_ref = prefix11_ref.reset_index()
    prefix11_ref = prefix11_ref.drop("index",1)
    prefix12_ref = prefix12[prefix12['case_len']>=22]
    prefix12_ref = prefix12_ref.reset_index()
    prefix12_ref = prefix12_ref.drop("index",1)
    prefix13_ref = prefix13[prefix13['case_len']>=22]
    prefix13_ref = prefix13_ref.reset_index()
    prefix13_ref = prefix13_ref.drop("index",1)
    prefix14_ref = prefix14[prefix14['case_len']>=22]
    prefix14_ref = prefix14_ref.reset_index()
    prefix14_ref = prefix14_ref.drop("index",1)
    prefix15_ref = prefix15[prefix15['case_len']>=22]
    prefix15_ref = prefix15_ref.reset_index()
    prefix15_ref = prefix15_ref.drop("index",1)
    prefix16_ref = prefix16[prefix16['case_len']>=22]
    prefix16_ref = prefix16_ref.reset_index()
    prefix16_ref = prefix16_ref.drop("index",1)
    prefix17_ref = prefix17[prefix17['case_len']>=22]
    prefix17_ref = prefix17_ref.reset_index()
    prefix17_ref = prefix17_ref.drop("index",1)
    prefix18_ref = prefix18[prefix18['case_len']>=22]
    prefix18_ref = prefix18_ref.reset_index()
    prefix18_ref = prefix18_ref.drop("index",1)
    prefix19_ref = prefix19[prefix19['case_len']>=22]
    prefix19_ref = prefix19_ref.reset_index()
    prefix19_ref = prefix19_ref.drop("index",1)
    prefix20_ref = prefix20[prefix20['case_len']>=22]
    prefix20_ref = prefix20_ref.reset_index()
    prefix20_ref = prefix20_ref.drop("index",1)
    prefix21_ref = prefix21[prefix21['case_len']>=22]
    prefix21_ref = prefix21_ref.reset_index()
    prefix21_ref = prefix21_ref.drop("index",1)
    prefix22 = data_dummy_prefix[data_dummy_prefix['prefix']==22]
    prefix22 = prefix22.reset_index()
    prefix22 = prefix22.drop("index",1)
    print(len(prefix22))
    print("Processing columns with prefix 22...")
    def max_prefix_40_step22(iter):
        for col in prefix22:
            if col[len(col)-10:len(col)] == "prefixlen1":
                prefix22[col] = prefix1_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen2":
                prefix22[col] = prefix2_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen3":
                prefix22[col] = prefix3_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen4":
                prefix22[col] = prefix4_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen5":
                prefix22[col] = prefix5_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen6":
                prefix22[col] = prefix6_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen7":
                prefix22[col] = prefix7_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen8":
                prefix22[col] = prefix8_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen9":
                prefix22[col] = prefix9_ref[col[0:len(col)-11]]
            if col[len(col)-11:len(col)] == "prefixlen10":
                prefix22[col] = prefix10_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen11":
                prefix22[col] = prefix11_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen12":
                prefix22[col] = prefix12_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen13":
                prefix22[col] = prefix13_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen14":
                prefix22[col] = prefix14_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen15":
                prefix22[col] = prefix15_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen16":
                prefix22[col] = prefix16_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen17":
                prefix22[col] = prefix17_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen18":
                prefix22[col] = prefix18_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen19":
                prefix22[col] = prefix19_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen20":
                prefix22[col] = prefix20_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen21":
                prefix22[col] = prefix21_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen22":
                prefix22[col] = prefix22[col[0:len(col)-12]]
        return prefix22
    if __name__ == '__main__':
        pool = multiprocessing.Pool(processes=num_cpu_cores)
        prefix22 = pool.map(max_prefix_40_step22, [0])[0]
        pool.close()
        pool.join()

    prefix1_ref = prefix1[prefix1['case_len']>=23]
    prefix1_ref = prefix1_ref.reset_index()
    prefix1_ref = prefix1_ref.drop("index",1)
    prefix2_ref = prefix2[prefix2['case_len']>=23]
    prefix2_ref = prefix2_ref.reset_index()
    prefix2_ref = prefix2_ref.drop("index",1)
    prefix3_ref = prefix3[prefix3['case_len']>=23]
    prefix3_ref = prefix3_ref.reset_index()
    prefix3_ref = prefix3_ref.drop("index",1)
    prefix4_ref = prefix4[prefix4['case_len']>=23]
    prefix4_ref = prefix4_ref.reset_index()
    prefix4_ref = prefix4_ref.drop("index",1)
    prefix5_ref = prefix5[prefix5['case_len']>=23]
    prefix5_ref = prefix5_ref.reset_index()
    prefix5_ref = prefix5_ref.drop("index",1)
    prefix6_ref = prefix6[prefix6['case_len']>=23]
    prefix6_ref = prefix6_ref.reset_index()
    prefix6_ref = prefix6_ref.drop("index",1)
    prefix7_ref = prefix7[prefix7['case_len']>=23]
    prefix7_ref = prefix7_ref.reset_index()
    prefix7_ref = prefix7_ref.drop("index",1)
    prefix8_ref = prefix8[prefix8['case_len']>=23]
    prefix8_ref = prefix8_ref.reset_index()
    prefix8_ref = prefix8_ref.drop("index",1)
    prefix9_ref = prefix9[prefix9['case_len']>=23]
    prefix9_ref = prefix9_ref.reset_index()
    prefix9_ref = prefix9_ref.drop("index",1)
    prefix10_ref = prefix10[prefix10['case_len']>=23]
    prefix10_ref = prefix10_ref.reset_index()
    prefix10_ref = prefix10_ref.drop("index",1)
    prefix11_ref = prefix11[prefix11['case_len']>=23]
    prefix11_ref = prefix11_ref.reset_index()
    prefix11_ref = prefix11_ref.drop("index",1)
    prefix12_ref = prefix12[prefix12['case_len']>=23]
    prefix12_ref = prefix12_ref.reset_index()
    prefix12_ref = prefix12_ref.drop("index",1)
    prefix13_ref = prefix13[prefix13['case_len']>=23]
    prefix13_ref = prefix13_ref.reset_index()
    prefix13_ref = prefix13_ref.drop("index",1)
    prefix14_ref = prefix14[prefix14['case_len']>=23]
    prefix14_ref = prefix14_ref.reset_index()
    prefix14_ref = prefix14_ref.drop("index",1)
    prefix15_ref = prefix15[prefix15['case_len']>=23]
    prefix15_ref = prefix15_ref.reset_index()
    prefix15_ref = prefix15_ref.drop("index",1)
    prefix16_ref = prefix16[prefix16['case_len']>=23]
    prefix16_ref = prefix16_ref.reset_index()
    prefix16_ref = prefix16_ref.drop("index",1)
    prefix17_ref = prefix17[prefix17['case_len']>=23]
    prefix17_ref = prefix17_ref.reset_index()
    prefix17_ref = prefix17_ref.drop("index",1)
    prefix18_ref = prefix18[prefix18['case_len']>=23]
    prefix18_ref = prefix18_ref.reset_index()
    prefix18_ref = prefix18_ref.drop("index",1)
    prefix19_ref = prefix19[prefix19['case_len']>=23]
    prefix19_ref = prefix19_ref.reset_index()
    prefix19_ref = prefix19_ref.drop("index",1)
    prefix20_ref = prefix20[prefix20['case_len']>=23]
    prefix20_ref = prefix20_ref.reset_index()
    prefix20_ref = prefix20_ref.drop("index",1)
    prefix21_ref = prefix21[prefix21['case_len']>=23]
    prefix21_ref = prefix21_ref.reset_index()
    prefix21_ref = prefix21_ref.drop("index",1)
    prefix22_ref = prefix22[prefix22['case_len']>=23]
    prefix22_ref = prefix22_ref.reset_index()
    prefix22_ref = prefix22_ref.drop("index",1)
    prefix23 = data_dummy_prefix[data_dummy_prefix['prefix']==23]
    prefix23 = prefix23.reset_index()
    prefix23 = prefix23.drop("index",1)
    print(len(prefix23))
    print("Processing columns with prefix 23...")
    def max_prefix_40_step23(iter):
        for col in prefix23:
            if col[len(col)-10:len(col)] == "prefixlen1":
                prefix23[col] = prefix1_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen2":
                prefix23[col] = prefix2_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen3":
                prefix23[col] = prefix3_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen4":
                prefix23[col] = prefix4_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen5":
                prefix23[col] = prefix5_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen6":
                prefix23[col] = prefix6_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen7":
                prefix23[col] = prefix7_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen8":
                prefix23[col] = prefix8_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen9":
                prefix23[col] = prefix9_ref[col[0:len(col)-11]]
            if col[len(col)-11:len(col)] == "prefixlen10":
                prefix23[col] = prefix10_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen11":
                prefix23[col] = prefix11_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen12":
                prefix23[col] = prefix12_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen13":
                prefix23[col] = prefix13_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen14":
                prefix23[col] = prefix14_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen15":
                prefix23[col] = prefix15_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen16":
                prefix23[col] = prefix16_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen17":
                prefix23[col] = prefix17_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen18":
                prefix23[col] = prefix18_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen19":
                prefix23[col] = prefix19_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen20":
                prefix23[col] = prefix20_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen21":
                prefix23[col] = prefix21_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen22":
                prefix23[col] = prefix22_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen23":
                prefix23[col] = prefix23[col[0:len(col)-12]]
        return prefix23
    if __name__ == '__main__':
        pool = multiprocessing.Pool(processes=num_cpu_cores)
        prefix23 = pool.map(max_prefix_40_step23, [0])[0]
        pool.close()
        pool.join()

    prefix1_ref = prefix1[prefix1['case_len']>=24]
    prefix1_ref = prefix1_ref.reset_index()
    prefix1_ref = prefix1_ref.drop("index",1)
    prefix2_ref = prefix2[prefix2['case_len']>=24]
    prefix2_ref = prefix2_ref.reset_index()
    prefix2_ref = prefix2_ref.drop("index",1)
    prefix3_ref = prefix3[prefix3['case_len']>=24]
    prefix3_ref = prefix3_ref.reset_index()
    prefix3_ref = prefix3_ref.drop("index",1)
    prefix4_ref = prefix4[prefix4['case_len']>=24]
    prefix4_ref = prefix4_ref.reset_index()
    prefix4_ref = prefix4_ref.drop("index",1)
    prefix5_ref = prefix5[prefix5['case_len']>=24]
    prefix5_ref = prefix5_ref.reset_index()
    prefix5_ref = prefix5_ref.drop("index",1)
    prefix6_ref = prefix6[prefix6['case_len']>=24]
    prefix6_ref = prefix6_ref.reset_index()
    prefix6_ref = prefix6_ref.drop("index",1)
    prefix7_ref = prefix7[prefix7['case_len']>=24]
    prefix7_ref = prefix7_ref.reset_index()
    prefix7_ref = prefix7_ref.drop("index",1)
    prefix8_ref = prefix8[prefix8['case_len']>=24]
    prefix8_ref = prefix8_ref.reset_index()
    prefix8_ref = prefix8_ref.drop("index",1)
    prefix9_ref = prefix9[prefix9['case_len']>=24]
    prefix9_ref = prefix9_ref.reset_index()
    prefix9_ref = prefix9_ref.drop("index",1)
    prefix10_ref = prefix10[prefix10['case_len']>=24]
    prefix10_ref = prefix10_ref.reset_index()
    prefix10_ref = prefix10_ref.drop("index",1)
    prefix11_ref = prefix11[prefix11['case_len']>=24]
    prefix11_ref = prefix11_ref.reset_index()
    prefix11_ref = prefix11_ref.drop("index",1)
    prefix12_ref = prefix12[prefix12['case_len']>=24]
    prefix12_ref = prefix12_ref.reset_index()
    prefix12_ref = prefix12_ref.drop("index",1)
    prefix13_ref = prefix13[prefix13['case_len']>=24]
    prefix13_ref = prefix13_ref.reset_index()
    prefix13_ref = prefix13_ref.drop("index",1)
    prefix14_ref = prefix14[prefix14['case_len']>=24]
    prefix14_ref = prefix14_ref.reset_index()
    prefix14_ref = prefix14_ref.drop("index",1)
    prefix15_ref = prefix15[prefix15['case_len']>=24]
    prefix15_ref = prefix15_ref.reset_index()
    prefix15_ref = prefix15_ref.drop("index",1)
    prefix16_ref = prefix16[prefix16['case_len']>=24]
    prefix16_ref = prefix16_ref.reset_index()
    prefix16_ref = prefix16_ref.drop("index",1)
    prefix17_ref = prefix17[prefix17['case_len']>=24]
    prefix17_ref = prefix17_ref.reset_index()
    prefix17_ref = prefix17_ref.drop("index",1)
    prefix18_ref = prefix18[prefix18['case_len']>=24]
    prefix18_ref = prefix18_ref.reset_index()
    prefix18_ref = prefix18_ref.drop("index",1)
    prefix19_ref = prefix19[prefix19['case_len']>=24]
    prefix19_ref = prefix19_ref.reset_index()
    prefix19_ref = prefix19_ref.drop("index",1)
    prefix20_ref = prefix20[prefix20['case_len']>=24]
    prefix20_ref = prefix20_ref.reset_index()
    prefix20_ref = prefix20_ref.drop("index",1)
    prefix21_ref = prefix21[prefix21['case_len']>=24]
    prefix21_ref = prefix21_ref.reset_index()
    prefix21_ref = prefix21_ref.drop("index",1)
    prefix22_ref = prefix22[prefix22['case_len']>=24]
    prefix22_ref = prefix22_ref.reset_index()
    prefix22_ref = prefix22_ref.drop("index",1)
    prefix23_ref = prefix23[prefix23['case_len']>=24]
    prefix23_ref = prefix23_ref.reset_index()
    prefix23_ref = prefix23_ref.drop("index",1)
    prefix24 = data_dummy_prefix[data_dummy_prefix['prefix']==24]
    prefix24 = prefix24.reset_index()
    prefix24 = prefix24.drop("index",1)
    print(len(prefix24))
    print("Processing columns with prefix 24...")
    def max_prefix_40_step24(iter):
        for col in prefix24:
            if col[len(col)-10:len(col)] == "prefixlen1":
                prefix24[col] = prefix1_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen2":
                prefix24[col] = prefix2_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen3":
                prefix24[col] = prefix3_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen4":
                prefix24[col] = prefix4_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen5":
                prefix24[col] = prefix5_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen6":
                prefix24[col] = prefix6_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen7":
                prefix24[col] = prefix7_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen8":
                prefix24[col] = prefix8_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen9":
                prefix24[col] = prefix9_ref[col[0:len(col)-11]]
            if col[len(col)-11:len(col)] == "prefixlen10":
                prefix24[col] = prefix10_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen11":
                prefix24[col] = prefix11_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen12":
                prefix24[col] = prefix12_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen13":
                prefix24[col] = prefix13_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen14":
                prefix24[col] = prefix14_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen15":
                prefix24[col] = prefix15_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen16":
                prefix24[col] = prefix16_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen17":
                prefix24[col] = prefix17_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen18":
                prefix24[col] = prefix18_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen19":
                prefix24[col] = prefix19_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen20":
                prefix24[col] = prefix20_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen21":
                prefix24[col] = prefix21_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen22":
                prefix24[col] = prefix22_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen23":
                prefix24[col] = prefix23_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen24":
                prefix24[col] = prefix24[col[0:len(col)-12]]
        return prefix24
    if __name__ == '__main__':
        pool = multiprocessing.Pool(processes=num_cpu_cores)
        prefix24 = pool.map(max_prefix_40_step24, [0])[0]
        pool.close()
        pool.join()

    prefix1_ref = prefix1[prefix1['case_len']>=25]
    prefix1_ref = prefix1_ref.reset_index()
    prefix1_ref = prefix1_ref.drop("index",1)
    prefix2_ref = prefix2[prefix2['case_len']>=25]
    prefix2_ref = prefix2_ref.reset_index()
    prefix2_ref = prefix2_ref.drop("index",1)
    prefix3_ref = prefix3[prefix3['case_len']>=25]
    prefix3_ref = prefix3_ref.reset_index()
    prefix3_ref = prefix3_ref.drop("index",1)
    prefix4_ref = prefix4[prefix4['case_len']>=25]
    prefix4_ref = prefix4_ref.reset_index()
    prefix4_ref = prefix4_ref.drop("index",1)
    prefix5_ref = prefix5[prefix5['case_len']>=25]
    prefix5_ref = prefix5_ref.reset_index()
    prefix5_ref = prefix5_ref.drop("index",1)
    prefix6_ref = prefix6[prefix6['case_len']>=25]
    prefix6_ref = prefix6_ref.reset_index()
    prefix6_ref = prefix6_ref.drop("index",1)
    prefix7_ref = prefix7[prefix7['case_len']>=25]
    prefix7_ref = prefix7_ref.reset_index()
    prefix7_ref = prefix7_ref.drop("index",1)
    prefix8_ref = prefix8[prefix8['case_len']>=25]
    prefix8_ref = prefix8_ref.reset_index()
    prefix8_ref = prefix8_ref.drop("index",1)
    prefix9_ref = prefix9[prefix9['case_len']>=25]
    prefix9_ref = prefix9_ref.reset_index()
    prefix9_ref = prefix9_ref.drop("index",1)
    prefix10_ref = prefix10[prefix10['case_len']>=25]
    prefix10_ref = prefix10_ref.reset_index()
    prefix10_ref = prefix10_ref.drop("index",1)
    prefix11_ref = prefix11[prefix11['case_len']>=25]
    prefix11_ref = prefix11_ref.reset_index()
    prefix11_ref = prefix11_ref.drop("index",1)
    prefix12_ref = prefix12[prefix12['case_len']>=25]
    prefix12_ref = prefix12_ref.reset_index()
    prefix12_ref = prefix12_ref.drop("index",1)
    prefix13_ref = prefix13[prefix13['case_len']>=25]
    prefix13_ref = prefix13_ref.reset_index()
    prefix13_ref = prefix13_ref.drop("index",1)
    prefix14_ref = prefix14[prefix14['case_len']>=25]
    prefix14_ref = prefix14_ref.reset_index()
    prefix14_ref = prefix14_ref.drop("index",1)
    prefix15_ref = prefix15[prefix15['case_len']>=25]
    prefix15_ref = prefix15_ref.reset_index()
    prefix15_ref = prefix15_ref.drop("index",1)
    prefix16_ref = prefix16[prefix16['case_len']>=25]
    prefix16_ref = prefix16_ref.reset_index()
    prefix16_ref = prefix16_ref.drop("index",1)
    prefix17_ref = prefix17[prefix17['case_len']>=25]
    prefix17_ref = prefix17_ref.reset_index()
    prefix17_ref = prefix17_ref.drop("index",1)
    prefix18_ref = prefix18[prefix18['case_len']>=25]
    prefix18_ref = prefix18_ref.reset_index()
    prefix18_ref = prefix18_ref.drop("index",1)
    prefix19_ref = prefix19[prefix19['case_len']>=25]
    prefix19_ref = prefix19_ref.reset_index()
    prefix19_ref = prefix19_ref.drop("index",1)
    prefix20_ref = prefix20[prefix20['case_len']>=25]
    prefix20_ref = prefix20_ref.reset_index()
    prefix20_ref = prefix20_ref.drop("index",1)
    prefix21_ref = prefix21[prefix21['case_len']>=25]
    prefix21_ref = prefix21_ref.reset_index()
    prefix21_ref = prefix21_ref.drop("index",1)
    prefix22_ref = prefix22[prefix22['case_len']>=25]
    prefix22_ref = prefix22_ref.reset_index()
    prefix22_ref = prefix22_ref.drop("index",1)
    prefix23_ref = prefix23[prefix23['case_len']>=25]
    prefix23_ref = prefix23_ref.reset_index()
    prefix23_ref = prefix23_ref.drop("index",1)
    prefix24_ref = prefix24[prefix24['case_len']>=25]
    prefix24_ref = prefix24_ref.reset_index()
    prefix24_ref = prefix24_ref.drop("index",1)
    prefix25 = data_dummy_prefix[data_dummy_prefix['prefix']==25]
    prefix25 = prefix25.reset_index()
    prefix25 = prefix25.drop("index",1)
    print(len(prefix25))
    print("Processing columns with prefix 25...")
    def max_prefix_40_step25(iter):
        for col in prefix25:
            if col[len(col)-10:len(col)] == "prefixlen1":
                prefix25[col] = prefix1_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen2":
                prefix25[col] = prefix2_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen3":
                prefix25[col] = prefix3_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen4":
                prefix25[col] = prefix4_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen5":
                prefix25[col] = prefix5_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen6":
                prefix25[col] = prefix6_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen7":
                prefix25[col] = prefix7_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen8":
                prefix25[col] = prefix8_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen9":
                prefix25[col] = prefix9_ref[col[0:len(col)-11]]
            if col[len(col)-11:len(col)] == "prefixlen10":
                prefix25[col] = prefix10_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen11":
                prefix25[col] = prefix11_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen12":
                prefix25[col] = prefix12_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen13":
                prefix25[col] = prefix13_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen14":
                prefix25[col] = prefix14_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen15":
                prefix25[col] = prefix15_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen16":
                prefix25[col] = prefix16_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen17":
                prefix25[col] = prefix17_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen18":
                prefix25[col] = prefix18_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen19":
                prefix25[col] = prefix19_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen20":
                prefix25[col] = prefix20_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen21":
                prefix25[col] = prefix21_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen22":
                prefix25[col] = prefix22_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen23":
                prefix25[col] = prefix23_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen24":
                prefix25[col] = prefix24_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen25":
                prefix25[col] = prefix25[col[0:len(col)-12]]
        return prefix25
    if __name__ == '__main__':
        pool = multiprocessing.Pool(processes=num_cpu_cores)
        prefix25 = pool.map(max_prefix_40_step25, [0])[0]
        pool.close()
        pool.join()

    prefix1_ref = prefix1[prefix1['case_len']>=26]
    prefix1_ref = prefix1_ref.reset_index()
    prefix1_ref = prefix1_ref.drop("index",1)
    prefix2_ref = prefix2[prefix2['case_len']>=26]
    prefix2_ref = prefix2_ref.reset_index()
    prefix2_ref = prefix2_ref.drop("index",1)
    prefix3_ref = prefix3[prefix3['case_len']>=26]
    prefix3_ref = prefix3_ref.reset_index()
    prefix3_ref = prefix3_ref.drop("index",1)
    prefix4_ref = prefix4[prefix4['case_len']>=26]
    prefix4_ref = prefix4_ref.reset_index()
    prefix4_ref = prefix4_ref.drop("index",1)
    prefix5_ref = prefix5[prefix5['case_len']>=26]
    prefix5_ref = prefix5_ref.reset_index()
    prefix5_ref = prefix5_ref.drop("index",1)
    prefix6_ref = prefix6[prefix6['case_len']>=26]
    prefix6_ref = prefix6_ref.reset_index()
    prefix6_ref = prefix6_ref.drop("index",1)
    prefix7_ref = prefix7[prefix7['case_len']>=26]
    prefix7_ref = prefix7_ref.reset_index()
    prefix7_ref = prefix7_ref.drop("index",1)
    prefix8_ref = prefix8[prefix8['case_len']>=26]
    prefix8_ref = prefix8_ref.reset_index()
    prefix8_ref = prefix8_ref.drop("index",1)
    prefix9_ref = prefix9[prefix9['case_len']>=26]
    prefix9_ref = prefix9_ref.reset_index()
    prefix9_ref = prefix9_ref.drop("index",1)
    prefix10_ref = prefix10[prefix10['case_len']>=26]
    prefix10_ref = prefix10_ref.reset_index()
    prefix10_ref = prefix10_ref.drop("index",1)
    prefix11_ref = prefix11[prefix11['case_len']>=26]
    prefix11_ref = prefix11_ref.reset_index()
    prefix11_ref = prefix11_ref.drop("index",1)
    prefix12_ref = prefix12[prefix12['case_len']>=26]
    prefix12_ref = prefix12_ref.reset_index()
    prefix12_ref = prefix12_ref.drop("index",1)
    prefix13_ref = prefix13[prefix13['case_len']>=26]
    prefix13_ref = prefix13_ref.reset_index()
    prefix13_ref = prefix13_ref.drop("index",1)
    prefix14_ref = prefix14[prefix14['case_len']>=26]
    prefix14_ref = prefix14_ref.reset_index()
    prefix14_ref = prefix14_ref.drop("index",1)
    prefix15_ref = prefix15[prefix15['case_len']>=26]
    prefix15_ref = prefix15_ref.reset_index()
    prefix15_ref = prefix15_ref.drop("index",1)
    prefix16_ref = prefix16[prefix16['case_len']>=26]
    prefix16_ref = prefix16_ref.reset_index()
    prefix16_ref = prefix16_ref.drop("index",1)
    prefix17_ref = prefix17[prefix17['case_len']>=26]
    prefix17_ref = prefix17_ref.reset_index()
    prefix17_ref = prefix17_ref.drop("index",1)
    prefix18_ref = prefix18[prefix18['case_len']>=26]
    prefix18_ref = prefix18_ref.reset_index()
    prefix18_ref = prefix18_ref.drop("index",1)
    prefix19_ref = prefix19[prefix19['case_len']>=26]
    prefix19_ref = prefix19_ref.reset_index()
    prefix19_ref = prefix19_ref.drop("index",1)
    prefix20_ref = prefix20[prefix20['case_len']>=26]
    prefix20_ref = prefix20_ref.reset_index()
    prefix20_ref = prefix20_ref.drop("index",1)
    prefix21_ref = prefix21[prefix21['case_len']>=26]
    prefix21_ref = prefix21_ref.reset_index()
    prefix21_ref = prefix21_ref.drop("index",1)
    prefix22_ref = prefix22[prefix22['case_len']>=26]
    prefix22_ref = prefix22_ref.reset_index()
    prefix22_ref = prefix22_ref.drop("index",1)
    prefix23_ref = prefix23[prefix23['case_len']>=26]
    prefix23_ref = prefix23_ref.reset_index()
    prefix23_ref = prefix23_ref.drop("index",1)
    prefix24_ref = prefix24[prefix24['case_len']>=26]
    prefix24_ref = prefix24_ref.reset_index()
    prefix24_ref = prefix24_ref.drop("index",1)
    prefix25_ref = prefix25[prefix25['case_len']>=26]
    prefix25_ref = prefix25_ref.reset_index()
    prefix25_ref = prefix25_ref.drop("index",1)
    prefix26 = data_dummy_prefix[data_dummy_prefix['prefix']==26]
    prefix26 = prefix26.reset_index()
    prefix26 = prefix26.drop("index",1)
    print(len(prefix26))
    print("Processing columns with prefix 26...")
    def max_prefix_40_step26(iter):
        for col in prefix26:
            if col[len(col)-10:len(col)] == "prefixlen1":
                prefix26[col] = prefix1_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen2":
                prefix26[col] = prefix2_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen3":
                prefix26[col] = prefix3_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen4":
                prefix26[col] = prefix4_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen5":
                prefix26[col] = prefix5_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen6":
                prefix26[col] = prefix6_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen7":
                prefix26[col] = prefix7_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen8":
                prefix26[col] = prefix8_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen9":
                prefix26[col] = prefix9_ref[col[0:len(col)-11]]
            if col[len(col)-11:len(col)] == "prefixlen10":
                prefix26[col] = prefix10_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen11":
                prefix26[col] = prefix11_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen12":
                prefix26[col] = prefix12_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen13":
                prefix26[col] = prefix13_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen14":
                prefix26[col] = prefix14_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen15":
                prefix26[col] = prefix15_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen16":
                prefix26[col] = prefix16_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen17":
                prefix26[col] = prefix17_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen18":
                prefix26[col] = prefix18_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen19":
                prefix26[col] = prefix19_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen20":
                prefix26[col] = prefix20_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen21":
                prefix26[col] = prefix21_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen22":
                prefix26[col] = prefix22_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen23":
                prefix26[col] = prefix23_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen24":
                prefix26[col] = prefix24_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen25":
                prefix26[col] = prefix25_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen26":
                prefix26[col] = prefix26[col[0:len(col)-12]]
        return prefix26
    if __name__ == '__main__':
        pool = multiprocessing.Pool(processes=num_cpu_cores)
        prefix26 = pool.map(max_prefix_40_step26, [0])[0]
        pool.close()
        pool.join()

    prefix1_ref = prefix1[prefix1['case_len']>=27]
    prefix1_ref = prefix1_ref.reset_index()
    prefix1_ref = prefix1_ref.drop("index",1)
    prefix2_ref = prefix2[prefix2['case_len']>=27]
    prefix2_ref = prefix2_ref.reset_index()
    prefix2_ref = prefix2_ref.drop("index",1)
    prefix3_ref = prefix3[prefix3['case_len']>=27]
    prefix3_ref = prefix3_ref.reset_index()
    prefix3_ref = prefix3_ref.drop("index",1)
    prefix4_ref = prefix4[prefix4['case_len']>=27]
    prefix4_ref = prefix4_ref.reset_index()
    prefix4_ref = prefix4_ref.drop("index",1)
    prefix5_ref = prefix5[prefix5['case_len']>=27]
    prefix5_ref = prefix5_ref.reset_index()
    prefix5_ref = prefix5_ref.drop("index",1)
    prefix6_ref = prefix6[prefix6['case_len']>=27]
    prefix6_ref = prefix6_ref.reset_index()
    prefix6_ref = prefix6_ref.drop("index",1)
    prefix7_ref = prefix7[prefix7['case_len']>=27]
    prefix7_ref = prefix7_ref.reset_index()
    prefix7_ref = prefix7_ref.drop("index",1)
    prefix8_ref = prefix8[prefix8['case_len']>=27]
    prefix8_ref = prefix8_ref.reset_index()
    prefix8_ref = prefix8_ref.drop("index",1)
    prefix9_ref = prefix9[prefix9['case_len']>=27]
    prefix9_ref = prefix9_ref.reset_index()
    prefix9_ref = prefix9_ref.drop("index",1)
    prefix10_ref = prefix10[prefix10['case_len']>=27]
    prefix10_ref = prefix10_ref.reset_index()
    prefix10_ref = prefix10_ref.drop("index",1)
    prefix11_ref = prefix11[prefix11['case_len']>=27]
    prefix11_ref = prefix11_ref.reset_index()
    prefix11_ref = prefix11_ref.drop("index",1)
    prefix12_ref = prefix12[prefix12['case_len']>=27]
    prefix12_ref = prefix12_ref.reset_index()
    prefix12_ref = prefix12_ref.drop("index",1)
    prefix13_ref = prefix13[prefix13['case_len']>=27]
    prefix13_ref = prefix13_ref.reset_index()
    prefix13_ref = prefix13_ref.drop("index",1)
    prefix14_ref = prefix14[prefix14['case_len']>=27]
    prefix14_ref = prefix14_ref.reset_index()
    prefix14_ref = prefix14_ref.drop("index",1)
    prefix15_ref = prefix15[prefix15['case_len']>=27]
    prefix15_ref = prefix15_ref.reset_index()
    prefix15_ref = prefix15_ref.drop("index",1)
    prefix16_ref = prefix16[prefix16['case_len']>=27]
    prefix16_ref = prefix16_ref.reset_index()
    prefix16_ref = prefix16_ref.drop("index",1)
    prefix17_ref = prefix17[prefix17['case_len']>=27]
    prefix17_ref = prefix17_ref.reset_index()
    prefix17_ref = prefix17_ref.drop("index",1)
    prefix18_ref = prefix18[prefix18['case_len']>=27]
    prefix18_ref = prefix18_ref.reset_index()
    prefix18_ref = prefix18_ref.drop("index",1)
    prefix19_ref = prefix19[prefix19['case_len']>=27]
    prefix19_ref = prefix19_ref.reset_index()
    prefix19_ref = prefix19_ref.drop("index",1)
    prefix20_ref = prefix20[prefix20['case_len']>=27]
    prefix20_ref = prefix20_ref.reset_index()
    prefix20_ref = prefix20_ref.drop("index",1)
    prefix21_ref = prefix21[prefix21['case_len']>=27]
    prefix21_ref = prefix21_ref.reset_index()
    prefix21_ref = prefix21_ref.drop("index",1)
    prefix22_ref = prefix22[prefix22['case_len']>=27]
    prefix22_ref = prefix22_ref.reset_index()
    prefix22_ref = prefix22_ref.drop("index",1)
    prefix23_ref = prefix23[prefix23['case_len']>=27]
    prefix23_ref = prefix23_ref.reset_index()
    prefix23_ref = prefix23_ref.drop("index",1)
    prefix24_ref = prefix24[prefix24['case_len']>=27]
    prefix24_ref = prefix24_ref.reset_index()
    prefix24_ref = prefix24_ref.drop("index",1)
    prefix25_ref = prefix25[prefix25['case_len']>=27]
    prefix25_ref = prefix25_ref.reset_index()
    prefix25_ref = prefix25_ref.drop("index",1)
    prefix26_ref = prefix26[prefix26['case_len']>=27]
    prefix26_ref = prefix26_ref.reset_index()
    prefix26_ref = prefix26_ref.drop("index",1)
    prefix27 = data_dummy_prefix[data_dummy_prefix['prefix']==27]
    prefix27 = prefix27.reset_index()
    prefix27 = prefix27.drop("index",1)
    print(len(prefix27))
    print("Processing columns with prefix 27...")
    def max_prefix_40_step27(iter):
        for col in prefix27:
            if col[len(col)-10:len(col)] == "prefixlen1":
                prefix27[col] = prefix1_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen2":
                prefix27[col] = prefix2_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen3":
                prefix27[col] = prefix3_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen4":
                prefix27[col] = prefix4_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen5":
                prefix27[col] = prefix5_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen6":
                prefix27[col] = prefix6_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen7":
                prefix27[col] = prefix7_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen8":
                prefix27[col] = prefix8_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen9":
                prefix27[col] = prefix9_ref[col[0:len(col)-11]]
            if col[len(col)-11:len(col)] == "prefixlen10":
                prefix27[col] = prefix10_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen11":
                prefix27[col] = prefix11_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen12":
                prefix27[col] = prefix12_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen13":
                prefix27[col] = prefix13_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen14":
                prefix27[col] = prefix14_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen15":
                prefix27[col] = prefix15_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen16":
                prefix27[col] = prefix16_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen17":
                prefix27[col] = prefix17_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen18":
                prefix27[col] = prefix18_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen19":
                prefix27[col] = prefix19_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen20":
                prefix27[col] = prefix20_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen21":
                prefix27[col] = prefix21_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen22":
                prefix27[col] = prefix22_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen23":
                prefix27[col] = prefix23_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen24":
                prefix27[col] = prefix24_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen25":
                prefix27[col] = prefix25_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen26":
                prefix27[col] = prefix26_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen27":
                prefix27[col] = prefix27[col[0:len(col)-12]]
        return prefix27
    if __name__ == '__main__':
        pool = multiprocessing.Pool(processes=num_cpu_cores)
        prefix27 = pool.map(max_prefix_40_step27, [0])[0]
        pool.close()
        pool.join()

    prefix1_ref = prefix1[prefix1['case_len']>=28]
    prefix1_ref = prefix1_ref.reset_index()
    prefix1_ref = prefix1_ref.drop("index",1)
    prefix2_ref = prefix2[prefix2['case_len']>=28]
    prefix2_ref = prefix2_ref.reset_index()
    prefix2_ref = prefix2_ref.drop("index",1)
    prefix3_ref = prefix3[prefix3['case_len']>=28]
    prefix3_ref = prefix3_ref.reset_index()
    prefix3_ref = prefix3_ref.drop("index",1)
    prefix4_ref = prefix4[prefix4['case_len']>=28]
    prefix4_ref = prefix4_ref.reset_index()
    prefix4_ref = prefix4_ref.drop("index",1)
    prefix5_ref = prefix5[prefix5['case_len']>=28]
    prefix5_ref = prefix5_ref.reset_index()
    prefix5_ref = prefix5_ref.drop("index",1)
    prefix6_ref = prefix6[prefix6['case_len']>=28]
    prefix6_ref = prefix6_ref.reset_index()
    prefix6_ref = prefix6_ref.drop("index",1)
    prefix7_ref = prefix7[prefix7['case_len']>=28]
    prefix7_ref = prefix7_ref.reset_index()
    prefix7_ref = prefix7_ref.drop("index",1)
    prefix8_ref = prefix8[prefix8['case_len']>=28]
    prefix8_ref = prefix8_ref.reset_index()
    prefix8_ref = prefix8_ref.drop("index",1)
    prefix9_ref = prefix9[prefix9['case_len']>=28]
    prefix9_ref = prefix9_ref.reset_index()
    prefix9_ref = prefix9_ref.drop("index",1)
    prefix10_ref = prefix10[prefix10['case_len']>=28]
    prefix10_ref = prefix10_ref.reset_index()
    prefix10_ref = prefix10_ref.drop("index",1)
    prefix11_ref = prefix11[prefix11['case_len']>=28]
    prefix11_ref = prefix11_ref.reset_index()
    prefix11_ref = prefix11_ref.drop("index",1)
    prefix12_ref = prefix12[prefix12['case_len']>=28]
    prefix12_ref = prefix12_ref.reset_index()
    prefix12_ref = prefix12_ref.drop("index",1)
    prefix13_ref = prefix13[prefix13['case_len']>=28]
    prefix13_ref = prefix13_ref.reset_index()
    prefix13_ref = prefix13_ref.drop("index",1)
    prefix14_ref = prefix14[prefix14['case_len']>=28]
    prefix14_ref = prefix14_ref.reset_index()
    prefix14_ref = prefix14_ref.drop("index",1)
    prefix15_ref = prefix15[prefix15['case_len']>=28]
    prefix15_ref = prefix15_ref.reset_index()
    prefix15_ref = prefix15_ref.drop("index",1)
    prefix16_ref = prefix16[prefix16['case_len']>=28]
    prefix16_ref = prefix16_ref.reset_index()
    prefix16_ref = prefix16_ref.drop("index",1)
    prefix17_ref = prefix17[prefix17['case_len']>=28]
    prefix17_ref = prefix17_ref.reset_index()
    prefix17_ref = prefix17_ref.drop("index",1)
    prefix18_ref = prefix18[prefix18['case_len']>=28]
    prefix18_ref = prefix18_ref.reset_index()
    prefix18_ref = prefix18_ref.drop("index",1)
    prefix19_ref = prefix19[prefix19['case_len']>=28]
    prefix19_ref = prefix19_ref.reset_index()
    prefix19_ref = prefix19_ref.drop("index",1)
    prefix20_ref = prefix20[prefix20['case_len']>=28]
    prefix20_ref = prefix20_ref.reset_index()
    prefix20_ref = prefix20_ref.drop("index",1)
    prefix21_ref = prefix21[prefix21['case_len']>=28]
    prefix21_ref = prefix21_ref.reset_index()
    prefix21_ref = prefix21_ref.drop("index",1)
    prefix22_ref = prefix22[prefix22['case_len']>=28]
    prefix22_ref = prefix22_ref.reset_index()
    prefix22_ref = prefix22_ref.drop("index",1)
    prefix23_ref = prefix23[prefix23['case_len']>=28]
    prefix23_ref = prefix23_ref.reset_index()
    prefix23_ref = prefix23_ref.drop("index",1)
    prefix24_ref = prefix24[prefix24['case_len']>=28]
    prefix24_ref = prefix24_ref.reset_index()
    prefix24_ref = prefix24_ref.drop("index",1)
    prefix25_ref = prefix25[prefix25['case_len']>=28]
    prefix25_ref = prefix25_ref.reset_index()
    prefix25_ref = prefix25_ref.drop("index",1)
    prefix26_ref = prefix26[prefix26['case_len']>=28]
    prefix26_ref = prefix26_ref.reset_index()
    prefix26_ref = prefix26_ref.drop("index",1)
    prefix27_ref = prefix27[prefix27['case_len']>=28]
    prefix27_ref = prefix27_ref.reset_index()
    prefix27_ref = prefix27_ref.drop("index",1)
    prefix28 = data_dummy_prefix[data_dummy_prefix['prefix']==28]
    prefix28 = prefix28.reset_index()
    prefix28 = prefix28.drop("index",1)
    print(len(prefix28))
    print("Processing columns with prefix 28...")
    def max_prefix_40_step28(iter):
        for col in prefix28:
            if col[len(col)-10:len(col)] == "prefixlen1":
                prefix28[col] = prefix1_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen2":
                prefix28[col] = prefix2_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen3":
                prefix28[col] = prefix3_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen4":
                prefix28[col] = prefix4_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen5":
                prefix28[col] = prefix5_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen6":
                prefix28[col] = prefix6_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen7":
                prefix28[col] = prefix7_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen8":
                prefix28[col] = prefix8_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen9":
                prefix28[col] = prefix9_ref[col[0:len(col)-11]]
            if col[len(col)-11:len(col)] == "prefixlen10":
                prefix28[col] = prefix10_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen11":
                prefix28[col] = prefix11_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen12":
                prefix28[col] = prefix12_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen13":
                prefix28[col] = prefix13_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen14":
                prefix28[col] = prefix14_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen15":
                prefix28[col] = prefix15_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen16":
                prefix28[col] = prefix16_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen17":
                prefix28[col] = prefix17_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen18":
                prefix28[col] = prefix18_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen19":
                prefix28[col] = prefix19_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen20":
                prefix28[col] = prefix20_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen21":
                prefix28[col] = prefix21_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen22":
                prefix28[col] = prefix22_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen23":
                prefix28[col] = prefix23_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen24":
                prefix28[col] = prefix24_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen25":
                prefix28[col] = prefix25_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen26":
                prefix28[col] = prefix26_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen27":
                prefix28[col] = prefix27_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen28":
                prefix28[col] = prefix28[col[0:len(col)-12]]
        return prefix28
    if __name__ == '__main__':
        pool = multiprocessing.Pool(processes=num_cpu_cores)
        prefix28 = pool.map(max_prefix_40_step28, [0])[0]
        pool.close()
        pool.join()

    prefix1_ref = prefix1[prefix1['case_len']>=29]
    prefix1_ref = prefix1_ref.reset_index()
    prefix1_ref = prefix1_ref.drop("index",1)
    prefix2_ref = prefix2[prefix2['case_len']>=29]
    prefix2_ref = prefix2_ref.reset_index()
    prefix2_ref = prefix2_ref.drop("index",1)
    prefix3_ref = prefix3[prefix3['case_len']>=29]
    prefix3_ref = prefix3_ref.reset_index()
    prefix3_ref = prefix3_ref.drop("index",1)
    prefix4_ref = prefix4[prefix4['case_len']>=29]
    prefix4_ref = prefix4_ref.reset_index()
    prefix4_ref = prefix4_ref.drop("index",1)
    prefix5_ref = prefix5[prefix5['case_len']>=29]
    prefix5_ref = prefix5_ref.reset_index()
    prefix5_ref = prefix5_ref.drop("index",1)
    prefix6_ref = prefix6[prefix6['case_len']>=29]
    prefix6_ref = prefix6_ref.reset_index()
    prefix6_ref = prefix6_ref.drop("index",1)
    prefix7_ref = prefix7[prefix7['case_len']>=29]
    prefix7_ref = prefix7_ref.reset_index()
    prefix7_ref = prefix7_ref.drop("index",1)
    prefix8_ref = prefix8[prefix8['case_len']>=29]
    prefix8_ref = prefix8_ref.reset_index()
    prefix8_ref = prefix8_ref.drop("index",1)
    prefix9_ref = prefix9[prefix9['case_len']>=29]
    prefix9_ref = prefix9_ref.reset_index()
    prefix9_ref = prefix9_ref.drop("index",1)
    prefix10_ref = prefix10[prefix10['case_len']>=29]
    prefix10_ref = prefix10_ref.reset_index()
    prefix10_ref = prefix10_ref.drop("index",1)
    prefix11_ref = prefix11[prefix11['case_len']>=29]
    prefix11_ref = prefix11_ref.reset_index()
    prefix11_ref = prefix11_ref.drop("index",1)
    prefix12_ref = prefix12[prefix12['case_len']>=29]
    prefix12_ref = prefix12_ref.reset_index()
    prefix12_ref = prefix12_ref.drop("index",1)
    prefix13_ref = prefix13[prefix13['case_len']>=29]
    prefix13_ref = prefix13_ref.reset_index()
    prefix13_ref = prefix13_ref.drop("index",1)
    prefix14_ref = prefix14[prefix14['case_len']>=29]
    prefix14_ref = prefix14_ref.reset_index()
    prefix14_ref = prefix14_ref.drop("index",1)
    prefix15_ref = prefix15[prefix15['case_len']>=29]
    prefix15_ref = prefix15_ref.reset_index()
    prefix15_ref = prefix15_ref.drop("index",1)
    prefix16_ref = prefix16[prefix16['case_len']>=29]
    prefix16_ref = prefix16_ref.reset_index()
    prefix16_ref = prefix16_ref.drop("index",1)
    prefix17_ref = prefix17[prefix17['case_len']>=29]
    prefix17_ref = prefix17_ref.reset_index()
    prefix17_ref = prefix17_ref.drop("index",1)
    prefix18_ref = prefix18[prefix18['case_len']>=29]
    prefix18_ref = prefix18_ref.reset_index()
    prefix18_ref = prefix18_ref.drop("index",1)
    prefix19_ref = prefix19[prefix19['case_len']>=29]
    prefix19_ref = prefix19_ref.reset_index()
    prefix19_ref = prefix19_ref.drop("index",1)
    prefix20_ref = prefix20[prefix20['case_len']>=29]
    prefix20_ref = prefix20_ref.reset_index()
    prefix20_ref = prefix20_ref.drop("index",1)
    prefix21_ref = prefix21[prefix21['case_len']>=29]
    prefix21_ref = prefix21_ref.reset_index()
    prefix21_ref = prefix21_ref.drop("index",1)
    prefix22_ref = prefix22[prefix22['case_len']>=29]
    prefix22_ref = prefix22_ref.reset_index()
    prefix22_ref = prefix22_ref.drop("index",1)
    prefix23_ref = prefix23[prefix23['case_len']>=29]
    prefix23_ref = prefix23_ref.reset_index()
    prefix23_ref = prefix23_ref.drop("index",1)
    prefix24_ref = prefix24[prefix24['case_len']>=29]
    prefix24_ref = prefix24_ref.reset_index()
    prefix24_ref = prefix24_ref.drop("index",1)
    prefix25_ref = prefix25[prefix25['case_len']>=29]
    prefix25_ref = prefix25_ref.reset_index()
    prefix25_ref = prefix25_ref.drop("index",1)
    prefix26_ref = prefix26[prefix26['case_len']>=29]
    prefix26_ref = prefix26_ref.reset_index()
    prefix26_ref = prefix26_ref.drop("index",1)
    prefix27_ref = prefix27[prefix27['case_len']>=29]
    prefix27_ref = prefix27_ref.reset_index()
    prefix27_ref = prefix27_ref.drop("index",1)
    prefix28_ref = prefix28[prefix28['case_len']>=29]
    prefix28_ref = prefix28_ref.reset_index()
    prefix28_ref = prefix28_ref.drop("index",1)
    prefix29 = data_dummy_prefix[data_dummy_prefix['prefix']==29]
    prefix29 = prefix29.reset_index()
    prefix29 = prefix29.drop("index",1)
    print(len(prefix29))
    print("Processing columns with prefix 29...")
    def max_prefix_40_step29(iter):
        for col in prefix29:
            if col[len(col)-10:len(col)] == "prefixlen1":
                prefix29[col] = prefix1_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen2":
                prefix29[col] = prefix2_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen3":
                prefix29[col] = prefix3_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen4":
                prefix29[col] = prefix4_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen5":
                prefix29[col] = prefix5_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen6":
                prefix29[col] = prefix6_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen7":
                prefix29[col] = prefix7_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen8":
                prefix29[col] = prefix8_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen9":
                prefix29[col] = prefix9_ref[col[0:len(col)-11]]
            if col[len(col)-11:len(col)] == "prefixlen10":
                prefix29[col] = prefix10_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen11":
                prefix29[col] = prefix11_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen12":
                prefix29[col] = prefix12_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen13":
                prefix29[col] = prefix13_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen14":
                prefix29[col] = prefix14_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen15":
                prefix29[col] = prefix15_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen16":
                prefix29[col] = prefix16_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen17":
                prefix29[col] = prefix17_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen18":
                prefix29[col] = prefix18_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen19":
                prefix29[col] = prefix19_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen20":
                prefix29[col] = prefix20_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen21":
                prefix29[col] = prefix21_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen22":
                prefix29[col] = prefix22_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen23":
                prefix29[col] = prefix23_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen24":
                prefix29[col] = prefix24_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen25":
                prefix29[col] = prefix25_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen26":
                prefix29[col] = prefix26_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen27":
                prefix29[col] = prefix27_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen28":
                prefix29[col] = prefix28_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen29":
                prefix29[col] = prefix29[col[0:len(col)-12]]
        return prefix29
    if __name__ == '__main__':
        pool = multiprocessing.Pool(processes=num_cpu_cores)
        prefix29 = pool.map(max_prefix_40_step29, [0])[0]
        pool.close()
        pool.join()

    prefix1_ref = prefix1[prefix1['case_len']>=30]
    prefix1_ref = prefix1_ref.reset_index()
    prefix1_ref = prefix1_ref.drop("index",1)
    prefix2_ref = prefix2[prefix2['case_len']>=30]
    prefix2_ref = prefix2_ref.reset_index()
    prefix2_ref = prefix2_ref.drop("index",1)
    prefix3_ref = prefix3[prefix3['case_len']>=30]
    prefix3_ref = prefix3_ref.reset_index()
    prefix3_ref = prefix3_ref.drop("index",1)
    prefix4_ref = prefix4[prefix4['case_len']>=30]
    prefix4_ref = prefix4_ref.reset_index()
    prefix4_ref = prefix4_ref.drop("index",1)
    prefix5_ref = prefix5[prefix5['case_len']>=30]
    prefix5_ref = prefix5_ref.reset_index()
    prefix5_ref = prefix5_ref.drop("index",1)
    prefix6_ref = prefix6[prefix6['case_len']>=30]
    prefix6_ref = prefix6_ref.reset_index()
    prefix6_ref = prefix6_ref.drop("index",1)
    prefix7_ref = prefix7[prefix7['case_len']>=30]
    prefix7_ref = prefix7_ref.reset_index()
    prefix7_ref = prefix7_ref.drop("index",1)
    prefix8_ref = prefix8[prefix8['case_len']>=30]
    prefix8_ref = prefix8_ref.reset_index()
    prefix8_ref = prefix8_ref.drop("index",1)
    prefix9_ref = prefix9[prefix9['case_len']>=30]
    prefix9_ref = prefix9_ref.reset_index()
    prefix9_ref = prefix9_ref.drop("index",1)
    prefix10_ref = prefix10[prefix10['case_len']>=30]
    prefix10_ref = prefix10_ref.reset_index()
    prefix10_ref = prefix10_ref.drop("index",1)
    prefix11_ref = prefix11[prefix11['case_len']>=30]
    prefix11_ref = prefix11_ref.reset_index()
    prefix11_ref = prefix11_ref.drop("index",1)
    prefix12_ref = prefix12[prefix12['case_len']>=30]
    prefix12_ref = prefix12_ref.reset_index()
    prefix12_ref = prefix12_ref.drop("index",1)
    prefix13_ref = prefix13[prefix13['case_len']>=30]
    prefix13_ref = prefix13_ref.reset_index()
    prefix13_ref = prefix13_ref.drop("index",1)
    prefix14_ref = prefix14[prefix14['case_len']>=30]
    prefix14_ref = prefix14_ref.reset_index()
    prefix14_ref = prefix14_ref.drop("index",1)
    prefix15_ref = prefix15[prefix15['case_len']>=30]
    prefix15_ref = prefix15_ref.reset_index()
    prefix15_ref = prefix15_ref.drop("index",1)
    prefix16_ref = prefix16[prefix16['case_len']>=30]
    prefix16_ref = prefix16_ref.reset_index()
    prefix16_ref = prefix16_ref.drop("index",1)
    prefix17_ref = prefix17[prefix17['case_len']>=30]
    prefix17_ref = prefix17_ref.reset_index()
    prefix17_ref = prefix17_ref.drop("index",1)
    prefix18_ref = prefix18[prefix18['case_len']>=30]
    prefix18_ref = prefix18_ref.reset_index()
    prefix18_ref = prefix18_ref.drop("index",1)
    prefix19_ref = prefix19[prefix19['case_len']>=30]
    prefix19_ref = prefix19_ref.reset_index()
    prefix19_ref = prefix19_ref.drop("index",1)
    prefix20_ref = prefix20[prefix20['case_len']>=30]
    prefix20_ref = prefix20_ref.reset_index()
    prefix20_ref = prefix20_ref.drop("index",1)
    prefix21_ref = prefix21[prefix21['case_len']>=30]
    prefix21_ref = prefix21_ref.reset_index()
    prefix21_ref = prefix21_ref.drop("index",1)
    prefix22_ref = prefix22[prefix22['case_len']>=30]
    prefix22_ref = prefix22_ref.reset_index()
    prefix22_ref = prefix22_ref.drop("index",1)
    prefix23_ref = prefix23[prefix23['case_len']>=30]
    prefix23_ref = prefix23_ref.reset_index()
    prefix23_ref = prefix23_ref.drop("index",1)
    prefix24_ref = prefix24[prefix24['case_len']>=30]
    prefix24_ref = prefix24_ref.reset_index()
    prefix24_ref = prefix24_ref.drop("index",1)
    prefix25_ref = prefix25[prefix25['case_len']>=30]
    prefix25_ref = prefix25_ref.reset_index()
    prefix25_ref = prefix25_ref.drop("index",1)
    prefix26_ref = prefix26[prefix26['case_len']>=30]
    prefix26_ref = prefix26_ref.reset_index()
    prefix26_ref = prefix26_ref.drop("index",1)
    prefix27_ref = prefix27[prefix27['case_len']>=30]
    prefix27_ref = prefix27_ref.reset_index()
    prefix27_ref = prefix27_ref.drop("index",1)
    prefix28_ref = prefix28[prefix28['case_len']>=30]
    prefix28_ref = prefix28_ref.reset_index()
    prefix28_ref = prefix28_ref.drop("index",1)
    prefix29_ref = prefix29[prefix29['case_len']>=30]
    prefix29_ref = prefix29_ref.reset_index()
    prefix29_ref = prefix29_ref.drop("index",1)
    prefix30 = data_dummy_prefix[data_dummy_prefix['prefix']==30]
    prefix30 = prefix30.reset_index()
    prefix30 = prefix30.drop("index",1)
    print(len(prefix30))
    print("Processing columns with prefix 30...")
    def max_prefix_40_step30(iter):
        for col in prefix30:
            if col[len(col)-10:len(col)] == "prefixlen1":
                prefix30[col] = prefix1_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen2":
                prefix30[col] = prefix2_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen3":
                prefix30[col] = prefix3_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen4":
                prefix30[col] = prefix4_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen5":
                prefix30[col] = prefix5_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen6":
                prefix30[col] = prefix6_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen7":
                prefix30[col] = prefix7_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen8":
                prefix30[col] = prefix8_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen9":
                prefix30[col] = prefix9_ref[col[0:len(col)-11]]
            if col[len(col)-11:len(col)] == "prefixlen10":
                prefix30[col] = prefix10_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen11":
                prefix30[col] = prefix11_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen12":
                prefix30[col] = prefix12_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen13":
                prefix30[col] = prefix13_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen14":
                prefix30[col] = prefix14_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen15":
                prefix30[col] = prefix15_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen16":
                prefix30[col] = prefix16_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen17":
                prefix30[col] = prefix17_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen18":
                prefix30[col] = prefix18_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen19":
                prefix30[col] = prefix19_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen20":
                prefix30[col] = prefix20_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen21":
                prefix30[col] = prefix21_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen22":
                prefix30[col] = prefix22_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen23":
                prefix30[col] = prefix23_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen24":
                prefix30[col] = prefix24_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen25":
                prefix30[col] = prefix25_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen26":
                prefix30[col] = prefix26_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen27":
                prefix30[col] = prefix27_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen28":
                prefix30[col] = prefix28_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen29":
                prefix30[col] = prefix29_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen30":
                prefix30[col] = prefix30[col[0:len(col)-12]]
        return prefix30
    if __name__ == '__main__':
        pool = multiprocessing.Pool(processes=num_cpu_cores)
        prefix30 = pool.map(max_prefix_40_step30, [0])[0]
        pool.close()
        pool.join()

    prefix1_ref = prefix1[prefix1['case_len']>=31]
    prefix1_ref = prefix1_ref.reset_index()
    prefix1_ref = prefix1_ref.drop("index",1)
    prefix2_ref = prefix2[prefix2['case_len']>=31]
    prefix2_ref = prefix2_ref.reset_index()
    prefix2_ref = prefix2_ref.drop("index",1)
    prefix3_ref = prefix3[prefix3['case_len']>=31]
    prefix3_ref = prefix3_ref.reset_index()
    prefix3_ref = prefix3_ref.drop("index",1)
    prefix4_ref = prefix4[prefix4['case_len']>=31]
    prefix4_ref = prefix4_ref.reset_index()
    prefix4_ref = prefix4_ref.drop("index",1)
    prefix5_ref = prefix5[prefix5['case_len']>=31]
    prefix5_ref = prefix5_ref.reset_index()
    prefix5_ref = prefix5_ref.drop("index",1)
    prefix6_ref = prefix6[prefix6['case_len']>=31]
    prefix6_ref = prefix6_ref.reset_index()
    prefix6_ref = prefix6_ref.drop("index",1)
    prefix7_ref = prefix7[prefix7['case_len']>=31]
    prefix7_ref = prefix7_ref.reset_index()
    prefix7_ref = prefix7_ref.drop("index",1)
    prefix8_ref = prefix8[prefix8['case_len']>=31]
    prefix8_ref = prefix8_ref.reset_index()
    prefix8_ref = prefix8_ref.drop("index",1)
    prefix9_ref = prefix9[prefix9['case_len']>=31]
    prefix9_ref = prefix9_ref.reset_index()
    prefix9_ref = prefix9_ref.drop("index",1)
    prefix10_ref = prefix10[prefix10['case_len']>=31]
    prefix10_ref = prefix10_ref.reset_index()
    prefix10_ref = prefix10_ref.drop("index",1)
    prefix11_ref = prefix11[prefix11['case_len']>=31]
    prefix11_ref = prefix11_ref.reset_index()
    prefix11_ref = prefix11_ref.drop("index",1)
    prefix12_ref = prefix12[prefix12['case_len']>=31]
    prefix12_ref = prefix12_ref.reset_index()
    prefix12_ref = prefix12_ref.drop("index",1)
    prefix13_ref = prefix13[prefix13['case_len']>=31]
    prefix13_ref = prefix13_ref.reset_index()
    prefix13_ref = prefix13_ref.drop("index",1)
    prefix14_ref = prefix14[prefix14['case_len']>=31]
    prefix14_ref = prefix14_ref.reset_index()
    prefix14_ref = prefix14_ref.drop("index",1)
    prefix15_ref = prefix15[prefix15['case_len']>=31]
    prefix15_ref = prefix15_ref.reset_index()
    prefix15_ref = prefix15_ref.drop("index",1)
    prefix16_ref = prefix16[prefix16['case_len']>=31]
    prefix16_ref = prefix16_ref.reset_index()
    prefix16_ref = prefix16_ref.drop("index",1)
    prefix17_ref = prefix17[prefix17['case_len']>=31]
    prefix17_ref = prefix17_ref.reset_index()
    prefix17_ref = prefix17_ref.drop("index",1)
    prefix18_ref = prefix18[prefix18['case_len']>=31]
    prefix18_ref = prefix18_ref.reset_index()
    prefix18_ref = prefix18_ref.drop("index",1)
    prefix19_ref = prefix19[prefix19['case_len']>=31]
    prefix19_ref = prefix19_ref.reset_index()
    prefix19_ref = prefix19_ref.drop("index",1)
    prefix20_ref = prefix20[prefix20['case_len']>=31]
    prefix20_ref = prefix20_ref.reset_index()
    prefix20_ref = prefix20_ref.drop("index",1)
    prefix21_ref = prefix21[prefix21['case_len']>=31]
    prefix21_ref = prefix21_ref.reset_index()
    prefix21_ref = prefix21_ref.drop("index",1)
    prefix22_ref = prefix22[prefix22['case_len']>=31]
    prefix22_ref = prefix22_ref.reset_index()
    prefix22_ref = prefix22_ref.drop("index",1)
    prefix23_ref = prefix23[prefix23['case_len']>=31]
    prefix23_ref = prefix23_ref.reset_index()
    prefix23_ref = prefix23_ref.drop("index",1)
    prefix24_ref = prefix24[prefix24['case_len']>=31]
    prefix24_ref = prefix24_ref.reset_index()
    prefix24_ref = prefix24_ref.drop("index",1)
    prefix25_ref = prefix25[prefix25['case_len']>=31]
    prefix25_ref = prefix25_ref.reset_index()
    prefix25_ref = prefix25_ref.drop("index",1)
    prefix26_ref = prefix26[prefix26['case_len']>=31]
    prefix26_ref = prefix26_ref.reset_index()
    prefix26_ref = prefix26_ref.drop("index",1)
    prefix27_ref = prefix27[prefix27['case_len']>=31]
    prefix27_ref = prefix27_ref.reset_index()
    prefix27_ref = prefix27_ref.drop("index",1)
    prefix28_ref = prefix28[prefix28['case_len']>=31]
    prefix28_ref = prefix28_ref.reset_index()
    prefix28_ref = prefix28_ref.drop("index",1)
    prefix29_ref = prefix29[prefix29['case_len']>=31]
    prefix29_ref = prefix29_ref.reset_index()
    prefix29_ref = prefix29_ref.drop("index",1)
    prefix30_ref = prefix30[prefix30['case_len']>=31]
    prefix30_ref = prefix30_ref.reset_index()
    prefix30_ref = prefix30_ref.drop("index",1)
    prefix31 = data_dummy_prefix[data_dummy_prefix['prefix']==31]
    prefix31 = prefix31.reset_index()
    prefix31 = prefix31.drop("index",1)
    print(len(prefix31))
    print("Processing columns with prefix 31...")
    def max_prefix_40_step31(iter):
        for col in prefix31:
            if col[len(col)-10:len(col)] == "prefixlen1":
                prefix31[col] = prefix1_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen2":
                prefix31[col] = prefix2_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen3":
                prefix31[col] = prefix3_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen4":
                prefix31[col] = prefix4_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen5":
                prefix31[col] = prefix5_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen6":
                prefix31[col] = prefix6_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen7":
                prefix31[col] = prefix7_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen8":
                prefix31[col] = prefix8_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen9":
                prefix31[col] = prefix9_ref[col[0:len(col)-11]]
            if col[len(col)-11:len(col)] == "prefixlen10":
                prefix31[col] = prefix10_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen11":
                prefix31[col] = prefix11_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen12":
                prefix31[col] = prefix12_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen13":
                prefix31[col] = prefix13_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen14":
                prefix31[col] = prefix14_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen15":
                prefix31[col] = prefix15_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen16":
                prefix31[col] = prefix16_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen17":
                prefix31[col] = prefix17_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen18":
                prefix31[col] = prefix18_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen19":
                prefix31[col] = prefix19_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen20":
                prefix31[col] = prefix20_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen21":
                prefix31[col] = prefix21_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen22":
                prefix31[col] = prefix22_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen23":
                prefix31[col] = prefix23_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen24":
                prefix31[col] = prefix24_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen25":
                prefix31[col] = prefix25_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen26":
                prefix31[col] = prefix26_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen27":
                prefix31[col] = prefix27_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen28":
                prefix31[col] = prefix28_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen29":
                prefix31[col] = prefix29_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen30":
                prefix31[col] = prefix30_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen31":
                prefix31[col] = prefix31[col[0:len(col)-12]]
        return prefix31
    if __name__ == '__main__':
        pool = multiprocessing.Pool(processes=num_cpu_cores)
        prefix31 = pool.map(max_prefix_40_step31, [0])[0]
        pool.close()
        pool.join()

    prefix1_ref = prefix1[prefix1['case_len']>=32]
    prefix1_ref = prefix1_ref.reset_index()
    prefix1_ref = prefix1_ref.drop("index",1)
    prefix2_ref = prefix2[prefix2['case_len']>=32]
    prefix2_ref = prefix2_ref.reset_index()
    prefix2_ref = prefix2_ref.drop("index",1)
    prefix3_ref = prefix3[prefix3['case_len']>=32]
    prefix3_ref = prefix3_ref.reset_index()
    prefix3_ref = prefix3_ref.drop("index",1)
    prefix4_ref = prefix4[prefix4['case_len']>=32]
    prefix4_ref = prefix4_ref.reset_index()
    prefix4_ref = prefix4_ref.drop("index",1)
    prefix5_ref = prefix5[prefix5['case_len']>=32]
    prefix5_ref = prefix5_ref.reset_index()
    prefix5_ref = prefix5_ref.drop("index",1)
    prefix6_ref = prefix6[prefix6['case_len']>=32]
    prefix6_ref = prefix6_ref.reset_index()
    prefix6_ref = prefix6_ref.drop("index",1)
    prefix7_ref = prefix7[prefix7['case_len']>=32]
    prefix7_ref = prefix7_ref.reset_index()
    prefix7_ref = prefix7_ref.drop("index",1)
    prefix8_ref = prefix8[prefix8['case_len']>=32]
    prefix8_ref = prefix8_ref.reset_index()
    prefix8_ref = prefix8_ref.drop("index",1)
    prefix9_ref = prefix9[prefix9['case_len']>=32]
    prefix9_ref = prefix9_ref.reset_index()
    prefix9_ref = prefix9_ref.drop("index",1)
    prefix10_ref = prefix10[prefix10['case_len']>=32]
    prefix10_ref = prefix10_ref.reset_index()
    prefix10_ref = prefix10_ref.drop("index",1)
    prefix11_ref = prefix11[prefix11['case_len']>=32]
    prefix11_ref = prefix11_ref.reset_index()
    prefix11_ref = prefix11_ref.drop("index",1)
    prefix12_ref = prefix12[prefix12['case_len']>=32]
    prefix12_ref = prefix12_ref.reset_index()
    prefix12_ref = prefix12_ref.drop("index",1)
    prefix13_ref = prefix13[prefix13['case_len']>=32]
    prefix13_ref = prefix13_ref.reset_index()
    prefix13_ref = prefix13_ref.drop("index",1)
    prefix14_ref = prefix14[prefix14['case_len']>=32]
    prefix14_ref = prefix14_ref.reset_index()
    prefix14_ref = prefix14_ref.drop("index",1)
    prefix15_ref = prefix15[prefix15['case_len']>=32]
    prefix15_ref = prefix15_ref.reset_index()
    prefix15_ref = prefix15_ref.drop("index",1)
    prefix16_ref = prefix16[prefix16['case_len']>=32]
    prefix16_ref = prefix16_ref.reset_index()
    prefix16_ref = prefix16_ref.drop("index",1)
    prefix17_ref = prefix17[prefix17['case_len']>=32]
    prefix17_ref = prefix17_ref.reset_index()
    prefix17_ref = prefix17_ref.drop("index",1)
    prefix18_ref = prefix18[prefix18['case_len']>=32]
    prefix18_ref = prefix18_ref.reset_index()
    prefix18_ref = prefix18_ref.drop("index",1)
    prefix19_ref = prefix19[prefix19['case_len']>=32]
    prefix19_ref = prefix19_ref.reset_index()
    prefix19_ref = prefix19_ref.drop("index",1)
    prefix20_ref = prefix20[prefix20['case_len']>=32]
    prefix20_ref = prefix20_ref.reset_index()
    prefix20_ref = prefix20_ref.drop("index",1)
    prefix21_ref = prefix21[prefix21['case_len']>=32]
    prefix21_ref = prefix21_ref.reset_index()
    prefix21_ref = prefix21_ref.drop("index",1)
    prefix22_ref = prefix22[prefix22['case_len']>=32]
    prefix22_ref = prefix22_ref.reset_index()
    prefix22_ref = prefix22_ref.drop("index",1)
    prefix23_ref = prefix23[prefix23['case_len']>=32]
    prefix23_ref = prefix23_ref.reset_index()
    prefix23_ref = prefix23_ref.drop("index",1)
    prefix24_ref = prefix24[prefix24['case_len']>=32]
    prefix24_ref = prefix24_ref.reset_index()
    prefix24_ref = prefix24_ref.drop("index",1)
    prefix25_ref = prefix25[prefix25['case_len']>=32]
    prefix25_ref = prefix25_ref.reset_index()
    prefix25_ref = prefix25_ref.drop("index",1)
    prefix26_ref = prefix26[prefix26['case_len']>=32]
    prefix26_ref = prefix26_ref.reset_index()
    prefix26_ref = prefix26_ref.drop("index",1)
    prefix27_ref = prefix27[prefix27['case_len']>=32]
    prefix27_ref = prefix27_ref.reset_index()
    prefix27_ref = prefix27_ref.drop("index",1)
    prefix28_ref = prefix28[prefix28['case_len']>=32]
    prefix28_ref = prefix28_ref.reset_index()
    prefix28_ref = prefix28_ref.drop("index",1)
    prefix29_ref = prefix29[prefix29['case_len']>=32]
    prefix29_ref = prefix29_ref.reset_index()
    prefix29_ref = prefix29_ref.drop("index",1)
    prefix30_ref = prefix30[prefix30['case_len']>=32]
    prefix30_ref = prefix30_ref.reset_index()
    prefix30_ref = prefix30_ref.drop("index",1)
    prefix31_ref = prefix31[prefix31['case_len']>=32]
    prefix31_ref = prefix31_ref.reset_index()
    prefix31_ref = prefix31_ref.drop("index",1)
    prefix32 = data_dummy_prefix[data_dummy_prefix['prefix']==32]
    prefix32 = prefix32.reset_index()
    prefix32 = prefix32.drop("index",1)
    print(len(prefix32))
    print("Processing columns with prefix 32...")
    def max_prefix_40_step32(iter):
        for col in prefix32:
            if col[len(col)-10:len(col)] == "prefixlen1":
                prefix32[col] = prefix1_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen2":
                prefix32[col] = prefix2_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen3":
                prefix32[col] = prefix3_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen4":
                prefix32[col] = prefix4_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen5":
                prefix32[col] = prefix5_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen6":
                prefix32[col] = prefix6_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen7":
                prefix32[col] = prefix7_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen8":
                prefix32[col] = prefix8_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen9":
                prefix32[col] = prefix9_ref[col[0:len(col)-11]]
            if col[len(col)-11:len(col)] == "prefixlen10":
                prefix32[col] = prefix10_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen11":
                prefix32[col] = prefix11_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen12":
                prefix32[col] = prefix12_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen13":
                prefix32[col] = prefix13_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen14":
                prefix32[col] = prefix14_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen15":
                prefix32[col] = prefix15_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen16":
                prefix32[col] = prefix16_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen17":
                prefix32[col] = prefix17_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen18":
                prefix32[col] = prefix18_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen19":
                prefix32[col] = prefix19_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen20":
                prefix32[col] = prefix20_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen21":
                prefix32[col] = prefix21_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen22":
                prefix32[col] = prefix22_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen23":
                prefix32[col] = prefix23_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen24":
                prefix32[col] = prefix24_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen25":
                prefix32[col] = prefix25_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen26":
                prefix32[col] = prefix26_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen27":
                prefix32[col] = prefix27_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen28":
                prefix32[col] = prefix28_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen29":
                prefix32[col] = prefix29_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen30":
                prefix32[col] = prefix30_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen31":
                prefix32[col] = prefix31_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen32":
                prefix32[col] = prefix32[col[0:len(col)-12]]
        return prefix32
    if __name__ == '__main__':
        pool = multiprocessing.Pool(processes=num_cpu_cores)
        prefix32 = pool.map(max_prefix_40_step32, [0])[0]
        pool.close()
        pool.join()

    prefix1_ref = prefix1[prefix1['case_len']>=33]
    prefix1_ref = prefix1_ref.reset_index()
    prefix1_ref = prefix1_ref.drop("index",1)
    prefix2_ref = prefix2[prefix2['case_len']>=33]
    prefix2_ref = prefix2_ref.reset_index()
    prefix2_ref = prefix2_ref.drop("index",1)
    prefix3_ref = prefix3[prefix3['case_len']>=33]
    prefix3_ref = prefix3_ref.reset_index()
    prefix3_ref = prefix3_ref.drop("index",1)
    prefix4_ref = prefix4[prefix4['case_len']>=33]
    prefix4_ref = prefix4_ref.reset_index()
    prefix4_ref = prefix4_ref.drop("index",1)
    prefix5_ref = prefix5[prefix5['case_len']>=33]
    prefix5_ref = prefix5_ref.reset_index()
    prefix5_ref = prefix5_ref.drop("index",1)
    prefix6_ref = prefix6[prefix6['case_len']>=33]
    prefix6_ref = prefix6_ref.reset_index()
    prefix6_ref = prefix6_ref.drop("index",1)
    prefix7_ref = prefix7[prefix7['case_len']>=33]
    prefix7_ref = prefix7_ref.reset_index()
    prefix7_ref = prefix7_ref.drop("index",1)
    prefix8_ref = prefix8[prefix8['case_len']>=33]
    prefix8_ref = prefix8_ref.reset_index()
    prefix8_ref = prefix8_ref.drop("index",1)
    prefix9_ref = prefix9[prefix9['case_len']>=33]
    prefix9_ref = prefix9_ref.reset_index()
    prefix9_ref = prefix9_ref.drop("index",1)
    prefix10_ref = prefix10[prefix10['case_len']>=33]
    prefix10_ref = prefix10_ref.reset_index()
    prefix10_ref = prefix10_ref.drop("index",1)
    prefix11_ref = prefix11[prefix11['case_len']>=33]
    prefix11_ref = prefix11_ref.reset_index()
    prefix11_ref = prefix11_ref.drop("index",1)
    prefix12_ref = prefix12[prefix12['case_len']>=33]
    prefix12_ref = prefix12_ref.reset_index()
    prefix12_ref = prefix12_ref.drop("index",1)
    prefix13_ref = prefix13[prefix13['case_len']>=33]
    prefix13_ref = prefix13_ref.reset_index()
    prefix13_ref = prefix13_ref.drop("index",1)
    prefix14_ref = prefix14[prefix14['case_len']>=33]
    prefix14_ref = prefix14_ref.reset_index()
    prefix14_ref = prefix14_ref.drop("index",1)
    prefix15_ref = prefix15[prefix15['case_len']>=33]
    prefix15_ref = prefix15_ref.reset_index()
    prefix15_ref = prefix15_ref.drop("index",1)
    prefix16_ref = prefix16[prefix16['case_len']>=33]
    prefix16_ref = prefix16_ref.reset_index()
    prefix16_ref = prefix16_ref.drop("index",1)
    prefix17_ref = prefix17[prefix17['case_len']>=33]
    prefix17_ref = prefix17_ref.reset_index()
    prefix17_ref = prefix17_ref.drop("index",1)
    prefix18_ref = prefix18[prefix18['case_len']>=33]
    prefix18_ref = prefix18_ref.reset_index()
    prefix18_ref = prefix18_ref.drop("index",1)
    prefix19_ref = prefix19[prefix19['case_len']>=33]
    prefix19_ref = prefix19_ref.reset_index()
    prefix19_ref = prefix19_ref.drop("index",1)
    prefix20_ref = prefix20[prefix20['case_len']>=33]
    prefix20_ref = prefix20_ref.reset_index()
    prefix20_ref = prefix20_ref.drop("index",1)
    prefix21_ref = prefix21[prefix21['case_len']>=33]
    prefix21_ref = prefix21_ref.reset_index()
    prefix21_ref = prefix21_ref.drop("index",1)
    prefix22_ref = prefix22[prefix22['case_len']>=33]
    prefix22_ref = prefix22_ref.reset_index()
    prefix22_ref = prefix22_ref.drop("index",1)
    prefix23_ref = prefix23[prefix23['case_len']>=33]
    prefix23_ref = prefix23_ref.reset_index()
    prefix23_ref = prefix23_ref.drop("index",1)
    prefix24_ref = prefix24[prefix24['case_len']>=33]
    prefix24_ref = prefix24_ref.reset_index()
    prefix24_ref = prefix24_ref.drop("index",1)
    prefix25_ref = prefix25[prefix25['case_len']>=33]
    prefix25_ref = prefix25_ref.reset_index()
    prefix25_ref = prefix25_ref.drop("index",1)
    prefix26_ref = prefix26[prefix26['case_len']>=33]
    prefix26_ref = prefix26_ref.reset_index()
    prefix26_ref = prefix26_ref.drop("index",1)
    prefix27_ref = prefix27[prefix27['case_len']>=33]
    prefix27_ref = prefix27_ref.reset_index()
    prefix27_ref = prefix27_ref.drop("index",1)
    prefix28_ref = prefix28[prefix28['case_len']>=33]
    prefix28_ref = prefix28_ref.reset_index()
    prefix28_ref = prefix28_ref.drop("index",1)
    prefix29_ref = prefix29[prefix29['case_len']>=33]
    prefix29_ref = prefix29_ref.reset_index()
    prefix29_ref = prefix29_ref.drop("index",1)
    prefix30_ref = prefix30[prefix30['case_len']>=33]
    prefix30_ref = prefix30_ref.reset_index()
    prefix30_ref = prefix30_ref.drop("index",1)
    prefix31_ref = prefix31[prefix31['case_len']>=33]
    prefix31_ref = prefix31_ref.reset_index()
    prefix31_ref = prefix31_ref.drop("index",1)
    prefix32_ref = prefix32[prefix32['case_len']>=33]
    prefix32_ref = prefix32_ref.reset_index()
    prefix32_ref = prefix32_ref.drop("index",1)
    prefix33 = data_dummy_prefix[data_dummy_prefix['prefix']==33]
    prefix33 = prefix33.reset_index()
    prefix33 = prefix33.drop("index",1)
    print(len(prefix33))
    print("Processing columns with prefix 33...")
    def max_prefix_40_step33(iter):
        for col in prefix33:
            if col[len(col)-10:len(col)] == "prefixlen1":
                prefix33[col] = prefix1_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen2":
                prefix33[col] = prefix2_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen3":
                prefix33[col] = prefix3_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen4":
                prefix33[col] = prefix4_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen5":
                prefix33[col] = prefix5_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen6":
                prefix33[col] = prefix6_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen7":
                prefix33[col] = prefix7_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen8":
                prefix33[col] = prefix8_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen9":
                prefix33[col] = prefix9_ref[col[0:len(col)-11]]
            if col[len(col)-11:len(col)] == "prefixlen10":
                prefix33[col] = prefix10_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen11":
                prefix33[col] = prefix11_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen12":
                prefix33[col] = prefix12_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen13":
                prefix33[col] = prefix13_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen14":
                prefix33[col] = prefix14_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen15":
                prefix33[col] = prefix15_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen16":
                prefix33[col] = prefix16_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen17":
                prefix33[col] = prefix17_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen18":
                prefix33[col] = prefix18_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen19":
                prefix33[col] = prefix19_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen20":
                prefix33[col] = prefix20_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen21":
                prefix33[col] = prefix21_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen22":
                prefix33[col] = prefix22_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen23":
                prefix33[col] = prefix23_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen24":
                prefix33[col] = prefix24_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen25":
                prefix33[col] = prefix25_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen26":
                prefix33[col] = prefix26_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen27":
                prefix33[col] = prefix27_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen28":
                prefix33[col] = prefix28_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen29":
                prefix33[col] = prefix29_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen30":
                prefix33[col] = prefix30_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen31":
                prefix33[col] = prefix31_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen32":
                prefix33[col] = prefix32_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen33":
                prefix33[col] = prefix33[col[0:len(col)-12]]
        return prefix33
    if __name__ == '__main__':
        pool = multiprocessing.Pool(processes=num_cpu_cores)
        prefix33 = pool.map(max_prefix_40_step33, [0])[0]
        pool.close()
        pool.join()

    prefix1_ref = prefix1[prefix1['case_len']>=34]
    prefix1_ref = prefix1_ref.reset_index()
    prefix1_ref = prefix1_ref.drop("index",1)
    prefix2_ref = prefix2[prefix2['case_len']>=34]
    prefix2_ref = prefix2_ref.reset_index()
    prefix2_ref = prefix2_ref.drop("index",1)
    prefix3_ref = prefix3[prefix3['case_len']>=34]
    prefix3_ref = prefix3_ref.reset_index()
    prefix3_ref = prefix3_ref.drop("index",1)
    prefix4_ref = prefix4[prefix4['case_len']>=34]
    prefix4_ref = prefix4_ref.reset_index()
    prefix4_ref = prefix4_ref.drop("index",1)
    prefix5_ref = prefix5[prefix5['case_len']>=34]
    prefix5_ref = prefix5_ref.reset_index()
    prefix5_ref = prefix5_ref.drop("index",1)
    prefix6_ref = prefix6[prefix6['case_len']>=34]
    prefix6_ref = prefix6_ref.reset_index()
    prefix6_ref = prefix6_ref.drop("index",1)
    prefix7_ref = prefix7[prefix7['case_len']>=34]
    prefix7_ref = prefix7_ref.reset_index()
    prefix7_ref = prefix7_ref.drop("index",1)
    prefix8_ref = prefix8[prefix8['case_len']>=34]
    prefix8_ref = prefix8_ref.reset_index()
    prefix8_ref = prefix8_ref.drop("index",1)
    prefix9_ref = prefix9[prefix9['case_len']>=34]
    prefix9_ref = prefix9_ref.reset_index()
    prefix9_ref = prefix9_ref.drop("index",1)
    prefix10_ref = prefix10[prefix10['case_len']>=34]
    prefix10_ref = prefix10_ref.reset_index()
    prefix10_ref = prefix10_ref.drop("index",1)
    prefix11_ref = prefix11[prefix11['case_len']>=34]
    prefix11_ref = prefix11_ref.reset_index()
    prefix11_ref = prefix11_ref.drop("index",1)
    prefix12_ref = prefix12[prefix12['case_len']>=34]
    prefix12_ref = prefix12_ref.reset_index()
    prefix12_ref = prefix12_ref.drop("index",1)
    prefix13_ref = prefix13[prefix13['case_len']>=34]
    prefix13_ref = prefix13_ref.reset_index()
    prefix13_ref = prefix13_ref.drop("index",1)
    prefix14_ref = prefix14[prefix14['case_len']>=34]
    prefix14_ref = prefix14_ref.reset_index()
    prefix14_ref = prefix14_ref.drop("index",1)
    prefix15_ref = prefix15[prefix15['case_len']>=34]
    prefix15_ref = prefix15_ref.reset_index()
    prefix15_ref = prefix15_ref.drop("index",1)
    prefix16_ref = prefix16[prefix16['case_len']>=34]
    prefix16_ref = prefix16_ref.reset_index()
    prefix16_ref = prefix16_ref.drop("index",1)
    prefix17_ref = prefix17[prefix17['case_len']>=34]
    prefix17_ref = prefix17_ref.reset_index()
    prefix17_ref = prefix17_ref.drop("index",1)
    prefix18_ref = prefix18[prefix18['case_len']>=34]
    prefix18_ref = prefix18_ref.reset_index()
    prefix18_ref = prefix18_ref.drop("index",1)
    prefix19_ref = prefix19[prefix19['case_len']>=34]
    prefix19_ref = prefix19_ref.reset_index()
    prefix19_ref = prefix19_ref.drop("index",1)
    prefix20_ref = prefix20[prefix20['case_len']>=34]
    prefix20_ref = prefix20_ref.reset_index()
    prefix20_ref = prefix20_ref.drop("index",1)
    prefix21_ref = prefix21[prefix21['case_len']>=34]
    prefix21_ref = prefix21_ref.reset_index()
    prefix21_ref = prefix21_ref.drop("index",1)
    prefix22_ref = prefix22[prefix22['case_len']>=34]
    prefix22_ref = prefix22_ref.reset_index()
    prefix22_ref = prefix22_ref.drop("index",1)
    prefix23_ref = prefix23[prefix23['case_len']>=34]
    prefix23_ref = prefix23_ref.reset_index()
    prefix23_ref = prefix23_ref.drop("index",1)
    prefix24_ref = prefix24[prefix24['case_len']>=34]
    prefix24_ref = prefix24_ref.reset_index()
    prefix24_ref = prefix24_ref.drop("index",1)
    prefix25_ref = prefix25[prefix25['case_len']>=34]
    prefix25_ref = prefix25_ref.reset_index()
    prefix25_ref = prefix25_ref.drop("index",1)
    prefix26_ref = prefix26[prefix26['case_len']>=34]
    prefix26_ref = prefix26_ref.reset_index()
    prefix26_ref = prefix26_ref.drop("index",1)
    prefix27_ref = prefix27[prefix27['case_len']>=34]
    prefix27_ref = prefix27_ref.reset_index()
    prefix27_ref = prefix27_ref.drop("index",1)
    prefix28_ref = prefix28[prefix28['case_len']>=34]
    prefix28_ref = prefix28_ref.reset_index()
    prefix28_ref = prefix28_ref.drop("index",1)
    prefix29_ref = prefix29[prefix29['case_len']>=34]
    prefix29_ref = prefix29_ref.reset_index()
    prefix29_ref = prefix29_ref.drop("index",1)
    prefix30_ref = prefix30[prefix30['case_len']>=34]
    prefix30_ref = prefix30_ref.reset_index()
    prefix30_ref = prefix30_ref.drop("index",1)
    prefix31_ref = prefix31[prefix31['case_len']>=34]
    prefix31_ref = prefix31_ref.reset_index()
    prefix31_ref = prefix31_ref.drop("index",1)
    prefix32_ref = prefix32[prefix32['case_len']>=34]
    prefix32_ref = prefix32_ref.reset_index()
    prefix32_ref = prefix32_ref.drop("index",1)
    prefix33_ref = prefix33[prefix33['case_len']>=34]
    prefix33_ref = prefix33_ref.reset_index()
    prefix33_ref = prefix33_ref.drop("index",1)
    prefix34 = data_dummy_prefix[data_dummy_prefix['prefix']==34]
    prefix34 = prefix34.reset_index()
    prefix34 = prefix34.drop("index",1)
    print(len(prefix34))
    print("Processing columns with prefix 34...")
    def max_prefix_40_step34(iter):
        for col in prefix34:
            if col[len(col)-10:len(col)] == "prefixlen1":
                prefix34[col] = prefix1_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen2":
                prefix34[col] = prefix2_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen3":
                prefix34[col] = prefix3_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen4":
                prefix34[col] = prefix4_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen5":
                prefix34[col] = prefix5_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen6":
                prefix34[col] = prefix6_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen7":
                prefix34[col] = prefix7_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen8":
                prefix34[col] = prefix8_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen9":
                prefix34[col] = prefix9_ref[col[0:len(col)-11]]
            if col[len(col)-11:len(col)] == "prefixlen10":
                prefix34[col] = prefix10_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen11":
                prefix34[col] = prefix11_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen12":
                prefix34[col] = prefix12_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen13":
                prefix34[col] = prefix13_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen14":
                prefix34[col] = prefix14_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen15":
                prefix34[col] = prefix15_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen16":
                prefix34[col] = prefix16_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen17":
                prefix34[col] = prefix17_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen18":
                prefix34[col] = prefix18_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen19":
                prefix34[col] = prefix19_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen20":
                prefix34[col] = prefix20_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen21":
                prefix34[col] = prefix21_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen22":
                prefix34[col] = prefix22_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen23":
                prefix34[col] = prefix23_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen24":
                prefix34[col] = prefix24_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen25":
                prefix34[col] = prefix25_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen26":
                prefix34[col] = prefix26_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen27":
                prefix34[col] = prefix27_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen28":
                prefix34[col] = prefix28_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen29":
                prefix34[col] = prefix29_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen30":
                prefix34[col] = prefix30_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen31":
                prefix34[col] = prefix31_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen32":
                prefix34[col] = prefix32_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen33":
                prefix34[col] = prefix33_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen34":
                prefix34[col] = prefix34[col[0:len(col)-12]]
        return prefix34
    if __name__ == '__main__':
        pool = multiprocessing.Pool(processes=num_cpu_cores)
        prefix34 = pool.map(max_prefix_40_step34, [0])[0]
        pool.close()
        pool.join()

    prefix1_ref = prefix1[prefix1['case_len']>=35]
    prefix1_ref = prefix1_ref.reset_index()
    prefix1_ref = prefix1_ref.drop("index",1)
    prefix2_ref = prefix2[prefix2['case_len']>=35]
    prefix2_ref = prefix2_ref.reset_index()
    prefix2_ref = prefix2_ref.drop("index",1)
    prefix3_ref = prefix3[prefix3['case_len']>=35]
    prefix3_ref = prefix3_ref.reset_index()
    prefix3_ref = prefix3_ref.drop("index",1)
    prefix4_ref = prefix4[prefix4['case_len']>=35]
    prefix4_ref = prefix4_ref.reset_index()
    prefix4_ref = prefix4_ref.drop("index",1)
    prefix5_ref = prefix5[prefix5['case_len']>=35]
    prefix5_ref = prefix5_ref.reset_index()
    prefix5_ref = prefix5_ref.drop("index",1)
    prefix6_ref = prefix6[prefix6['case_len']>=35]
    prefix6_ref = prefix6_ref.reset_index()
    prefix6_ref = prefix6_ref.drop("index",1)
    prefix7_ref = prefix7[prefix7['case_len']>=35]
    prefix7_ref = prefix7_ref.reset_index()
    prefix7_ref = prefix7_ref.drop("index",1)
    prefix8_ref = prefix8[prefix8['case_len']>=35]
    prefix8_ref = prefix8_ref.reset_index()
    prefix8_ref = prefix8_ref.drop("index",1)
    prefix9_ref = prefix9[prefix9['case_len']>=35]
    prefix9_ref = prefix9_ref.reset_index()
    prefix9_ref = prefix9_ref.drop("index",1)
    prefix10_ref = prefix10[prefix10['case_len']>=35]
    prefix10_ref = prefix10_ref.reset_index()
    prefix10_ref = prefix10_ref.drop("index",1)
    prefix11_ref = prefix11[prefix11['case_len']>=35]
    prefix11_ref = prefix11_ref.reset_index()
    prefix11_ref = prefix11_ref.drop("index",1)
    prefix12_ref = prefix12[prefix12['case_len']>=35]
    prefix12_ref = prefix12_ref.reset_index()
    prefix12_ref = prefix12_ref.drop("index",1)
    prefix13_ref = prefix13[prefix13['case_len']>=35]
    prefix13_ref = prefix13_ref.reset_index()
    prefix13_ref = prefix13_ref.drop("index",1)
    prefix14_ref = prefix14[prefix14['case_len']>=35]
    prefix14_ref = prefix14_ref.reset_index()
    prefix14_ref = prefix14_ref.drop("index",1)
    prefix15_ref = prefix15[prefix15['case_len']>=35]
    prefix15_ref = prefix15_ref.reset_index()
    prefix15_ref = prefix15_ref.drop("index",1)
    prefix16_ref = prefix16[prefix16['case_len']>=35]
    prefix16_ref = prefix16_ref.reset_index()
    prefix16_ref = prefix16_ref.drop("index",1)
    prefix17_ref = prefix17[prefix17['case_len']>=35]
    prefix17_ref = prefix17_ref.reset_index()
    prefix17_ref = prefix17_ref.drop("index",1)
    prefix18_ref = prefix18[prefix18['case_len']>=35]
    prefix18_ref = prefix18_ref.reset_index()
    prefix18_ref = prefix18_ref.drop("index",1)
    prefix19_ref = prefix19[prefix19['case_len']>=35]
    prefix19_ref = prefix19_ref.reset_index()
    prefix19_ref = prefix19_ref.drop("index",1)
    prefix20_ref = prefix20[prefix20['case_len']>=35]
    prefix20_ref = prefix20_ref.reset_index()
    prefix20_ref = prefix20_ref.drop("index",1)
    prefix21_ref = prefix21[prefix21['case_len']>=35]
    prefix21_ref = prefix21_ref.reset_index()
    prefix21_ref = prefix21_ref.drop("index",1)
    prefix22_ref = prefix22[prefix22['case_len']>=35]
    prefix22_ref = prefix22_ref.reset_index()
    prefix22_ref = prefix22_ref.drop("index",1)
    prefix23_ref = prefix23[prefix23['case_len']>=35]
    prefix23_ref = prefix23_ref.reset_index()
    prefix23_ref = prefix23_ref.drop("index",1)
    prefix24_ref = prefix24[prefix24['case_len']>=35]
    prefix24_ref = prefix24_ref.reset_index()
    prefix24_ref = prefix24_ref.drop("index",1)
    prefix25_ref = prefix25[prefix25['case_len']>=35]
    prefix25_ref = prefix25_ref.reset_index()
    prefix25_ref = prefix25_ref.drop("index",1)
    prefix26_ref = prefix26[prefix26['case_len']>=35]
    prefix26_ref = prefix26_ref.reset_index()
    prefix26_ref = prefix26_ref.drop("index",1)
    prefix27_ref = prefix27[prefix27['case_len']>=35]
    prefix27_ref = prefix27_ref.reset_index()
    prefix27_ref = prefix27_ref.drop("index",1)
    prefix28_ref = prefix28[prefix28['case_len']>=35]
    prefix28_ref = prefix28_ref.reset_index()
    prefix28_ref = prefix28_ref.drop("index",1)
    prefix29_ref = prefix29[prefix29['case_len']>=35]
    prefix29_ref = prefix29_ref.reset_index()
    prefix29_ref = prefix29_ref.drop("index",1)
    prefix30_ref = prefix30[prefix30['case_len']>=35]
    prefix30_ref = prefix30_ref.reset_index()
    prefix30_ref = prefix30_ref.drop("index",1)
    prefix31_ref = prefix31[prefix31['case_len']>=35]
    prefix31_ref = prefix31_ref.reset_index()
    prefix31_ref = prefix31_ref.drop("index",1)
    prefix32_ref = prefix32[prefix32['case_len']>=35]
    prefix32_ref = prefix32_ref.reset_index()
    prefix32_ref = prefix32_ref.drop("index",1)
    prefix33_ref = prefix33[prefix33['case_len']>=35]
    prefix33_ref = prefix33_ref.reset_index()
    prefix33_ref = prefix33_ref.drop("index",1)
    prefix34_ref = prefix34[prefix34['case_len']>=35]
    prefix34_ref = prefix34_ref.reset_index()
    prefix34_ref = prefix34_ref.drop("index",1)
    prefix35 = data_dummy_prefix[data_dummy_prefix['prefix']==35]
    prefix35 = prefix35.reset_index()
    prefix35 = prefix35.drop("index",1)
    print(len(prefix35))
    print("Processing columns with prefix 35...")
    def max_prefix_40_step35(iter):
        for col in prefix35:
            if col[len(col)-10:len(col)] == "prefixlen1":
                prefix35[col] = prefix1_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen2":
                prefix35[col] = prefix2_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen3":
                prefix35[col] = prefix3_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen4":
                prefix35[col] = prefix4_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen5":
                prefix35[col] = prefix5_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen6":
                prefix35[col] = prefix6_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen7":
                prefix35[col] = prefix7_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen8":
                prefix35[col] = prefix8_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen9":
                prefix35[col] = prefix9_ref[col[0:len(col)-11]]
            if col[len(col)-11:len(col)] == "prefixlen10":
                prefix35[col] = prefix10_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen11":
                prefix35[col] = prefix11_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen12":
                prefix35[col] = prefix12_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen13":
                prefix35[col] = prefix13_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen14":
                prefix35[col] = prefix14_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen15":
                prefix35[col] = prefix15_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen16":
                prefix35[col] = prefix16_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen17":
                prefix35[col] = prefix17_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen18":
                prefix35[col] = prefix18_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen19":
                prefix35[col] = prefix19_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen20":
                prefix35[col] = prefix20_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen21":
                prefix35[col] = prefix21_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen22":
                prefix35[col] = prefix22_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen23":
                prefix35[col] = prefix23_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen24":
                prefix35[col] = prefix24_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen25":
                prefix35[col] = prefix25_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen26":
                prefix35[col] = prefix26_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen27":
                prefix35[col] = prefix27_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen28":
                prefix35[col] = prefix28_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen29":
                prefix35[col] = prefix29_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen30":
                prefix35[col] = prefix30_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen31":
                prefix35[col] = prefix31_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen32":
                prefix35[col] = prefix32_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen33":
                prefix35[col] = prefix33_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen34":
                prefix35[col] = prefix34_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen35":
                prefix35[col] = prefix35[col[0:len(col)-12]]
        return prefix35
    if __name__ == '__main__':
        pool = multiprocessing.Pool(processes=num_cpu_cores)
        prefix35 = pool.map(max_prefix_40_step35, [0])[0]
        pool.close()
        pool.join()

    prefix1_ref = prefix1[prefix1['case_len']>=36]
    prefix1_ref = prefix1_ref.reset_index()
    prefix1_ref = prefix1_ref.drop("index",1)
    prefix2_ref = prefix2[prefix2['case_len']>=36]
    prefix2_ref = prefix2_ref.reset_index()
    prefix2_ref = prefix2_ref.drop("index",1)
    prefix3_ref = prefix3[prefix3['case_len']>=36]
    prefix3_ref = prefix3_ref.reset_index()
    prefix3_ref = prefix3_ref.drop("index",1)
    prefix4_ref = prefix4[prefix4['case_len']>=36]
    prefix4_ref = prefix4_ref.reset_index()
    prefix4_ref = prefix4_ref.drop("index",1)
    prefix5_ref = prefix5[prefix5['case_len']>=36]
    prefix5_ref = prefix5_ref.reset_index()
    prefix5_ref = prefix5_ref.drop("index",1)
    prefix6_ref = prefix6[prefix6['case_len']>=36]
    prefix6_ref = prefix6_ref.reset_index()
    prefix6_ref = prefix6_ref.drop("index",1)
    prefix7_ref = prefix7[prefix7['case_len']>=36]
    prefix7_ref = prefix7_ref.reset_index()
    prefix7_ref = prefix7_ref.drop("index",1)
    prefix8_ref = prefix8[prefix8['case_len']>=36]
    prefix8_ref = prefix8_ref.reset_index()
    prefix8_ref = prefix8_ref.drop("index",1)
    prefix9_ref = prefix9[prefix9['case_len']>=36]
    prefix9_ref = prefix9_ref.reset_index()
    prefix9_ref = prefix9_ref.drop("index",1)
    prefix10_ref = prefix10[prefix10['case_len']>=36]
    prefix10_ref = prefix10_ref.reset_index()
    prefix10_ref = prefix10_ref.drop("index",1)
    prefix11_ref = prefix11[prefix11['case_len']>=36]
    prefix11_ref = prefix11_ref.reset_index()
    prefix11_ref = prefix11_ref.drop("index",1)
    prefix12_ref = prefix12[prefix12['case_len']>=36]
    prefix12_ref = prefix12_ref.reset_index()
    prefix12_ref = prefix12_ref.drop("index",1)
    prefix13_ref = prefix13[prefix13['case_len']>=36]
    prefix13_ref = prefix13_ref.reset_index()
    prefix13_ref = prefix13_ref.drop("index",1)
    prefix14_ref = prefix14[prefix14['case_len']>=36]
    prefix14_ref = prefix14_ref.reset_index()
    prefix14_ref = prefix14_ref.drop("index",1)
    prefix15_ref = prefix15[prefix15['case_len']>=36]
    prefix15_ref = prefix15_ref.reset_index()
    prefix15_ref = prefix15_ref.drop("index",1)
    prefix16_ref = prefix16[prefix16['case_len']>=36]
    prefix16_ref = prefix16_ref.reset_index()
    prefix16_ref = prefix16_ref.drop("index",1)
    prefix17_ref = prefix17[prefix17['case_len']>=36]
    prefix17_ref = prefix17_ref.reset_index()
    prefix17_ref = prefix17_ref.drop("index",1)
    prefix18_ref = prefix18[prefix18['case_len']>=36]
    prefix18_ref = prefix18_ref.reset_index()
    prefix18_ref = prefix18_ref.drop("index",1)
    prefix19_ref = prefix19[prefix19['case_len']>=36]
    prefix19_ref = prefix19_ref.reset_index()
    prefix19_ref = prefix19_ref.drop("index",1)
    prefix20_ref = prefix20[prefix20['case_len']>=36]
    prefix20_ref = prefix20_ref.reset_index()
    prefix20_ref = prefix20_ref.drop("index",1)
    prefix21_ref = prefix21[prefix21['case_len']>=36]
    prefix21_ref = prefix21_ref.reset_index()
    prefix21_ref = prefix21_ref.drop("index",1)
    prefix22_ref = prefix22[prefix22['case_len']>=36]
    prefix22_ref = prefix22_ref.reset_index()
    prefix22_ref = prefix22_ref.drop("index",1)
    prefix23_ref = prefix23[prefix23['case_len']>=36]
    prefix23_ref = prefix23_ref.reset_index()
    prefix23_ref = prefix23_ref.drop("index",1)
    prefix24_ref = prefix24[prefix24['case_len']>=36]
    prefix24_ref = prefix24_ref.reset_index()
    prefix24_ref = prefix24_ref.drop("index",1)
    prefix25_ref = prefix25[prefix25['case_len']>=36]
    prefix25_ref = prefix25_ref.reset_index()
    prefix25_ref = prefix25_ref.drop("index",1)
    prefix26_ref = prefix26[prefix26['case_len']>=36]
    prefix26_ref = prefix26_ref.reset_index()
    prefix26_ref = prefix26_ref.drop("index",1)
    prefix27_ref = prefix27[prefix27['case_len']>=36]
    prefix27_ref = prefix27_ref.reset_index()
    prefix27_ref = prefix27_ref.drop("index",1)
    prefix28_ref = prefix28[prefix28['case_len']>=36]
    prefix28_ref = prefix28_ref.reset_index()
    prefix28_ref = prefix28_ref.drop("index",1)
    prefix29_ref = prefix29[prefix29['case_len']>=36]
    prefix29_ref = prefix29_ref.reset_index()
    prefix29_ref = prefix29_ref.drop("index",1)
    prefix30_ref = prefix30[prefix30['case_len']>=36]
    prefix30_ref = prefix30_ref.reset_index()
    prefix30_ref = prefix30_ref.drop("index",1)
    prefix31_ref = prefix31[prefix31['case_len']>=36]
    prefix31_ref = prefix31_ref.reset_index()
    prefix31_ref = prefix31_ref.drop("index",1)
    prefix32_ref = prefix32[prefix32['case_len']>=36]
    prefix32_ref = prefix32_ref.reset_index()
    prefix32_ref = prefix32_ref.drop("index",1)
    prefix33_ref = prefix33[prefix33['case_len']>=36]
    prefix33_ref = prefix33_ref.reset_index()
    prefix33_ref = prefix33_ref.drop("index",1)
    prefix34_ref = prefix34[prefix34['case_len']>=36]
    prefix34_ref = prefix34_ref.reset_index()
    prefix34_ref = prefix34_ref.drop("index",1)
    prefix35_ref = prefix35[prefix35['case_len']>=36]
    prefix35_ref = prefix35_ref.reset_index()
    prefix35_ref = prefix35_ref.drop("index",1)
    prefix36 = data_dummy_prefix[data_dummy_prefix['prefix']==36]
    prefix36 = prefix36.reset_index()
    prefix36 = prefix36.drop("index",1)
    print(len(prefix36))
    print("Processing columns with prefix 36...")
    def max_prefix_40_step36(iter):
        for col in prefix36:
            if col[len(col)-10:len(col)] == "prefixlen1":
                prefix36[col] = prefix1_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen2":
                prefix36[col] = prefix2_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen3":
                prefix36[col] = prefix3_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen4":
                prefix36[col] = prefix4_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen5":
                prefix36[col] = prefix5_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen6":
                prefix36[col] = prefix6_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen7":
                prefix36[col] = prefix7_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen8":
                prefix36[col] = prefix8_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen9":
                prefix36[col] = prefix9_ref[col[0:len(col)-11]]
            if col[len(col)-11:len(col)] == "prefixlen10":
                prefix36[col] = prefix10_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen11":
                prefix36[col] = prefix11_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen12":
                prefix36[col] = prefix12_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen13":
                prefix36[col] = prefix13_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen14":
                prefix36[col] = prefix14_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen15":
                prefix36[col] = prefix15_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen16":
                prefix36[col] = prefix16_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen17":
                prefix36[col] = prefix17_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen18":
                prefix36[col] = prefix18_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen19":
                prefix36[col] = prefix19_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen20":
                prefix36[col] = prefix20_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen21":
                prefix36[col] = prefix21_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen22":
                prefix36[col] = prefix22_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen23":
                prefix36[col] = prefix23_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen24":
                prefix36[col] = prefix24_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen25":
                prefix36[col] = prefix25_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen26":
                prefix36[col] = prefix26_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen27":
                prefix36[col] = prefix27_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen28":
                prefix36[col] = prefix28_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen29":
                prefix36[col] = prefix29_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen30":
                prefix36[col] = prefix30_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen31":
                prefix36[col] = prefix31_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen32":
                prefix36[col] = prefix32_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen33":
                prefix36[col] = prefix33_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen34":
                prefix36[col] = prefix34_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen35":
                prefix36[col] = prefix35_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen36":
                prefix36[col] = prefix36[col[0:len(col)-12]]
        return prefix36
    if __name__ == '__main__':
        pool = multiprocessing.Pool(processes=num_cpu_cores)
        prefix36 = pool.map(max_prefix_40_step36, [0])[0]
        pool.close()
        pool.join()

    prefix1_ref = prefix1[prefix1['case_len']>=37]
    prefix1_ref = prefix1_ref.reset_index()
    prefix1_ref = prefix1_ref.drop("index",1)
    prefix2_ref = prefix2[prefix2['case_len']>=37]
    prefix2_ref = prefix2_ref.reset_index()
    prefix2_ref = prefix2_ref.drop("index",1)
    prefix3_ref = prefix3[prefix3['case_len']>=37]
    prefix3_ref = prefix3_ref.reset_index()
    prefix3_ref = prefix3_ref.drop("index",1)
    prefix4_ref = prefix4[prefix4['case_len']>=37]
    prefix4_ref = prefix4_ref.reset_index()
    prefix4_ref = prefix4_ref.drop("index",1)
    prefix5_ref = prefix5[prefix5['case_len']>=37]
    prefix5_ref = prefix5_ref.reset_index()
    prefix5_ref = prefix5_ref.drop("index",1)
    prefix6_ref = prefix6[prefix6['case_len']>=37]
    prefix6_ref = prefix6_ref.reset_index()
    prefix6_ref = prefix6_ref.drop("index",1)
    prefix7_ref = prefix7[prefix7['case_len']>=37]
    prefix7_ref = prefix7_ref.reset_index()
    prefix7_ref = prefix7_ref.drop("index",1)
    prefix8_ref = prefix8[prefix8['case_len']>=37]
    prefix8_ref = prefix8_ref.reset_index()
    prefix8_ref = prefix8_ref.drop("index",1)
    prefix9_ref = prefix9[prefix9['case_len']>=37]
    prefix9_ref = prefix9_ref.reset_index()
    prefix9_ref = prefix9_ref.drop("index",1)
    prefix10_ref = prefix10[prefix10['case_len']>=37]
    prefix10_ref = prefix10_ref.reset_index()
    prefix10_ref = prefix10_ref.drop("index",1)
    prefix11_ref = prefix11[prefix11['case_len']>=37]
    prefix11_ref = prefix11_ref.reset_index()
    prefix11_ref = prefix11_ref.drop("index",1)
    prefix12_ref = prefix12[prefix12['case_len']>=37]
    prefix12_ref = prefix12_ref.reset_index()
    prefix12_ref = prefix12_ref.drop("index",1)
    prefix13_ref = prefix13[prefix13['case_len']>=37]
    prefix13_ref = prefix13_ref.reset_index()
    prefix13_ref = prefix13_ref.drop("index",1)
    prefix14_ref = prefix14[prefix14['case_len']>=37]
    prefix14_ref = prefix14_ref.reset_index()
    prefix14_ref = prefix14_ref.drop("index",1)
    prefix15_ref = prefix15[prefix15['case_len']>=37]
    prefix15_ref = prefix15_ref.reset_index()
    prefix15_ref = prefix15_ref.drop("index",1)
    prefix16_ref = prefix16[prefix16['case_len']>=37]
    prefix16_ref = prefix16_ref.reset_index()
    prefix16_ref = prefix16_ref.drop("index",1)
    prefix17_ref = prefix17[prefix17['case_len']>=37]
    prefix17_ref = prefix17_ref.reset_index()
    prefix17_ref = prefix17_ref.drop("index",1)
    prefix18_ref = prefix18[prefix18['case_len']>=37]
    prefix18_ref = prefix18_ref.reset_index()
    prefix18_ref = prefix18_ref.drop("index",1)
    prefix19_ref = prefix19[prefix19['case_len']>=37]
    prefix19_ref = prefix19_ref.reset_index()
    prefix19_ref = prefix19_ref.drop("index",1)
    prefix20_ref = prefix20[prefix20['case_len']>=37]
    prefix20_ref = prefix20_ref.reset_index()
    prefix20_ref = prefix20_ref.drop("index",1)
    prefix21_ref = prefix21[prefix21['case_len']>=37]
    prefix21_ref = prefix21_ref.reset_index()
    prefix21_ref = prefix21_ref.drop("index",1)
    prefix22_ref = prefix22[prefix22['case_len']>=37]
    prefix22_ref = prefix22_ref.reset_index()
    prefix22_ref = prefix22_ref.drop("index",1)
    prefix23_ref = prefix23[prefix23['case_len']>=37]
    prefix23_ref = prefix23_ref.reset_index()
    prefix23_ref = prefix23_ref.drop("index",1)
    prefix24_ref = prefix24[prefix24['case_len']>=37]
    prefix24_ref = prefix24_ref.reset_index()
    prefix24_ref = prefix24_ref.drop("index",1)
    prefix25_ref = prefix25[prefix25['case_len']>=37]
    prefix25_ref = prefix25_ref.reset_index()
    prefix25_ref = prefix25_ref.drop("index",1)
    prefix26_ref = prefix26[prefix26['case_len']>=37]
    prefix26_ref = prefix26_ref.reset_index()
    prefix26_ref = prefix26_ref.drop("index",1)
    prefix27_ref = prefix27[prefix27['case_len']>=37]
    prefix27_ref = prefix27_ref.reset_index()
    prefix27_ref = prefix27_ref.drop("index",1)
    prefix28_ref = prefix28[prefix28['case_len']>=37]
    prefix28_ref = prefix28_ref.reset_index()
    prefix28_ref = prefix28_ref.drop("index",1)
    prefix29_ref = prefix29[prefix29['case_len']>=37]
    prefix29_ref = prefix29_ref.reset_index()
    prefix29_ref = prefix29_ref.drop("index",1)
    prefix30_ref = prefix30[prefix30['case_len']>=37]
    prefix30_ref = prefix30_ref.reset_index()
    prefix30_ref = prefix30_ref.drop("index",1)
    prefix31_ref = prefix31[prefix31['case_len']>=37]
    prefix31_ref = prefix31_ref.reset_index()
    prefix31_ref = prefix31_ref.drop("index",1)
    prefix32_ref = prefix32[prefix32['case_len']>=37]
    prefix32_ref = prefix32_ref.reset_index()
    prefix32_ref = prefix32_ref.drop("index",1)
    prefix33_ref = prefix33[prefix33['case_len']>=37]
    prefix33_ref = prefix33_ref.reset_index()
    prefix33_ref = prefix33_ref.drop("index",1)
    prefix34_ref = prefix34[prefix34['case_len']>=37]
    prefix34_ref = prefix34_ref.reset_index()
    prefix34_ref = prefix34_ref.drop("index",1)
    prefix35_ref = prefix35[prefix35['case_len']>=37]
    prefix35_ref = prefix35_ref.reset_index()
    prefix35_ref = prefix35_ref.drop("index",1)
    prefix36_ref = prefix36[prefix36['case_len']>=37]
    prefix36_ref = prefix36_ref.reset_index()
    prefix36_ref = prefix36_ref.drop("index",1)
    prefix37 = data_dummy_prefix[data_dummy_prefix['prefix']==37]
    prefix37 = prefix37.reset_index()
    prefix37 = prefix37.drop("index",1)
    print(len(prefix37))
    print("Processing columns with prefix 37...")
    def max_prefix_40_step37(iter):
        for col in prefix37:
            if col[len(col)-10:len(col)] == "prefixlen1":
                prefix37[col] = prefix1_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen2":
                prefix37[col] = prefix2_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen3":
                prefix37[col] = prefix3_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen4":
                prefix37[col] = prefix4_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen5":
                prefix37[col] = prefix5_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen6":
                prefix37[col] = prefix6_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen7":
                prefix37[col] = prefix7_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen8":
                prefix37[col] = prefix8_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen9":
                prefix37[col] = prefix9_ref[col[0:len(col)-11]]
            if col[len(col)-11:len(col)] == "prefixlen10":
                prefix37[col] = prefix10_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen11":
                prefix37[col] = prefix11_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen12":
                prefix37[col] = prefix12_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen13":
                prefix37[col] = prefix13_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen14":
                prefix37[col] = prefix14_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen15":
                prefix37[col] = prefix15_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen16":
                prefix37[col] = prefix16_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen17":
                prefix37[col] = prefix17_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen18":
                prefix37[col] = prefix18_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen19":
                prefix37[col] = prefix19_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen20":
                prefix37[col] = prefix20_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen21":
                prefix37[col] = prefix21_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen22":
                prefix37[col] = prefix22_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen23":
                prefix37[col] = prefix23_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen24":
                prefix37[col] = prefix24_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen25":
                prefix37[col] = prefix25_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen26":
                prefix37[col] = prefix26_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen27":
                prefix37[col] = prefix27_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen28":
                prefix37[col] = prefix28_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen29":
                prefix37[col] = prefix29_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen30":
                prefix37[col] = prefix30_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen31":
                prefix37[col] = prefix31_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen32":
                prefix37[col] = prefix32_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen33":
                prefix37[col] = prefix33_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen34":
                prefix37[col] = prefix34_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen35":
                prefix37[col] = prefix35_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen36":
                prefix37[col] = prefix36_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen37":
                prefix37[col] = prefix37[col[0:len(col)-12]]
        return prefix37
    if __name__ == '__main__':
        pool = multiprocessing.Pool(processes=num_cpu_cores)
        prefix37 = pool.map(max_prefix_40_step37, [0])[0]
        pool.close()
        pool.join()

    prefix1_ref = prefix1[prefix1['case_len']>=38]
    prefix1_ref = prefix1_ref.reset_index()
    prefix1_ref = prefix1_ref.drop("index",1)
    prefix2_ref = prefix2[prefix2['case_len']>=38]
    prefix2_ref = prefix2_ref.reset_index()
    prefix2_ref = prefix2_ref.drop("index",1)
    prefix3_ref = prefix3[prefix3['case_len']>=38]
    prefix3_ref = prefix3_ref.reset_index()
    prefix3_ref = prefix3_ref.drop("index",1)
    prefix4_ref = prefix4[prefix4['case_len']>=38]
    prefix4_ref = prefix4_ref.reset_index()
    prefix4_ref = prefix4_ref.drop("index",1)
    prefix5_ref = prefix5[prefix5['case_len']>=38]
    prefix5_ref = prefix5_ref.reset_index()
    prefix5_ref = prefix5_ref.drop("index",1)
    prefix6_ref = prefix6[prefix6['case_len']>=38]
    prefix6_ref = prefix6_ref.reset_index()
    prefix6_ref = prefix6_ref.drop("index",1)
    prefix7_ref = prefix7[prefix7['case_len']>=38]
    prefix7_ref = prefix7_ref.reset_index()
    prefix7_ref = prefix7_ref.drop("index",1)
    prefix8_ref = prefix8[prefix8['case_len']>=38]
    prefix8_ref = prefix8_ref.reset_index()
    prefix8_ref = prefix8_ref.drop("index",1)
    prefix9_ref = prefix9[prefix9['case_len']>=38]
    prefix9_ref = prefix9_ref.reset_index()
    prefix9_ref = prefix9_ref.drop("index",1)
    prefix10_ref = prefix10[prefix10['case_len']>=38]
    prefix10_ref = prefix10_ref.reset_index()
    prefix10_ref = prefix10_ref.drop("index",1)
    prefix11_ref = prefix11[prefix11['case_len']>=38]
    prefix11_ref = prefix11_ref.reset_index()
    prefix11_ref = prefix11_ref.drop("index",1)
    prefix12_ref = prefix12[prefix12['case_len']>=38]
    prefix12_ref = prefix12_ref.reset_index()
    prefix12_ref = prefix12_ref.drop("index",1)
    prefix13_ref = prefix13[prefix13['case_len']>=38]
    prefix13_ref = prefix13_ref.reset_index()
    prefix13_ref = prefix13_ref.drop("index",1)
    prefix14_ref = prefix14[prefix14['case_len']>=38]
    prefix14_ref = prefix14_ref.reset_index()
    prefix14_ref = prefix14_ref.drop("index",1)
    prefix15_ref = prefix15[prefix15['case_len']>=38]
    prefix15_ref = prefix15_ref.reset_index()
    prefix15_ref = prefix15_ref.drop("index",1)
    prefix16_ref = prefix16[prefix16['case_len']>=38]
    prefix16_ref = prefix16_ref.reset_index()
    prefix16_ref = prefix16_ref.drop("index",1)
    prefix17_ref = prefix17[prefix17['case_len']>=38]
    prefix17_ref = prefix17_ref.reset_index()
    prefix17_ref = prefix17_ref.drop("index",1)
    prefix18_ref = prefix18[prefix18['case_len']>=38]
    prefix18_ref = prefix18_ref.reset_index()
    prefix18_ref = prefix18_ref.drop("index",1)
    prefix19_ref = prefix19[prefix19['case_len']>=38]
    prefix19_ref = prefix19_ref.reset_index()
    prefix19_ref = prefix19_ref.drop("index",1)
    prefix20_ref = prefix20[prefix20['case_len']>=38]
    prefix20_ref = prefix20_ref.reset_index()
    prefix20_ref = prefix20_ref.drop("index",1)
    prefix21_ref = prefix21[prefix21['case_len']>=38]
    prefix21_ref = prefix21_ref.reset_index()
    prefix21_ref = prefix21_ref.drop("index",1)
    prefix22_ref = prefix22[prefix22['case_len']>=38]
    prefix22_ref = prefix22_ref.reset_index()
    prefix22_ref = prefix22_ref.drop("index",1)
    prefix23_ref = prefix23[prefix23['case_len']>=38]
    prefix23_ref = prefix23_ref.reset_index()
    prefix23_ref = prefix23_ref.drop("index",1)
    prefix24_ref = prefix24[prefix24['case_len']>=38]
    prefix24_ref = prefix24_ref.reset_index()
    prefix24_ref = prefix24_ref.drop("index",1)
    prefix25_ref = prefix25[prefix25['case_len']>=38]
    prefix25_ref = prefix25_ref.reset_index()
    prefix25_ref = prefix25_ref.drop("index",1)
    prefix26_ref = prefix26[prefix26['case_len']>=38]
    prefix26_ref = prefix26_ref.reset_index()
    prefix26_ref = prefix26_ref.drop("index",1)
    prefix27_ref = prefix27[prefix27['case_len']>=38]
    prefix27_ref = prefix27_ref.reset_index()
    prefix27_ref = prefix27_ref.drop("index",1)
    prefix28_ref = prefix28[prefix28['case_len']>=38]
    prefix28_ref = prefix28_ref.reset_index()
    prefix28_ref = prefix28_ref.drop("index",1)
    prefix29_ref = prefix29[prefix29['case_len']>=38]
    prefix29_ref = prefix29_ref.reset_index()
    prefix29_ref = prefix29_ref.drop("index",1)
    prefix30_ref = prefix30[prefix30['case_len']>=38]
    prefix30_ref = prefix30_ref.reset_index()
    prefix30_ref = prefix30_ref.drop("index",1)
    prefix31_ref = prefix31[prefix31['case_len']>=38]
    prefix31_ref = prefix31_ref.reset_index()
    prefix31_ref = prefix31_ref.drop("index",1)
    prefix32_ref = prefix32[prefix32['case_len']>=38]
    prefix32_ref = prefix32_ref.reset_index()
    prefix32_ref = prefix32_ref.drop("index",1)
    prefix33_ref = prefix33[prefix33['case_len']>=38]
    prefix33_ref = prefix33_ref.reset_index()
    prefix33_ref = prefix33_ref.drop("index",1)
    prefix34_ref = prefix34[prefix34['case_len']>=38]
    prefix34_ref = prefix34_ref.reset_index()
    prefix34_ref = prefix34_ref.drop("index",1)
    prefix35_ref = prefix35[prefix35['case_len']>=38]
    prefix35_ref = prefix35_ref.reset_index()
    prefix35_ref = prefix35_ref.drop("index",1)
    prefix36_ref = prefix36[prefix36['case_len']>=38]
    prefix36_ref = prefix36_ref.reset_index()
    prefix36_ref = prefix36_ref.drop("index",1)
    prefix37_ref = prefix37[prefix37['case_len']>=38]
    prefix37_ref = prefix37_ref.reset_index()
    prefix37_ref = prefix37_ref.drop("index",1)
    prefix38 = data_dummy_prefix[data_dummy_prefix['prefix']==38]
    prefix38 = prefix38.reset_index()
    prefix38 = prefix38.drop("index",1)
    print(len(prefix38))
    print("Processing columns with prefix 38...")
    def max_prefix_40_step38(iter):
        for col in prefix38:
            if col[len(col)-10:len(col)] == "prefixlen1":
                prefix38[col] = prefix1_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen2":
                prefix38[col] = prefix2_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen3":
                prefix38[col] = prefix3_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen4":
                prefix38[col] = prefix4_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen5":
                prefix38[col] = prefix5_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen6":
                prefix38[col] = prefix6_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen7":
                prefix38[col] = prefix7_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen8":
                prefix38[col] = prefix8_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen9":
                prefix38[col] = prefix9_ref[col[0:len(col)-11]]
            if col[len(col)-11:len(col)] == "prefixlen10":
                prefix38[col] = prefix10_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen11":
                prefix38[col] = prefix11_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen12":
                prefix38[col] = prefix12_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen13":
                prefix38[col] = prefix13_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen14":
                prefix38[col] = prefix14_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen15":
                prefix38[col] = prefix15_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen16":
                prefix38[col] = prefix16_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen17":
                prefix38[col] = prefix17_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen18":
                prefix38[col] = prefix18_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen19":
                prefix38[col] = prefix19_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen20":
                prefix38[col] = prefix20_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen21":
                prefix38[col] = prefix21_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen22":
                prefix38[col] = prefix22_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen23":
                prefix38[col] = prefix23_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen24":
                prefix38[col] = prefix24_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen25":
                prefix38[col] = prefix25_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen26":
                prefix38[col] = prefix26_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen27":
                prefix38[col] = prefix27_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen28":
                prefix38[col] = prefix28_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen29":
                prefix38[col] = prefix29_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen30":
                prefix38[col] = prefix30_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen31":
                prefix38[col] = prefix31_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen32":
                prefix38[col] = prefix32_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen33":
                prefix38[col] = prefix33_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen34":
                prefix38[col] = prefix34_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen35":
                prefix38[col] = prefix35_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen36":
                prefix38[col] = prefix36_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen37":
                prefix38[col] = prefix37_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen38":
                prefix38[col] = prefix38[col[0:len(col)-12]]
        return prefix38
    if __name__ == '__main__':
        pool = multiprocessing.Pool(processes=num_cpu_cores)
        prefix38 = pool.map(max_prefix_40_step38, [0])[0]
        pool.close()
        pool.join()

    prefix1_ref = prefix1[prefix1['case_len']>=39]
    prefix1_ref = prefix1_ref.reset_index()
    prefix1_ref = prefix1_ref.drop("index",1)
    prefix2_ref = prefix2[prefix2['case_len']>=39]
    prefix2_ref = prefix2_ref.reset_index()
    prefix2_ref = prefix2_ref.drop("index",1)
    prefix3_ref = prefix3[prefix3['case_len']>=39]
    prefix3_ref = prefix3_ref.reset_index()
    prefix3_ref = prefix3_ref.drop("index",1)
    prefix4_ref = prefix4[prefix4['case_len']>=39]
    prefix4_ref = prefix4_ref.reset_index()
    prefix4_ref = prefix4_ref.drop("index",1)
    prefix5_ref = prefix5[prefix5['case_len']>=39]
    prefix5_ref = prefix5_ref.reset_index()
    prefix5_ref = prefix5_ref.drop("index",1)
    prefix6_ref = prefix6[prefix6['case_len']>=39]
    prefix6_ref = prefix6_ref.reset_index()
    prefix6_ref = prefix6_ref.drop("index",1)
    prefix7_ref = prefix7[prefix7['case_len']>=39]
    prefix7_ref = prefix7_ref.reset_index()
    prefix7_ref = prefix7_ref.drop("index",1)
    prefix8_ref = prefix8[prefix8['case_len']>=39]
    prefix8_ref = prefix8_ref.reset_index()
    prefix8_ref = prefix8_ref.drop("index",1)
    prefix9_ref = prefix9[prefix9['case_len']>=39]
    prefix9_ref = prefix9_ref.reset_index()
    prefix9_ref = prefix9_ref.drop("index",1)
    prefix10_ref = prefix10[prefix10['case_len']>=39]
    prefix10_ref = prefix10_ref.reset_index()
    prefix10_ref = prefix10_ref.drop("index",1)
    prefix11_ref = prefix11[prefix11['case_len']>=39]
    prefix11_ref = prefix11_ref.reset_index()
    prefix11_ref = prefix11_ref.drop("index",1)
    prefix12_ref = prefix12[prefix12['case_len']>=39]
    prefix12_ref = prefix12_ref.reset_index()
    prefix12_ref = prefix12_ref.drop("index",1)
    prefix13_ref = prefix13[prefix13['case_len']>=39]
    prefix13_ref = prefix13_ref.reset_index()
    prefix13_ref = prefix13_ref.drop("index",1)
    prefix14_ref = prefix14[prefix14['case_len']>=39]
    prefix14_ref = prefix14_ref.reset_index()
    prefix14_ref = prefix14_ref.drop("index",1)
    prefix15_ref = prefix15[prefix15['case_len']>=39]
    prefix15_ref = prefix15_ref.reset_index()
    prefix15_ref = prefix15_ref.drop("index",1)
    prefix16_ref = prefix16[prefix16['case_len']>=39]
    prefix16_ref = prefix16_ref.reset_index()
    prefix16_ref = prefix16_ref.drop("index",1)
    prefix17_ref = prefix17[prefix17['case_len']>=39]
    prefix17_ref = prefix17_ref.reset_index()
    prefix17_ref = prefix17_ref.drop("index",1)
    prefix18_ref = prefix18[prefix18['case_len']>=39]
    prefix18_ref = prefix18_ref.reset_index()
    prefix18_ref = prefix18_ref.drop("index",1)
    prefix19_ref = prefix19[prefix19['case_len']>=39]
    prefix19_ref = prefix19_ref.reset_index()
    prefix19_ref = prefix19_ref.drop("index",1)
    prefix20_ref = prefix20[prefix20['case_len']>=39]
    prefix20_ref = prefix20_ref.reset_index()
    prefix20_ref = prefix20_ref.drop("index",1)
    prefix21_ref = prefix21[prefix21['case_len']>=39]
    prefix21_ref = prefix21_ref.reset_index()
    prefix21_ref = prefix21_ref.drop("index",1)
    prefix22_ref = prefix22[prefix22['case_len']>=39]
    prefix22_ref = prefix22_ref.reset_index()
    prefix22_ref = prefix22_ref.drop("index",1)
    prefix23_ref = prefix23[prefix23['case_len']>=39]
    prefix23_ref = prefix23_ref.reset_index()
    prefix23_ref = prefix23_ref.drop("index",1)
    prefix24_ref = prefix24[prefix24['case_len']>=39]
    prefix24_ref = prefix24_ref.reset_index()
    prefix24_ref = prefix24_ref.drop("index",1)
    prefix25_ref = prefix25[prefix25['case_len']>=39]
    prefix25_ref = prefix25_ref.reset_index()
    prefix25_ref = prefix25_ref.drop("index",1)
    prefix26_ref = prefix26[prefix26['case_len']>=39]
    prefix26_ref = prefix26_ref.reset_index()
    prefix26_ref = prefix26_ref.drop("index",1)
    prefix27_ref = prefix27[prefix27['case_len']>=39]
    prefix27_ref = prefix27_ref.reset_index()
    prefix27_ref = prefix27_ref.drop("index",1)
    prefix28_ref = prefix28[prefix28['case_len']>=39]
    prefix28_ref = prefix28_ref.reset_index()
    prefix28_ref = prefix28_ref.drop("index",1)
    prefix29_ref = prefix29[prefix29['case_len']>=39]
    prefix29_ref = prefix29_ref.reset_index()
    prefix29_ref = prefix29_ref.drop("index",1)
    prefix30_ref = prefix30[prefix30['case_len']>=39]
    prefix30_ref = prefix30_ref.reset_index()
    prefix30_ref = prefix30_ref.drop("index",1)
    prefix31_ref = prefix31[prefix31['case_len']>=39]
    prefix31_ref = prefix31_ref.reset_index()
    prefix31_ref = prefix31_ref.drop("index",1)
    prefix32_ref = prefix32[prefix32['case_len']>=39]
    prefix32_ref = prefix32_ref.reset_index()
    prefix32_ref = prefix32_ref.drop("index",1)
    prefix33_ref = prefix33[prefix33['case_len']>=39]
    prefix33_ref = prefix33_ref.reset_index()
    prefix33_ref = prefix33_ref.drop("index",1)
    prefix34_ref = prefix34[prefix34['case_len']>=39]
    prefix34_ref = prefix34_ref.reset_index()
    prefix34_ref = prefix34_ref.drop("index",1)
    prefix35_ref = prefix35[prefix35['case_len']>=39]
    prefix35_ref = prefix35_ref.reset_index()
    prefix35_ref = prefix35_ref.drop("index",1)
    prefix36_ref = prefix36[prefix36['case_len']>=39]
    prefix36_ref = prefix36_ref.reset_index()
    prefix36_ref = prefix36_ref.drop("index",1)
    prefix37_ref = prefix37[prefix37['case_len']>=39]
    prefix37_ref = prefix37_ref.reset_index()
    prefix37_ref = prefix37_ref.drop("index",1)
    prefix38_ref = prefix38[prefix38['case_len']>=39]
    prefix38_ref = prefix38_ref.reset_index()
    prefix38_ref = prefix38_ref.drop("index",1)
    prefix39 = data_dummy_prefix[data_dummy_prefix['prefix']==39]
    prefix39 = prefix39.reset_index()
    prefix39 = prefix39.drop("index",1)
    print(len(prefix39))
    print("Processing columns with prefix 39...")
    def max_prefix_40_step39(iter):
        for col in prefix39:
            if col[len(col)-10:len(col)] == "prefixlen1":
                prefix39[col] = prefix1_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen2":
                prefix39[col] = prefix2_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen3":
                prefix39[col] = prefix3_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen4":
                prefix39[col] = prefix4_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen5":
                prefix39[col] = prefix5_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen6":
                prefix39[col] = prefix6_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen7":
                prefix39[col] = prefix7_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen8":
                prefix39[col] = prefix8_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen9":
                prefix39[col] = prefix9_ref[col[0:len(col)-11]]
            if col[len(col)-11:len(col)] == "prefixlen10":
                prefix39[col] = prefix10_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen11":
                prefix39[col] = prefix11_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen12":
                prefix39[col] = prefix12_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen13":
                prefix39[col] = prefix13_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen14":
                prefix39[col] = prefix14_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen15":
                prefix39[col] = prefix15_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen16":
                prefix39[col] = prefix16_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen17":
                prefix39[col] = prefix17_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen18":
                prefix39[col] = prefix18_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen19":
                prefix39[col] = prefix19_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen20":
                prefix39[col] = prefix20_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen21":
                prefix39[col] = prefix21_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen22":
                prefix39[col] = prefix22_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen23":
                prefix39[col] = prefix23_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen24":
                prefix39[col] = prefix24_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen25":
                prefix39[col] = prefix25_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen26":
                prefix39[col] = prefix26_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen27":
                prefix39[col] = prefix27_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen28":
                prefix39[col] = prefix28_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen29":
                prefix39[col] = prefix29_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen30":
                prefix39[col] = prefix30_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen31":
                prefix39[col] = prefix31_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen32":
                prefix39[col] = prefix32_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen33":
                prefix39[col] = prefix33_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen34":
                prefix39[col] = prefix34_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen35":
                prefix39[col] = prefix35_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen36":
                prefix39[col] = prefix36_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen37":
                prefix39[col] = prefix37_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen38":
                prefix39[col] = prefix38_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen39":
                prefix39[col] = prefix39[col[0:len(col)-12]]
        return prefix39
    if __name__ == '__main__':
        pool = multiprocessing.Pool(processes=num_cpu_cores)
        prefix39 = pool.map(max_prefix_40_step39, [0])[0]
        pool.close()
        pool.join()

    prefix1_ref = prefix1[prefix1['case_len']>=40]
    prefix1_ref = prefix1_ref.reset_index()
    prefix1_ref = prefix1_ref.drop("index",1)
    prefix2_ref = prefix2[prefix2['case_len']>=40]
    prefix2_ref = prefix2_ref.reset_index()
    prefix2_ref = prefix2_ref.drop("index",1)
    prefix3_ref = prefix3[prefix3['case_len']>=40]
    prefix3_ref = prefix3_ref.reset_index()
    prefix3_ref = prefix3_ref.drop("index",1)
    prefix4_ref = prefix4[prefix4['case_len']>=40]
    prefix4_ref = prefix4_ref.reset_index()
    prefix4_ref = prefix4_ref.drop("index",1)
    prefix5_ref = prefix5[prefix5['case_len']>=40]
    prefix5_ref = prefix5_ref.reset_index()
    prefix5_ref = prefix5_ref.drop("index",1)
    prefix6_ref = prefix6[prefix6['case_len']>=40]
    prefix6_ref = prefix6_ref.reset_index()
    prefix6_ref = prefix6_ref.drop("index",1)
    prefix7_ref = prefix7[prefix7['case_len']>=40]
    prefix7_ref = prefix7_ref.reset_index()
    prefix7_ref = prefix7_ref.drop("index",1)
    prefix8_ref = prefix8[prefix8['case_len']>=40]
    prefix8_ref = prefix8_ref.reset_index()
    prefix8_ref = prefix8_ref.drop("index",1)
    prefix9_ref = prefix9[prefix9['case_len']>=40]
    prefix9_ref = prefix9_ref.reset_index()
    prefix9_ref = prefix9_ref.drop("index",1)
    prefix10_ref = prefix10[prefix10['case_len']>=40]
    prefix10_ref = prefix10_ref.reset_index()
    prefix10_ref = prefix10_ref.drop("index",1)
    prefix11_ref = prefix11[prefix11['case_len']>=40]
    prefix11_ref = prefix11_ref.reset_index()
    prefix11_ref = prefix11_ref.drop("index",1)
    prefix12_ref = prefix12[prefix12['case_len']>=40]
    prefix12_ref = prefix12_ref.reset_index()
    prefix12_ref = prefix12_ref.drop("index",1)
    prefix13_ref = prefix13[prefix13['case_len']>=40]
    prefix13_ref = prefix13_ref.reset_index()
    prefix13_ref = prefix13_ref.drop("index",1)
    prefix14_ref = prefix14[prefix14['case_len']>=40]
    prefix14_ref = prefix14_ref.reset_index()
    prefix14_ref = prefix14_ref.drop("index",1)
    prefix15_ref = prefix15[prefix15['case_len']>=40]
    prefix15_ref = prefix15_ref.reset_index()
    prefix15_ref = prefix15_ref.drop("index",1)
    prefix16_ref = prefix16[prefix16['case_len']>=40]
    prefix16_ref = prefix16_ref.reset_index()
    prefix16_ref = prefix16_ref.drop("index",1)
    prefix17_ref = prefix17[prefix17['case_len']>=40]
    prefix17_ref = prefix17_ref.reset_index()
    prefix17_ref = prefix17_ref.drop("index",1)
    prefix18_ref = prefix18[prefix18['case_len']>=40]
    prefix18_ref = prefix18_ref.reset_index()
    prefix18_ref = prefix18_ref.drop("index",1)
    prefix19_ref = prefix19[prefix19['case_len']>=40]
    prefix19_ref = prefix19_ref.reset_index()
    prefix19_ref = prefix19_ref.drop("index",1)
    prefix20_ref = prefix20[prefix20['case_len']>=40]
    prefix20_ref = prefix20_ref.reset_index()
    prefix20_ref = prefix20_ref.drop("index",1)
    prefix21_ref = prefix21[prefix21['case_len']>=40]
    prefix21_ref = prefix21_ref.reset_index()
    prefix21_ref = prefix21_ref.drop("index",1)
    prefix22_ref = prefix22[prefix22['case_len']>=40]
    prefix22_ref = prefix22_ref.reset_index()
    prefix22_ref = prefix22_ref.drop("index",1)
    prefix23_ref = prefix23[prefix23['case_len']>=40]
    prefix23_ref = prefix23_ref.reset_index()
    prefix23_ref = prefix23_ref.drop("index",1)
    prefix24_ref = prefix24[prefix24['case_len']>=40]
    prefix24_ref = prefix24_ref.reset_index()
    prefix24_ref = prefix24_ref.drop("index",1)
    prefix25_ref = prefix25[prefix25['case_len']>=40]
    prefix25_ref = prefix25_ref.reset_index()
    prefix25_ref = prefix25_ref.drop("index",1)
    prefix26_ref = prefix26[prefix26['case_len']>=40]
    prefix26_ref = prefix26_ref.reset_index()
    prefix26_ref = prefix26_ref.drop("index",1)
    prefix27_ref = prefix27[prefix27['case_len']>=40]
    prefix27_ref = prefix27_ref.reset_index()
    prefix27_ref = prefix27_ref.drop("index",1)
    prefix28_ref = prefix28[prefix28['case_len']>=40]
    prefix28_ref = prefix28_ref.reset_index()
    prefix28_ref = prefix28_ref.drop("index",1)
    prefix29_ref = prefix29[prefix29['case_len']>=40]
    prefix29_ref = prefix29_ref.reset_index()
    prefix29_ref = prefix29_ref.drop("index",1)
    prefix30_ref = prefix30[prefix30['case_len']>=40]
    prefix30_ref = prefix30_ref.reset_index()
    prefix30_ref = prefix30_ref.drop("index",1)
    prefix31_ref = prefix31[prefix31['case_len']>=40]
    prefix31_ref = prefix31_ref.reset_index()
    prefix31_ref = prefix31_ref.drop("index",1)
    prefix32_ref = prefix32[prefix32['case_len']>=40]
    prefix32_ref = prefix32_ref.reset_index()
    prefix32_ref = prefix32_ref.drop("index",1)
    prefix33_ref = prefix33[prefix33['case_len']>=40]
    prefix33_ref = prefix33_ref.reset_index()
    prefix33_ref = prefix33_ref.drop("index",1)
    prefix34_ref = prefix34[prefix34['case_len']>=40]
    prefix34_ref = prefix34_ref.reset_index()
    prefix34_ref = prefix34_ref.drop("index",1)
    prefix35_ref = prefix35[prefix35['case_len']>=40]
    prefix35_ref = prefix35_ref.reset_index()
    prefix35_ref = prefix35_ref.drop("index",1)
    prefix36_ref = prefix36[prefix36['case_len']>=40]
    prefix36_ref = prefix36_ref.reset_index()
    prefix36_ref = prefix36_ref.drop("index",1)
    prefix37_ref = prefix37[prefix37['case_len']>=40]
    prefix37_ref = prefix37_ref.reset_index()
    prefix37_ref = prefix37_ref.drop("index",1)
    prefix38_ref = prefix38[prefix38['case_len']>=40]
    prefix38_ref = prefix38_ref.reset_index()
    prefix38_ref = prefix38_ref.drop("index",1)
    prefix39_ref = prefix39[prefix39['case_len']>=40]
    prefix39_ref = prefix39_ref.reset_index()
    prefix39_ref = prefix39_ref.drop("index",1)
    prefix40 = data_dummy_prefix[data_dummy_prefix['prefix']==40]
    prefix40 = prefix40.reset_index()
    prefix40 = prefix40.drop("index",1)
    print(len(prefix40))
    print("Processing columns with prefix 40...")
    def max_prefix_40_step40(iter):
        for col in prefix40:
            if col[len(col)-10:len(col)] == "prefixlen1":
                prefix40[col] = prefix1_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen2":
                prefix40[col] = prefix2_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen3":
                prefix40[col] = prefix3_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen4":
                prefix40[col] = prefix4_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen5":
                prefix40[col] = prefix5_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen6":
                prefix40[col] = prefix6_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen7":
                prefix40[col] = prefix7_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen8":
                prefix40[col] = prefix8_ref[col[0:len(col)-11]]
            if col[len(col)-10:len(col)] == "prefixlen9":
                prefix40[col] = prefix9_ref[col[0:len(col)-11]]
            if col[len(col)-11:len(col)] == "prefixlen10":
                prefix40[col] = prefix10_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen11":
                prefix40[col] = prefix11_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen12":
                prefix40[col] = prefix12_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen13":
                prefix40[col] = prefix13_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen14":
                prefix40[col] = prefix14_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen15":
                prefix40[col] = prefix15_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen16":
                prefix40[col] = prefix16_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen17":
                prefix40[col] = prefix17_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen18":
                prefix40[col] = prefix18_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen19":
                prefix40[col] = prefix19_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen20":
                prefix40[col] = prefix20_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen21":
                prefix40[col] = prefix21_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen22":
                prefix40[col] = prefix22_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen23":
                prefix40[col] = prefix23_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen24":
                prefix40[col] = prefix24_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen25":
                prefix40[col] = prefix25_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen26":
                prefix40[col] = prefix26_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen27":
                prefix40[col] = prefix27_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen28":
                prefix40[col] = prefix28_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen29":
                prefix40[col] = prefix29_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen30":
                prefix40[col] = prefix30_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen31":
                prefix40[col] = prefix31_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen32":
                prefix40[col] = prefix32_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen33":
                prefix40[col] = prefix33_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen34":
                prefix40[col] = prefix34_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen35":
                prefix40[col] = prefix35_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen36":
                prefix40[col] = prefix36_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen37":
                prefix40[col] = prefix37_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen38":
                prefix40[col] = prefix38_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen39":
                prefix40[col] = prefix39_ref[col[0:len(col)-12]]
            if col[len(col)-11:len(col)] == "prefixlen40":
                prefix40[col] = prefix40[col[0:len(col)-12]]
        return prefix40
    if __name__ == '__main__':
        pool = multiprocessing.Pool(processes=num_cpu_cores)
        prefix40 = pool.map(max_prefix_40_step40, [0])[0]
        pool.close()
        pool.join()

    prefix1 = prefix1.reset_index()
    prefix1 = prefix1.drop("index",1)
    prefix2 = prefix2.reset_index()
    prefix2 = prefix2.drop("index",1)
    prefix3 = prefix3.reset_index()
    prefix3 = prefix3.drop("index",1)
    prefix4 = prefix4.reset_index()
    prefix4 = prefix4.drop("index",1)
    prefix5 = prefix5.reset_index()
    prefix5 = prefix5.drop("index",1)
    prefix6 = prefix6.reset_index()
    prefix6 = prefix6.drop("index",1)
    prefix7 = prefix7.reset_index()
    prefix7 = prefix7.drop("index",1)
    prefix8 = prefix8.reset_index()
    prefix8 = prefix8.drop("index",1)
    prefix9 = prefix9.reset_index()
    prefix9 = prefix9.drop("index",1)
    prefix10 = prefix10.reset_index()
    prefix10 = prefix10.drop("index",1)
    prefix11 = prefix11.reset_index()
    prefix11 = prefix11.drop("index",1)
    prefix12 = prefix12.reset_index()
    prefix12 = prefix12.drop("index",1)
    prefix13 = prefix13.reset_index()
    prefix13 = prefix13.drop("index",1)
    prefix14 = prefix14.reset_index()
    prefix14 = prefix14.drop("index",1)
    prefix15 = prefix15.reset_index()
    prefix15 = prefix15.drop("index",1)
    prefix16 = prefix16.reset_index()
    prefix16 = prefix16.drop("index",1)
    prefix17 = prefix17.reset_index()
    prefix17 = prefix17.drop("index",1)
    prefix18 = prefix18.reset_index()
    prefix18 = prefix18.drop("index",1)
    prefix19 = prefix19.reset_index()
    prefix19 = prefix19.drop("index",1)
    prefix20 = prefix20.reset_index()
    prefix20 = prefix20.drop("index",1)
    prefix21 = prefix21.reset_index()
    prefix21 = prefix21.drop("index",1)
    prefix22 = prefix22.reset_index()
    prefix22 = prefix22.drop("index",1)
    prefix23 = prefix23.reset_index()
    prefix23 = prefix23.drop("index",1)
    prefix24 = prefix24.reset_index()
    prefix24 = prefix24.drop("index",1)
    prefix25 = prefix25.reset_index()
    prefix25 = prefix25.drop("index",1)
    prefix26 = prefix26.reset_index()
    prefix26 = prefix26.drop("index",1)
    prefix27 = prefix27.reset_index()
    prefix27 = prefix27.drop("index",1)
    prefix28 = prefix28.reset_index()
    prefix28 = prefix28.drop("index",1)
    prefix29 = prefix29.reset_index()
    prefix29 = prefix29.drop("index",1)
    prefix30 = prefix30.reset_index()
    prefix30 = prefix30.drop("index",1)
    prefix31 = prefix31.reset_index()
    prefix31 = prefix31.drop("index",1)
    prefix32 = prefix32.reset_index()
    prefix32 = prefix32.drop("index",1)
    prefix33 = prefix33.reset_index()
    prefix33 = prefix33.drop("index",1)
    prefix34 = prefix34.reset_index()
    prefix34 = prefix34.drop("index",1)
    prefix35 = prefix35.reset_index()
    prefix35 = prefix35.drop("index",1)
    prefix36 = prefix36.reset_index()
    prefix36 = prefix36.drop("index",1)
    prefix37 = prefix37.reset_index()
    prefix37 = prefix37.drop("index",1)
    prefix38 = prefix38.reset_index()
    prefix38 = prefix38.drop("index",1)
    prefix39 = prefix39.reset_index()
    prefix39 = prefix39.drop("index",1)
    prefix40 = prefix40.reset_index()
    prefix40 = prefix40.drop("index",1)
    prefix_index_encoding = prefix1.append(prefix2)
    prefix_index_encoding = prefix_index_encoding.append(prefix3)
    prefix_index_encoding = prefix_index_encoding.append(prefix4)
    prefix_index_encoding = prefix_index_encoding.append(prefix5)
    prefix_index_encoding = prefix_index_encoding.append(prefix6)
    prefix_index_encoding = prefix_index_encoding.append(prefix7)
    prefix_index_encoding = prefix_index_encoding.append(prefix8)
    prefix_index_encoding = prefix_index_encoding.append(prefix9)
    prefix_index_encoding = prefix_index_encoding.append(prefix10)
    prefix_index_encoding = prefix_index_encoding.append(prefix11)
    prefix_index_encoding = prefix_index_encoding.append(prefix12)
    prefix_index_encoding = prefix_index_encoding.append(prefix13)
    prefix_index_encoding = prefix_index_encoding.append(prefix14)
    prefix_index_encoding = prefix_index_encoding.append(prefix15)
    prefix_index_encoding = prefix_index_encoding.append(prefix16)
    prefix_index_encoding = prefix_index_encoding.append(prefix17)
    prefix_index_encoding = prefix_index_encoding.append(prefix18)
    prefix_index_encoding = prefix_index_encoding.append(prefix19)
    prefix_index_encoding = prefix_index_encoding.append(prefix20)
    prefix_index_encoding = prefix_index_encoding.append(prefix21)
    prefix_index_encoding = prefix_index_encoding.append(prefix22)
    prefix_index_encoding = prefix_index_encoding.append(prefix23)
    prefix_index_encoding = prefix_index_encoding.append(prefix24)
    prefix_index_encoding = prefix_index_encoding.append(prefix25)
    prefix_index_encoding = prefix_index_encoding.append(prefix26)
    prefix_index_encoding = prefix_index_encoding.append(prefix27)
    prefix_index_encoding = prefix_index_encoding.append(prefix28)
    prefix_index_encoding = prefix_index_encoding.append(prefix29)
    prefix_index_encoding = prefix_index_encoding.append(prefix30)
    prefix_index_encoding = prefix_index_encoding.append(prefix31)
    prefix_index_encoding = prefix_index_encoding.append(prefix32)
    prefix_index_encoding = prefix_index_encoding.append(prefix33)
    prefix_index_encoding = prefix_index_encoding.append(prefix34)
    prefix_index_encoding = prefix_index_encoding.append(prefix35)
    prefix_index_encoding = prefix_index_encoding.append(prefix36)
    prefix_index_encoding = prefix_index_encoding.append(prefix37)
    prefix_index_encoding = prefix_index_encoding.append(prefix38)
    prefix_index_encoding = prefix_index_encoding.append(prefix39)
    prefix_index_encoding = prefix_index_encoding.append(prefix40)
    prefix_index_encoding = prefix_index_encoding.reset_index()
    prefix_index_encoding = prefix_index_encoding.drop("index",1)
    prefix_index_encoding = prefix_index_encoding.filter(regex="prefixlen")
    prefix_index_encoding.to_csv("index_encoded_data_" + dataset_name + "_" + str(max_prefix) + ".csv", index = False)



# Splitting dataset with "dataset without resource" and "dataset with resource"
if dataset_name == "BPIC11_f1_exp_prefix":
    data_dummy_prefix_yes_resource = prefix_index_encoding
    prefix_index_encoding_case_id = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("case_id")]
    prefix_index_encoding_activity = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Activity.code")]
    prefix_index_encoding_producer = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Producer.code")]
    prefix_index_encoding_timesincelastevent = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("timesincelastevent")]
    prefix_index_encoding_timesincecasestart = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("timesincecasestart")]
    prefix_index_encoding_timesincemidnight = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("timesincemidnight")]
    prefix_index_encoding_event_nr = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("event_nr")] # 'prefix' column and 'event_nr' column have same values here
    prefix_index_encoding_month = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("month")]
    prefix_index_encoding_weekday = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("weekday")]
    prefix_index_encoding_hour = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("hour")]
    prefix_index_encoding_open_cases = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("open_cases")]
    prefix_index_encoding_label = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("label_prefixlen1")] #Should be "label_prefixlen1" all the time
    prefix_index_encoding_Age = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Age")]
    prefix_index_encoding_Treatment_code = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Treatment.code")]
    prefix_index_encoding_Diagnosis_code = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Diagnosis.code")]
    prefix_index_encoding_Specialism_code_1 = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Specialism.code.1")]
    prefix_index_encoding_Section = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Section")]
    prefix_index_encoding_Specialism_code_2 = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Specialism.code.2")]
    prefix_index_encoding_group = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("group")]
    prefix_index_encoding_num_of_exec = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Number.of.executions")]
    data_dummy_prefix_no_resource = pd.concat([prefix_index_encoding_case_id, prefix_index_encoding_activity, prefix_index_encoding_producer, prefix_index_encoding_timesincelastevent, prefix_index_encoding_timesincecasestart, prefix_index_encoding_timesincemidnight, prefix_index_encoding_event_nr, prefix_index_encoding_month, prefix_index_encoding_weekday, prefix_index_encoding_hour, prefix_index_encoding_open_cases, prefix_index_encoding_label, prefix_index_encoding_Age, prefix_index_encoding_Treatment_code, prefix_index_encoding_Section, prefix_index_encoding_Specialism_code_1, prefix_index_encoding_Specialism_code_2, prefix_index_encoding_group, prefix_index_encoding_num_of_exec], axis = 1)

if dataset_name == "BPIC11_f2_exp_prefix":
    data_dummy_prefix_yes_resource = prefix_index_encoding
    prefix_index_encoding_case_id = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("case_id")]
    prefix_index_encoding_activity = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Activity.code")]
    prefix_index_encoding_producer = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Producer.code")]
    prefix_index_encoding_timesincelastevent = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("timesincelastevent")]
    prefix_index_encoding_timesincecasestart = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("timesincecasestart")]
    prefix_index_encoding_timesincemidnight = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("timesincemidnight")]
    prefix_index_encoding_event_nr = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("event_nr")] # 'prefix' column and 'event_nr' column have same values here
    prefix_index_encoding_month = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("month")]
    prefix_index_encoding_weekday = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("weekday")]
    prefix_index_encoding_hour = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("hour")]
    prefix_index_encoding_open_cases = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("open_cases")]
    prefix_index_encoding_label = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("label_prefixlen1")] #Should be "label_prefixlen1" all the time
    prefix_index_encoding_Age = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Age")]
    prefix_index_encoding_Treatment_code = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Treatment.code")]
    prefix_index_encoding_Diagnosis_code = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Diagnosis.code")]
    prefix_index_encoding_Specialism_code_1 = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Specialism.code.1")]
    prefix_index_encoding_Section = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Section")]
    prefix_index_encoding_Specialism_code_2 = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Specialism.code.2")]
    prefix_index_encoding_group = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("group")]
    prefix_index_encoding_num_of_exec = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Number.of.executions")]
    data_dummy_prefix_no_resource = pd.concat([prefix_index_encoding_case_id, prefix_index_encoding_activity, prefix_index_encoding_producer, prefix_index_encoding_timesincelastevent, prefix_index_encoding_timesincecasestart, prefix_index_encoding_timesincemidnight, prefix_index_encoding_event_nr, prefix_index_encoding_month, prefix_index_encoding_weekday, prefix_index_encoding_hour, prefix_index_encoding_open_cases, prefix_index_encoding_label, prefix_index_encoding_Age, prefix_index_encoding_Treatment_code, prefix_index_encoding_Section, prefix_index_encoding_Specialism_code_1, prefix_index_encoding_Specialism_code_2, prefix_index_encoding_group, prefix_index_encoding_num_of_exec], axis = 1)

if dataset_name == "BPIC11_f3_exp_prefix":
    data_dummy_prefix_yes_resource = prefix_index_encoding
    prefix_index_encoding_case_id = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("case_id")]
    prefix_index_encoding_activity = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Activity.code")]
    prefix_index_encoding_producer = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Producer.code")]
    prefix_index_encoding_timesincelastevent = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("timesincelastevent")]
    prefix_index_encoding_timesincecasestart = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("timesincecasestart")]
    prefix_index_encoding_timesincemidnight = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("timesincemidnight")]
    prefix_index_encoding_event_nr = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("event_nr")] # 'prefix' column and 'event_nr' column have same values here
    prefix_index_encoding_month = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("month")]
    prefix_index_encoding_weekday = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("weekday")]
    prefix_index_encoding_hour = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("hour")]
    prefix_index_encoding_open_cases = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("open_cases")]
    prefix_index_encoding_label = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("label_prefixlen1")] #Should be "label_prefixlen1" all the time
    prefix_index_encoding_Age = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Age")]
    prefix_index_encoding_Treatment_code = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Treatment.code")]
    prefix_index_encoding_Diagnosis_code = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Diagnosis.code")]
    prefix_index_encoding_Specialism_code_1 = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Specialism.code.1")]
    prefix_index_encoding_Section = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Section")]
    prefix_index_encoding_Specialism_code_2 = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Specialism.code.2")]
    prefix_index_encoding_group = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("group")]
    prefix_index_encoding_num_of_exec = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Number.of.executions")]
    data_dummy_prefix_no_resource = pd.concat([prefix_index_encoding_case_id, prefix_index_encoding_activity, prefix_index_encoding_producer, prefix_index_encoding_timesincelastevent, prefix_index_encoding_timesincecasestart, prefix_index_encoding_timesincemidnight, prefix_index_encoding_event_nr, prefix_index_encoding_month, prefix_index_encoding_weekday, prefix_index_encoding_hour, prefix_index_encoding_open_cases, prefix_index_encoding_label, prefix_index_encoding_Age, prefix_index_encoding_Treatment_code, prefix_index_encoding_Section, prefix_index_encoding_Specialism_code_1, prefix_index_encoding_Specialism_code_2, prefix_index_encoding_group, prefix_index_encoding_num_of_exec], axis = 1)

if dataset_name == "BPIC11_f4_exp_prefix":
    data_dummy_prefix_yes_resource = prefix_index_encoding
    prefix_index_encoding_case_id = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("case_id")]
    prefix_index_encoding_activity = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Activity.code")]
    prefix_index_encoding_producer = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Producer.code")]
    prefix_index_encoding_timesincelastevent = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("timesincelastevent")]
    prefix_index_encoding_timesincecasestart = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("timesincecasestart")]
    prefix_index_encoding_timesincemidnight = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("timesincemidnight")]
    prefix_index_encoding_event_nr = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("event_nr")] # 'prefix' column and 'event_nr' column have same values here
    prefix_index_encoding_month = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("month")]
    prefix_index_encoding_weekday = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("weekday")]
    prefix_index_encoding_hour = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("hour")]
    prefix_index_encoding_open_cases = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("open_cases")]
    prefix_index_encoding_label = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("label_prefixlen1")] #Should be "label_prefixlen1" all the time
    prefix_index_encoding_Age = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Age")]
    prefix_index_encoding_Treatment_code = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Treatment.code")]
    prefix_index_encoding_Diagnosis_code = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Diagnosis.code")]
    prefix_index_encoding_Specialism_code_1 = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Specialism.code.1")]
    prefix_index_encoding_Section = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Section")]
    prefix_index_encoding_Specialism_code_2 = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Specialism.code.2")]
    prefix_index_encoding_group = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("group")]
    prefix_index_encoding_num_of_exec = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Number.of.executions")]
    data_dummy_prefix_no_resource = pd.concat([prefix_index_encoding_case_id, prefix_index_encoding_activity, prefix_index_encoding_producer, prefix_index_encoding_timesincelastevent, prefix_index_encoding_timesincecasestart, prefix_index_encoding_timesincemidnight, prefix_index_encoding_event_nr, prefix_index_encoding_month, prefix_index_encoding_weekday, prefix_index_encoding_hour, prefix_index_encoding_open_cases, prefix_index_encoding_label, prefix_index_encoding_Age, prefix_index_encoding_Treatment_code, prefix_index_encoding_Section, prefix_index_encoding_Specialism_code_1, prefix_index_encoding_Specialism_code_2, prefix_index_encoding_group, prefix_index_encoding_num_of_exec], axis = 1)

if dataset_name == "BPIC15_1_f2_exp_prefix":
    data_dummy_prefix_yes_resource = prefix_index_encoding
    prefix_index_encoding_case_id = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("case_id")]
    prefix_index_encoding_activity = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Activity")]
    prefix_index_encoding_monitoringResource = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("monitoringResource")]
    prefix_index_encoding_question = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("question")]
    prefix_index_encoding_responsible_actor = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Responsible_actor")]
    prefix_index_encoding_timesincelastevent = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("timesincelastevent")]
    prefix_index_encoding_timesincecasestart = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("timesincecasestart")]
    prefix_index_encoding_timesincemidnight = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("timesincemidnight")]
    prefix_index_encoding_event_nr = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("event_nr")] # 'prefix' column and 'event_nr' column have same values here
    prefix_index_encoding_month = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("month")]
    prefix_index_encoding_weekday = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("weekday")]
    prefix_index_encoding_hour = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("hour")]
    prefix_index_encoding_open_cases = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("open_cases")]
    prefix_index_encoding_label = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("label_prefixlen1")] #Should be "label_prefixlen1" all the time
    prefix_index_encoding_SUMleges = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("SUMleges")]
    prefix_index_encoding_Aanleg = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Aanleg")]
    prefix_index_encoding_Bouw = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Bouw")]
    prefix_index_encoding_Brandveilig = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Brandveilig")]
    prefix_index_encoding_Handelen = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Handelen")]
    prefix_index_encoding_Kap = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Kap")]
    prefix_index_encoding_Milieu = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Milieu")]
    prefix_index_encoding_Monument = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Monument")]
    prefix_index_encoding_Reclame = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Reclame")]
    prefix_index_encoding_Sloop = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Sloop")]
    prefix_index_encoding_resource = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("org.resource")]
    data_dummy_prefix_no_resource = pd.concat([prefix_index_encoding_case_id, prefix_index_encoding_activity, prefix_index_encoding_monitoringResource, prefix_index_encoding_question, prefix_index_encoding_responsible_actor, prefix_index_encoding_timesincelastevent, prefix_index_encoding_timesincecasestart, prefix_index_encoding_timesincemidnight, prefix_index_encoding_event_nr, prefix_index_encoding_month, prefix_index_encoding_weekday, prefix_index_encoding_hour, prefix_index_encoding_open_cases, prefix_index_encoding_label, prefix_index_encoding_SUMleges, prefix_index_encoding_Aanleg, prefix_index_encoding_Bouw, prefix_index_encoding_Brandveilig, prefix_index_encoding_Handelen, prefix_index_encoding_Kap, prefix_index_encoding_Milieu, prefix_index_encoding_Monument, prefix_index_encoding_Reclame, prefix_index_encoding_Sloop, prefix_index_encoding_resource], axis = 1)

if dataset_name == "BPIC15_2_f2_exp_prefix":
    data_dummy_prefix_yes_resource = prefix_index_encoding
    prefix_index_encoding_case_id = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("case_id")]
    prefix_index_encoding_activity = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Activity")]
    prefix_index_encoding_monitoringResource = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("monitoringResource")]
    prefix_index_encoding_question = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("question")]
    prefix_index_encoding_responsible_actor = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Responsible_actor")]
    prefix_index_encoding_timesincelastevent = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("timesincelastevent")]
    prefix_index_encoding_timesincecasestart = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("timesincecasestart")]
    prefix_index_encoding_timesincemidnight = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("timesincemidnight")]
    prefix_index_encoding_event_nr = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("event_nr")] # 'prefix' column and 'event_nr' column have same values here
    prefix_index_encoding_month = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("month")]
    prefix_index_encoding_weekday = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("weekday")]
    prefix_index_encoding_hour = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("hour")]
    prefix_index_encoding_open_cases = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("open_cases")]
    prefix_index_encoding_label = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("label_prefixlen1")] #Should be "label_prefixlen1" all the time
    prefix_index_encoding_SUMleges = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("SUMleges")]
    prefix_index_encoding_Aanleg = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Aanleg")]
    prefix_index_encoding_Bouw = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Bouw")]
    prefix_index_encoding_Brandveilig = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Brandveilig")]
    prefix_index_encoding_Handelen = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Handelen")]
    prefix_index_encoding_Kap = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Kap")]
    prefix_index_encoding_Milieu = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Milieu")]
    prefix_index_encoding_Monument = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Monument")]
    prefix_index_encoding_Reclame = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Reclame")]
    prefix_index_encoding_Sloop = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Sloop")]
    prefix_index_encoding_resource = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("org.resource")]
    data_dummy_prefix_no_resource = pd.concat([prefix_index_encoding_case_id, prefix_index_encoding_activity, prefix_index_encoding_monitoringResource, prefix_index_encoding_question, prefix_index_encoding_responsible_actor, prefix_index_encoding_timesincelastevent, prefix_index_encoding_timesincecasestart, prefix_index_encoding_timesincemidnight, prefix_index_encoding_event_nr, prefix_index_encoding_month, prefix_index_encoding_weekday, prefix_index_encoding_hour, prefix_index_encoding_open_cases, prefix_index_encoding_label, prefix_index_encoding_SUMleges, prefix_index_encoding_Aanleg, prefix_index_encoding_Bouw, prefix_index_encoding_Brandveilig, prefix_index_encoding_Handelen, prefix_index_encoding_Kap, prefix_index_encoding_Milieu, prefix_index_encoding_Monument, prefix_index_encoding_Reclame, prefix_index_encoding_Sloop, prefix_index_encoding_resource], axis = 1)

if dataset_name == "BPIC15_3_f2_exp_prefix":
    data_dummy_prefix_yes_resource = prefix_index_encoding
    prefix_index_encoding_case_id = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("case_id")]
    prefix_index_encoding_activity = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Activity")]
    prefix_index_encoding_monitoringResource = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("monitoringResource")]
    prefix_index_encoding_question = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("question")]
    prefix_index_encoding_responsible_actor = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Responsible_actor")]
    prefix_index_encoding_timesincelastevent = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("timesincelastevent")]
    prefix_index_encoding_timesincecasestart = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("timesincecasestart")]
    prefix_index_encoding_timesincemidnight = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("timesincemidnight")]
    prefix_index_encoding_event_nr = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("event_nr")] # 'prefix' column and 'event_nr' column have same values here
    prefix_index_encoding_month = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("month")]
    prefix_index_encoding_weekday = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("weekday")]
    prefix_index_encoding_hour = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("hour")]
    prefix_index_encoding_open_cases = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("open_cases")]
    prefix_index_encoding_label = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("label_prefixlen1")] #Should be "label_prefixlen1" all the time
    prefix_index_encoding_SUMleges = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("SUMleges")]
    prefix_index_encoding_Aanleg = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Aanleg")]
    prefix_index_encoding_Bouw = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Bouw")]
    prefix_index_encoding_Brandveilig = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Brandveilig")]
    prefix_index_encoding_Handelen = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Handelen")]
    prefix_index_encoding_Kap = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Kap")]
    prefix_index_encoding_Milieu = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Milieu")]
    prefix_index_encoding_Monument = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Monument")]
    prefix_index_encoding_Reclame = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Reclame")]
    prefix_index_encoding_Sloop = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Sloop")]
    prefix_index_encoding_resource = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("org.resource")]
    data_dummy_prefix_no_resource = pd.concat([prefix_index_encoding_case_id, prefix_index_encoding_activity, prefix_index_encoding_monitoringResource, prefix_index_encoding_question, prefix_index_encoding_responsible_actor, prefix_index_encoding_timesincelastevent, prefix_index_encoding_timesincecasestart, prefix_index_encoding_timesincemidnight, prefix_index_encoding_event_nr, prefix_index_encoding_month, prefix_index_encoding_weekday, prefix_index_encoding_hour, prefix_index_encoding_open_cases, prefix_index_encoding_label, prefix_index_encoding_SUMleges, prefix_index_encoding_Aanleg, prefix_index_encoding_Bouw, prefix_index_encoding_Brandveilig, prefix_index_encoding_Handelen, prefix_index_encoding_Kap, prefix_index_encoding_Milieu, prefix_index_encoding_Monument, prefix_index_encoding_Reclame, prefix_index_encoding_Sloop, prefix_index_encoding_resource], axis = 1)

if dataset_name == "BPIC15_4_f2_exp_prefix":
    data_dummy_prefix_yes_resource = prefix_index_encoding
    prefix_index_encoding_case_id = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("case_id")]
    prefix_index_encoding_activity = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Activity")]
    prefix_index_encoding_monitoringResource = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("monitoringResource")]
    prefix_index_encoding_question = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("question")]
    prefix_index_encoding_responsible_actor = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Responsible_actor")]
    prefix_index_encoding_timesincelastevent = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("timesincelastevent")]
    prefix_index_encoding_timesincecasestart = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("timesincecasestart")]
    prefix_index_encoding_timesincemidnight = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("timesincemidnight")]
    prefix_index_encoding_event_nr = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("event_nr")] # 'prefix' column and 'event_nr' column have same values here
    prefix_index_encoding_month = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("month")]
    prefix_index_encoding_weekday = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("weekday")]
    prefix_index_encoding_hour = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("hour")]
    prefix_index_encoding_open_cases = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("open_cases")]
    prefix_index_encoding_label = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("label_prefixlen1")] #Should be "label_prefixlen1" all the time
    prefix_index_encoding_SUMleges = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("SUMleges")]
    prefix_index_encoding_Aanleg = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Aanleg")]
    prefix_index_encoding_Bouw = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Bouw")]
    prefix_index_encoding_Brandveilig = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Brandveilig")]
    prefix_index_encoding_Handelen = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Handelen")]
    prefix_index_encoding_Kap = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Kap")]
    prefix_index_encoding_Milieu = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Milieu")]
    prefix_index_encoding_Monument = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Monument")]
    prefix_index_encoding_Reclame = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Reclame")]
    prefix_index_encoding_Sloop = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Sloop")]
    prefix_index_encoding_resource = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("org.resource")]
    data_dummy_prefix_no_resource = pd.concat([prefix_index_encoding_case_id, prefix_index_encoding_activity, prefix_index_encoding_monitoringResource, prefix_index_encoding_question, prefix_index_encoding_responsible_actor, prefix_index_encoding_timesincelastevent, prefix_index_encoding_timesincecasestart, prefix_index_encoding_timesincemidnight, prefix_index_encoding_event_nr, prefix_index_encoding_month, prefix_index_encoding_weekday, prefix_index_encoding_hour, prefix_index_encoding_open_cases, prefix_index_encoding_label, prefix_index_encoding_SUMleges, prefix_index_encoding_Aanleg, prefix_index_encoding_Bouw, prefix_index_encoding_Brandveilig, prefix_index_encoding_Handelen, prefix_index_encoding_Kap, prefix_index_encoding_Milieu, prefix_index_encoding_Monument, prefix_index_encoding_Reclame, prefix_index_encoding_Sloop, prefix_index_encoding_resource], axis = 1)

if dataset_name == "BPIC15_5_f2_exp_prefix":
    data_dummy_prefix_yes_resource = prefix_index_encoding
    prefix_index_encoding_case_id = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("case_id")]
    prefix_index_encoding_activity = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Activity")]
    prefix_index_encoding_monitoringResource = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("monitoringResource")]
    prefix_index_encoding_question = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("question")]
    prefix_index_encoding_responsible_actor = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Responsible_actor")]
    prefix_index_encoding_timesincelastevent = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("timesincelastevent")]
    prefix_index_encoding_timesincecasestart = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("timesincecasestart")]
    prefix_index_encoding_timesincemidnight = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("timesincemidnight")]
    prefix_index_encoding_event_nr = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("event_nr")] # 'prefix' column and 'event_nr' column have same values here
    prefix_index_encoding_month = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("month")]
    prefix_index_encoding_weekday = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("weekday")]
    prefix_index_encoding_hour = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("hour")]
    prefix_index_encoding_open_cases = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("open_cases")]
    prefix_index_encoding_label = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("label_prefixlen1")] #Should be "label_prefixlen1" all the time
    prefix_index_encoding_SUMleges = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("SUMleges")]
    prefix_index_encoding_Aanleg = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Aanleg")]
    prefix_index_encoding_Bouw = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Bouw")]
    prefix_index_encoding_Brandveilig = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Brandveilig")]
    prefix_index_encoding_Handelen = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Handelen")]
    prefix_index_encoding_Kap = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Kap")]
    prefix_index_encoding_Milieu = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Milieu")]
    prefix_index_encoding_Monument = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Monument")]
    prefix_index_encoding_Reclame = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Reclame")]
    prefix_index_encoding_Sloop = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Sloop")]
    prefix_index_encoding_resource = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("org.resource")]
    data_dummy_prefix_no_resource = pd.concat([prefix_index_encoding_case_id, prefix_index_encoding_activity, prefix_index_encoding_monitoringResource, prefix_index_encoding_question, prefix_index_encoding_responsible_actor, prefix_index_encoding_timesincelastevent, prefix_index_encoding_timesincecasestart, prefix_index_encoding_timesincemidnight, prefix_index_encoding_event_nr, prefix_index_encoding_month, prefix_index_encoding_weekday, prefix_index_encoding_hour, prefix_index_encoding_open_cases, prefix_index_encoding_label, prefix_index_encoding_SUMleges, prefix_index_encoding_Aanleg, prefix_index_encoding_Bouw, prefix_index_encoding_Brandveilig, prefix_index_encoding_Handelen, prefix_index_encoding_Kap, prefix_index_encoding_Milieu, prefix_index_encoding_Monument, prefix_index_encoding_Reclame, prefix_index_encoding_Sloop, prefix_index_encoding_resource], axis = 1)

if dataset_name == "traffic_fines_1_exp_prefix":
    data_dummy_prefix_yes_resource = prefix_index_encoding
    prefix_index_encoding_case_id = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("case_id")]
    prefix_index_encoding_activity = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Activity")]
    prefix_index_encoding_article = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("article")]
    prefix_index_encoding_amount = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("amount")]
    prefix_index_encoding_vehicleClass = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("vehicleClass")]
    prefix_index_encoding_points = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("points")]
    prefix_index_encoding_lastSent = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("lastSent")]
    prefix_index_encoding_notificationType = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("notificationType")]
    prefix_index_encoding_dismissal = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("dismissal")]
    prefix_index_encoding_expense = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("expense")]
    prefix_index_encoding_monitoringResource = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("monitoringResource")]
    prefix_index_encoding_responsible_actor = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Responsible_actor")]
    prefix_index_encoding_timesincelastevent = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("timesincelastevent")]
    prefix_index_encoding_timesincecasestart = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("timesincecasestart")]
    prefix_index_encoding_timesincemidnight = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("timesincemidnight")]
    prefix_index_encoding_event_nr = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("event_nr")] # 'prefix' column and 'event_nr' column have same values here
    prefix_index_encoding_month = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("month")]
    prefix_index_encoding_weekday = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("weekday")]
    prefix_index_encoding_hour = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("hour")]
    prefix_index_encoding_open_cases = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("open_cases")]
    prefix_index_encoding_label = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("label_prefixlen1")] #Should be "label_prefixlen1" all the time
    prefix_index_encoding_resource = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Resource")]
    data_dummy_prefix_no_resource = pd.concat([prefix_index_encoding_case_id, prefix_index_encoding_activity, prefix_index_encoding_article, prefix_index_encoding_amount, prefix_index_encoding_vehicleClass, prefix_index_encoding_points, prefix_index_encoding_lastSent, prefix_index_encoding_notificationType, prefix_index_encoding_dismissal, prefix_index_encoding_expense, prefix_index_encoding_monitoringResource, prefix_index_encoding_responsible_actor, prefix_index_encoding_timesincelastevent, prefix_index_encoding_timesincecasestart, prefix_index_encoding_timesincemidnight, prefix_index_encoding_event_nr, prefix_index_encoding_month, prefix_index_encoding_weekday, prefix_index_encoding_hour, prefix_index_encoding_open_cases, prefix_index_encoding_label, prefix_index_encoding_resource], axis = 1)

if dataset_name == "bpic2012_O_CANCELLED-COMPLETE_exp_prefix":
    data_dummy_prefix_yes_resource = prefix_index_encoding
    prefix_index_encoding_case_id = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("case_id")]
    prefix_index_encoding_activity = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Activity")]
    prefix_index_encoding_amount_req = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("AMOUNT_REQ")]
    prefix_index_encoding_timesincelastevent = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("timesincelastevent")]
    prefix_index_encoding_timesincecasestart = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("timesincecasestart")]
    prefix_index_encoding_timesincemidnight = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("timesincemidnight")]
    prefix_index_encoding_event_nr = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("event_nr")] # 'prefix' column and 'event_nr' column have same values here
    prefix_index_encoding_month = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("month")]
    prefix_index_encoding_weekday = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("weekday")]
    prefix_index_encoding_hour = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("hour")]
    prefix_index_encoding_open_cases = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("open_cases")]
    prefix_index_encoding_label = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("label_prefixlen1")] #Should be "label_prefixlen1" all the time
    prefix_index_encoding_resource = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Resource")]
    data_dummy_prefix_no_resource = pd.concat([prefix_index_encoding_case_id, prefix_index_encoding_activity, prefix_index_encoding_amount_req, prefix_index_encoding_timesincelastevent, prefix_index_encoding_timesincecasestart, prefix_index_encoding_timesincemidnight, prefix_index_encoding_event_nr, prefix_index_encoding_month, prefix_index_encoding_weekday, prefix_index_encoding_hour, prefix_index_encoding_open_cases, prefix_index_encoding_label, prefix_index_encoding_resource], axis = 1)

if dataset_name == "bpic2012_O_ACCEPTED-COMPLETE_exp_prefix":
    data_dummy_prefix_yes_resource = prefix_index_encoding
    prefix_index_encoding_case_id = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("case_id")]
    prefix_index_encoding_activity = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Activity")]
    prefix_index_encoding_amount_req = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("AMOUNT_REQ")]
    prefix_index_encoding_timesincelastevent = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("timesincelastevent")]
    prefix_index_encoding_timesincecasestart = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("timesincecasestart")]
    prefix_index_encoding_timesincemidnight = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("timesincemidnight")]
    prefix_index_encoding_event_nr = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("event_nr")] # 'prefix' column and 'event_nr' column have same values here
    prefix_index_encoding_month = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("month")]
    prefix_index_encoding_weekday = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("weekday")]
    prefix_index_encoding_hour = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("hour")]
    prefix_index_encoding_open_cases = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("open_cases")]
    prefix_index_encoding_label = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("label_prefixlen1")] #Should be "label_prefixlen1" all the time
    prefix_index_encoding_resource = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Resource")]
    data_dummy_prefix_no_resource = pd.concat([prefix_index_encoding_case_id, prefix_index_encoding_activity, prefix_index_encoding_amount_req, prefix_index_encoding_timesincelastevent, prefix_index_encoding_timesincecasestart, prefix_index_encoding_timesincemidnight, prefix_index_encoding_event_nr, prefix_index_encoding_month, prefix_index_encoding_weekday, prefix_index_encoding_hour, prefix_index_encoding_open_cases, prefix_index_encoding_label, prefix_index_encoding_resource], axis = 1)

if dataset_name == "bpic2012_O_DECLINED-COMPLETE_exp_prefix":
    data_dummy_prefix_yes_resource = prefix_index_encoding
    prefix_index_encoding_case_id = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("case_id")]
    prefix_index_encoding_activity = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Activity")]
    prefix_index_encoding_amount_req = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("AMOUNT_REQ")]
    prefix_index_encoding_timesincelastevent = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("timesincelastevent")]
    prefix_index_encoding_timesincecasestart = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("timesincecasestart")]
    prefix_index_encoding_timesincemidnight = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("timesincemidnight")]
    prefix_index_encoding_event_nr = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("event_nr")] # 'prefix' column and 'event_nr' column have same values here
    prefix_index_encoding_month = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("month")]
    prefix_index_encoding_weekday = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("weekday")]
    prefix_index_encoding_hour = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("hour")]
    prefix_index_encoding_open_cases = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("open_cases")]
    prefix_index_encoding_label = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("label_prefixlen1")] #Should be "label_prefixlen1" all the time
    prefix_index_encoding_resource = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Resource")]
    data_dummy_prefix_no_resource = pd.concat([prefix_index_encoding_case_id, prefix_index_encoding_activity, prefix_index_encoding_amount_req, prefix_index_encoding_timesincelastevent, prefix_index_encoding_timesincecasestart, prefix_index_encoding_timesincemidnight, prefix_index_encoding_event_nr, prefix_index_encoding_month, prefix_index_encoding_weekday, prefix_index_encoding_hour, prefix_index_encoding_open_cases, prefix_index_encoding_label, prefix_index_encoding_resource], axis = 1)

if dataset_name == "BPIC17_O_Cancelled_exp_prefix":
    data_dummy_prefix_yes_resource = prefix_index_encoding
    prefix_index_encoding_case_id = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("case_id")]
    prefix_index_encoding_activity_a = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Activity_A")]
    prefix_index_encoding_activity_o = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Activity_O")]
    prefix_index_encoding_activity_w = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Activity_W")]
    prefix_index_encoding_application = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("ApplicationType")]
    prefix_index_encoding_loangoal = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("LoanGoal")]
    prefix_index_encoding_action = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Action")]
    prefix_index_encoding_eventorigin = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("EventOrigin")]
    prefix_index_encoding_lifecycle_transition = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("lifecycle.transition")]
    prefix_index_encoding_accepted_false = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Accepted_FALSE")]
    prefix_index_encoding_accepted_true = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Accepted_TRUE")]
    prefix_index_encoding_accepted_missing = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Accepted_missing")]
    prefix_index_encoding_selected_false = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Selected_FALSE")]
    prefix_index_encoding_selected_true = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Selected_TRUE")]
    prefix_index_encoding_selected_missing = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Selected_missing")]
    prefix_index_encoding_firstwithdrawalamount = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("FirstWithdrawalAmount")]
    prefix_index_encoding_monthlycost = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("MonthlyCost")]
    prefix_index_encoding_numberofterms = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("NumberOfTerms")]
    prefix_index_encoding_offeredamount = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("OfferedAmount")]
    prefix_index_encoding_creditscore = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("CreditScore")]
    prefix_index_encoding_requestedamount = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("RequestedAmount")]
    prefix_index_encoding_timesincelastevent = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("timesincelastevent")]
    prefix_index_encoding_timesincecasestart = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("timesincecasestart")]
    prefix_index_encoding_timesincemidnight = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("timesincemidnight")]
    prefix_index_encoding_event_nr = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("event_nr")] # 'prefix' column and 'event_nr' column have same values here
    prefix_index_encoding_month = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("month")]
    prefix_index_encoding_weekday = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("weekday")]
    prefix_index_encoding_hour = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("hour")]
    prefix_index_encoding_open_cases = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("open_cases")]
    prefix_index_encoding_label = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("label_prefixlen1")] #Should be "label_prefixlen1" all the time
    prefix_index_encoding_resource = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("org.resource")]
    data_dummy_prefix_no_resource = pd.concat([prefix_index_encoding_case_id, prefix_index_encoding_activity_a, prefix_index_encoding_activity_o, prefix_index_encoding_activity_w, prefix_index_encoding_application, prefix_index_encoding_loangoal, prefix_index_encoding_action,prefix_index_encoding_eventorigin,prefix_index_encoding_lifecycle_transition,prefix_index_encoding_accepted_false,prefix_index_encoding_accepted_true,prefix_index_encoding_accepted_missing,prefix_index_encoding_selected_false,prefix_index_encoding_selected_true,prefix_index_encoding_selected_missing,prefix_index_encoding_firstwithdrawalamount,prefix_index_encoding_monthlycost,prefix_index_encoding_numberofterms,prefix_index_encoding_offeredamount,prefix_index_encoding_creditscore,prefix_index_encoding_requestedamount,prefix_index_encoding_timesincelastevent,prefix_index_encoding_timesincecasestart,prefix_index_encoding_timesincemidnight,prefix_index_encoding_event_nr,prefix_index_encoding_month,prefix_index_encoding_weekday,prefix_index_encoding_hour,prefix_index_encoding_open_cases,prefix_index_encoding_label,prefix_index_encoding_resource], axis = 1)

if dataset_name == "BPIC17_O_Accepted_exp_prefix":
    data_dummy_prefix_yes_resource = prefix_index_encoding
    prefix_index_encoding_case_id = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("case_id")]
    prefix_index_encoding_activity_a = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Activity_A")]
    prefix_index_encoding_activity_o = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Activity_O")]
    prefix_index_encoding_activity_w = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Activity_W")]
    prefix_index_encoding_application = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("ApplicationType")]
    prefix_index_encoding_loangoal = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("LoanGoal")]
    prefix_index_encoding_action = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Action")]
    prefix_index_encoding_eventorigin = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("EventOrigin")]
    prefix_index_encoding_lifecycle_transition = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("lifecycle.transition")]
    prefix_index_encoding_accepted_false = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Accepted_FALSE")]
    prefix_index_encoding_accepted_true = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Accepted_TRUE")]
    prefix_index_encoding_accepted_missing = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Accepted_missing")]
    prefix_index_encoding_selected_false = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Selected_FALSE")]
    prefix_index_encoding_selected_true = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Selected_TRUE")]
    prefix_index_encoding_selected_missing = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Selected_missing")]
    prefix_index_encoding_firstwithdrawalamount = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("FirstWithdrawalAmount")]
    prefix_index_encoding_monthlycost = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("MonthlyCost")]
    prefix_index_encoding_numberofterms = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("NumberOfTerms")]
    prefix_index_encoding_offeredamount = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("OfferedAmount")]
    prefix_index_encoding_creditscore = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("CreditScore")]
    prefix_index_encoding_requestedamount = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("RequestedAmount")]
    prefix_index_encoding_timesincelastevent = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("timesincelastevent")]
    prefix_index_encoding_timesincecasestart = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("timesincecasestart")]
    prefix_index_encoding_timesincemidnight = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("timesincemidnight")]
    prefix_index_encoding_event_nr = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("event_nr")] # 'prefix' column and 'event_nr' column have same values here
    prefix_index_encoding_month = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("month")]
    prefix_index_encoding_weekday = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("weekday")]
    prefix_index_encoding_hour = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("hour")]
    prefix_index_encoding_open_cases = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("open_cases")]
    prefix_index_encoding_label = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("label_prefixlen1")] #Should be "label_prefixlen1" all the time
    prefix_index_encoding_resource = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("org.resource")]
    data_dummy_prefix_no_resource = pd.concat([prefix_index_encoding_case_id, prefix_index_encoding_activity_a, prefix_index_encoding_activity_o, prefix_index_encoding_activity_w, prefix_index_encoding_application, prefix_index_encoding_loangoal, prefix_index_encoding_action,prefix_index_encoding_eventorigin,prefix_index_encoding_lifecycle_transition,prefix_index_encoding_accepted_false,prefix_index_encoding_accepted_true,prefix_index_encoding_accepted_missing,prefix_index_encoding_selected_false,prefix_index_encoding_selected_true,prefix_index_encoding_selected_missing,prefix_index_encoding_firstwithdrawalamount,prefix_index_encoding_monthlycost,prefix_index_encoding_numberofterms,prefix_index_encoding_offeredamount,prefix_index_encoding_creditscore,prefix_index_encoding_requestedamount,prefix_index_encoding_timesincelastevent,prefix_index_encoding_timesincecasestart,prefix_index_encoding_timesincemidnight,prefix_index_encoding_event_nr,prefix_index_encoding_month,prefix_index_encoding_weekday,prefix_index_encoding_hour,prefix_index_encoding_open_cases,prefix_index_encoding_label,prefix_index_encoding_resource], axis = 1)

if dataset_name == "BPIC17_O_Refused_exp_prefix":
    data_dummy_prefix_yes_resource = prefix_index_encoding
    prefix_index_encoding_case_id = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("case_id")]
    prefix_index_encoding_activity_a = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Activity_A")]
    prefix_index_encoding_activity_o = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Activity_O")]
    prefix_index_encoding_activity_w = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Activity_W")]
    prefix_index_encoding_application = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("ApplicationType")]
    prefix_index_encoding_loangoal = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("LoanGoal")]
    prefix_index_encoding_action = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Action")]
    prefix_index_encoding_eventorigin = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("EventOrigin")]
    prefix_index_encoding_lifecycle_transition = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("lifecycle.transition")]
    prefix_index_encoding_accepted_false = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Accepted_FALSE")]
    prefix_index_encoding_accepted_true = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Accepted_TRUE")]
    prefix_index_encoding_accepted_missing = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Accepted_missing")]
    prefix_index_encoding_selected_false = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Selected_FALSE")]
    prefix_index_encoding_selected_true = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Selected_TRUE")]
    prefix_index_encoding_selected_missing = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("Selected_missing")]
    prefix_index_encoding_firstwithdrawalamount = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("FirstWithdrawalAmount")]
    prefix_index_encoding_monthlycost = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("MonthlyCost")]
    prefix_index_encoding_numberofterms = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("NumberOfTerms")]
    prefix_index_encoding_offeredamount = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("OfferedAmount")]
    prefix_index_encoding_creditscore = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("CreditScore")]
    prefix_index_encoding_requestedamount = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("RequestedAmount")]
    prefix_index_encoding_timesincelastevent = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("timesincelastevent")]
    prefix_index_encoding_timesincecasestart = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("timesincecasestart")]
    prefix_index_encoding_timesincemidnight = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("timesincemidnight")]
    prefix_index_encoding_event_nr = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("event_nr")] # 'prefix' column and 'event_nr' column have same values here
    prefix_index_encoding_month = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("month")]
    prefix_index_encoding_weekday = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("weekday")]
    prefix_index_encoding_hour = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("hour")]
    prefix_index_encoding_open_cases = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("open_cases")]
    prefix_index_encoding_label = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("label_prefixlen1")] #Should be "label_prefixlen1" all the time
    prefix_index_encoding_resource = prefix_index_encoding.iloc[:,prefix_index_encoding.columns.str.contains("org.resource")]
    data_dummy_prefix_no_resource = pd.concat([prefix_index_encoding_case_id, prefix_index_encoding_activity_a, prefix_index_encoding_activity_o, prefix_index_encoding_activity_w, prefix_index_encoding_application, prefix_index_encoding_loangoal, prefix_index_encoding_action,prefix_index_encoding_eventorigin,prefix_index_encoding_lifecycle_transition,prefix_index_encoding_accepted_false,prefix_index_encoding_accepted_true,prefix_index_encoding_accepted_missing,prefix_index_encoding_selected_false,prefix_index_encoding_selected_true,prefix_index_encoding_selected_missing,prefix_index_encoding_firstwithdrawalamount,prefix_index_encoding_monthlycost,prefix_index_encoding_numberofterms,prefix_index_encoding_offeredamount,prefix_index_encoding_creditscore,prefix_index_encoding_requestedamount,prefix_index_encoding_timesincelastevent,prefix_index_encoding_timesincecasestart,prefix_index_encoding_timesincemidnight,prefix_index_encoding_event_nr,prefix_index_encoding_month,prefix_index_encoding_weekday,prefix_index_encoding_hour,prefix_index_encoding_open_cases,prefix_index_encoding_label,prefix_index_encoding_resource], axis = 1)

# Deleting "case_len" feature
data_dummy_prefix_yes_resource = data_dummy_prefix_yes_resource.drop("case_len_prefixlen1",1) #"case_len" feature only exists in data_dummy_prefix_yes_resource, not in data_dummy_no_resource

data_dummy_prefix_no_resource.to_csv('data_dummy_prefix_no_resource.csv', index = False)
data_dummy_prefix_yes_resource.to_csv('data_dummy_prefix_yes_resource.csv', index = False)

# Deleting NAs
data_dummy_prefix_no_resource = data_dummy_prefix_no_resource.dropna()
data_dummy_prefix_yes_resource = data_dummy_prefix_yes_resource.dropna()

# Splitting the data into X and y
y_dummy_no_resource = pd.DataFrame([0 if i == "regular" else 1 for i in data_dummy_prefix_no_resource['label_prefixlen1']])
y_dummy_yes_resource = pd.DataFrame([0 if i == "regular" else 1 for i in data_dummy_prefix_yes_resource['label_prefixlen1']])

X_dummy_no_resource = data_dummy_prefix_no_resource[data_dummy_prefix_no_resource.columns.drop('label_prefixlen1')]
X_dummy_yes_resource = data_dummy_prefix_yes_resource[data_dummy_prefix_yes_resource.columns.drop('label_prefixlen1')]
#X_dummy_no_resource = data_dummy_prefix_no_resource[data_dummy_prefix_no_resource.columns.drop(list(data_dummy_prefix_no_resource.filter(regex='label')))]
#X_dummy_yes_resource = data_dummy_prefix_yes_resource[data_dummy_prefix_yes_resource.columns.drop(list(data_dummy_prefix_yes_resource.filter(regex='label')))]
#X_dummy_no_resource = data_dummy_prefix_no_resource.drop(["label_prefixlen1"], axis = 1)
#X_dummy_yes_resource = data_dummy_prefix_yes_resource.drop(["label_prefixlen1"], axis = 1)


# Normalization of common features across all datasets
for i in range(max_prefix):
    feature_name = "timesincelastevent_prefixlen" + str(i+1)
    if np.var(X_dummy_no_resource[feature_name]) == 0:
        break
    X_dummy_no_resource[feature_name] = (X_dummy_no_resource[feature_name]-min(X_dummy_no_resource[feature_name]))/(max(X_dummy_no_resource[feature_name])-min(X_dummy_no_resource[feature_name])) # 0~1 normalization
for i in range(max_prefix):
    feature_name = "timesincemidnight_prefixlen" + str(i+1)
    if np.var(X_dummy_no_resource[feature_name]) == 0:
        break
    X_dummy_no_resource[feature_name] = (X_dummy_no_resource[feature_name]-min(X_dummy_no_resource[feature_name]))/(max(X_dummy_no_resource[feature_name])-min(X_dummy_no_resource[feature_name])) # 0~1 normalization
for i in range(max_prefix):
    feature_name = "event_nr_prefixlen" + str(i+1)
    if np.var(X_dummy_no_resource[feature_name]) == 0:
        break
    X_dummy_no_resource[feature_name] = (X_dummy_no_resource[feature_name]-min(X_dummy_no_resource[feature_name]))/(max(X_dummy_no_resource[feature_name])-min(X_dummy_no_resource[feature_name])) # 0~1 normalization
for i in range(max_prefix):
    feature_name = "month_prefixlen" + str(i+1)
    if np.var(X_dummy_no_resource[feature_name]) == 0:
        break
    X_dummy_no_resource[feature_name] = (X_dummy_no_resource[feature_name]-min(X_dummy_no_resource[feature_name]))/(max(X_dummy_no_resource[feature_name])-min(X_dummy_no_resource[feature_name])) # 0~1 normalization
for i in range(max_prefix):
    feature_name = "weekday_prefixlen" + str(i+1)
    if np.var(X_dummy_no_resource[feature_name]) == 0:
        break
    X_dummy_no_resource[feature_name] = (X_dummy_no_resource[feature_name]-min(X_dummy_no_resource[feature_name]))/(max(X_dummy_no_resource[feature_name])-min(X_dummy_no_resource[feature_name])) # 0~1 normalization
for i in range(max_prefix):
    feature_name = "hour_prefixlen" + str(i+1)
    if np.var(X_dummy_no_resource[feature_name]) == 0:
        break
    X_dummy_no_resource[feature_name] = (X_dummy_no_resource[feature_name]-min(X_dummy_no_resource[feature_name]))/(max(X_dummy_no_resource[feature_name])-min(X_dummy_no_resource[feature_name])) # 0~1 normalization
for i in range(max_prefix):
    feature_name = "open_cases_prefixlen" + str(i+1)
    if np.var(X_dummy_no_resource[feature_name]) == 0:
        break
    X_dummy_no_resource[feature_name] = (X_dummy_no_resource[feature_name]-min(X_dummy_no_resource[feature_name]))/(max(X_dummy_no_resource[feature_name])-min(X_dummy_no_resource[feature_name])) # 0~1 normalization


# Normalization for features only available in BPIC15_1_f2_exp_prefix
# If the feature is both case-level attribute and numeric, only one line will be written for normalization as below.
# If the feature is not a case-level attribute but numeric, five lines will be written (starting with for statement) for normalization as below.
if dataset_name == "BPIC11_f1_exp_prefix":
    X_dummy_no_resource['Age_prefixlen1'] = (X_dummy_no_resource['Age_prefixlen1']-min(X_dummy_no_resource['Age_prefixlen1']))/(max(X_dummy_no_resource['Age_prefixlen1'])-min(X_dummy_no_resource['Age_prefixlen1'])) # 0~1 normalization
    for i in range(max_prefix):
        feature_name = "Number.of.executions_prefixlen" + str(i+1)
        if np.var(X_dummy_no_resource[feature_name]) == 0:
            break
        X_dummy_no_resource[feature_name] = (X_dummy_no_resource[feature_name]-min(X_dummy_no_resource[feature_name]))/(max(X_dummy_no_resource[feature_name])-min(X_dummy_no_resource[feature_name])) # 0~1 normalization

if dataset_name == "BPIC11_f2_exp_prefix":
    X_dummy_no_resource['Age_prefixlen1'] = (X_dummy_no_resource['Age_prefixlen1']-min(X_dummy_no_resource['Age_prefixlen1']))/(max(X_dummy_no_resource['Age_prefixlen1'])-min(X_dummy_no_resource['Age_prefixlen1'])) # 0~1 normalization
    for i in range(max_prefix):
        feature_name = "Number.of.executions_prefixlen" + str(i+1)
        if np.var(X_dummy_no_resource[feature_name]) == 0:
            break
        X_dummy_no_resource[feature_name] = (X_dummy_no_resource[feature_name]-min(X_dummy_no_resource[feature_name]))/(max(X_dummy_no_resource[feature_name])-min(X_dummy_no_resource[feature_name])) # 0~1 normalization

if dataset_name == "BPIC11_f3_exp_prefix":
    X_dummy_no_resource['Age_prefixlen1'] = (X_dummy_no_resource['Age_prefixlen1']-min(X_dummy_no_resource['Age_prefixlen1']))/(max(X_dummy_no_resource['Age_prefixlen1'])-min(X_dummy_no_resource['Age_prefixlen1'])) # 0~1 normalization
    for i in range(max_prefix):
        feature_name = "Number.of.executions_prefixlen" + str(i+1)
        if np.var(X_dummy_no_resource[feature_name]) == 0:
            break
        X_dummy_no_resource[feature_name] = (X_dummy_no_resource[feature_name]-min(X_dummy_no_resource[feature_name]))/(max(X_dummy_no_resource[feature_name])-min(X_dummy_no_resource[feature_name])) # 0~1 normalization

if dataset_name == "BPIC11_f4_exp_prefix":
    X_dummy_no_resource['Age_prefixlen1'] = (X_dummy_no_resource['Age_prefixlen1']-min(X_dummy_no_resource['Age_prefixlen1']))/(max(X_dummy_no_resource['Age_prefixlen1'])-min(X_dummy_no_resource['Age_prefixlen1'])) # 0~1 normalization
    for i in range(max_prefix):
        feature_name = "Number.of.executions_prefixlen" + str(i+1)
        if np.var(X_dummy_no_resource[feature_name]) == 0:
            break
        X_dummy_no_resource[feature_name] = (X_dummy_no_resource[feature_name]-min(X_dummy_no_resource[feature_name]))/(max(X_dummy_no_resource[feature_name])-min(X_dummy_no_resource[feature_name])) # 0~1 normalization

if dataset_name == "BPIC15_1_f2_exp_prefix":
    X_dummy_no_resource['SUMleges_prefixlen1'] = (X_dummy_no_resource['SUMleges_prefixlen1']-min(X_dummy_no_resource['SUMleges_prefixlen1']))/(max(X_dummy_no_resource['SUMleges_prefixlen1'])-min(X_dummy_no_resource['SUMleges_prefixlen1'])) # 0~1 normalization

if dataset_name == "BPIC15_2_f2_exp_prefix":
    X_dummy_no_resource['SUMleges_prefixlen1'] = (X_dummy_no_resource['SUMleges_prefixlen1']-min(X_dummy_no_resource['SUMleges_prefixlen1']))/(max(X_dummy_no_resource['SUMleges_prefixlen1'])-min(X_dummy_no_resource['SUMleges_prefixlen1'])) # 0~1 normalization

if dataset_name == "BPIC15_3_f2_exp_prefix":
    X_dummy_no_resource['SUMleges_prefixlen1'] = (X_dummy_no_resource['SUMleges_prefixlen1']-min(X_dummy_no_resource['SUMleges_prefixlen1']))/(max(X_dummy_no_resource['SUMleges_prefixlen1'])-min(X_dummy_no_resource['SUMleges_prefixlen1'])) # 0~1 normalization

if dataset_name == "BPIC15_4_f2_exp_prefix":
    X_dummy_no_resource['SUMleges_prefixlen1'] = (X_dummy_no_resource['SUMleges_prefixlen1']-min(X_dummy_no_resource['SUMleges_prefixlen1']))/(max(X_dummy_no_resource['SUMleges_prefixlen1'])-min(X_dummy_no_resource['SUMleges_prefixlen1'])) # 0~1 normalization

if dataset_name == "BPIC15_5_f2_exp_prefix":
    X_dummy_no_resource['SUMleges_prefixlen1'] = (X_dummy_no_resource['SUMleges_prefixlen1']-min(X_dummy_no_resource['SUMleges_prefixlen1']))/(max(X_dummy_no_resource['SUMleges_prefixlen1'])-min(X_dummy_no_resource['SUMleges_prefixlen1'])) # 0~1 normalization

# Normalization for features only available in traffic_fines_1_exp_prefix
if dataset_name == "traffic_fines_1_exp_prefix":
    for i in range(max_prefix):
        feature_name = "amount_prefixlen" + str(i+1)
        if np.var(X_dummy_no_resource[feature_name]) == 0:
            break
        X_dummy_no_resource[feature_name] = (X_dummy_no_resource[feature_name]-min(X_dummy_no_resource[feature_name]))/(max(X_dummy_no_resource[feature_name])-min(X_dummy_no_resource[feature_name])) # 0~1 normalization
    for i in range(max_prefix):
        feature_name = "expense_prefixlen" + str(i+1)
        if np.var(X_dummy_no_resource[feature_name]) == 0:
            break
        X_dummy_no_resource[feature_name] = (X_dummy_no_resource[feature_name]-min(X_dummy_no_resource[feature_name]))/(max(X_dummy_no_resource[feature_name])-min(X_dummy_no_resource[feature_name])) # 0~1 normalization
    X_dummy_no_resource['points_prefixlen1'] = (X_dummy_no_resource['points_prefixlen1']-min(X_dummy_no_resource['points_prefixlen1']))/(max(X_dummy_no_resource['points_prefixlen1'])-min(X_dummy_no_resource['points_prefixlen1'])) # 0~1 normalization

# Normalization for features only available in bpic2012_O_CANCELLED-COMPLETE_exp_prefix
if dataset_name == "bpic2012_O_CANCELLED-COMPLETE_exp_prefix":
    X_dummy_no_resource['AMOUNT_REQ_prefixlen1'] = (X_dummy_no_resource['AMOUNT_REQ_prefixlen1']-min(X_dummy_no_resource['AMOUNT_REQ_prefixlen1']))/(max(X_dummy_no_resource['AMOUNT_REQ_prefixlen1'])-min(X_dummy_no_resource['AMOUNT_REQ_prefixlen1'])) # 0~1 normalization

# Normalization for features only available in bpic2012_O_ACCEPTED-COMPLETE_exp_prefix
if dataset_name == "bpic2012_O_ACCEPTED-COMPLETE_exp_prefix":
    X_dummy_no_resource['AMOUNT_REQ_prefixlen1'] = (X_dummy_no_resource['AMOUNT_REQ_prefixlen1']-min(X_dummy_no_resource['AMOUNT_REQ_prefixlen1']))/(max(X_dummy_no_resource['AMOUNT_REQ_prefixlen1'])-min(X_dummy_no_resource['AMOUNT_REQ_prefixlen1'])) # 0~1 normalization

# Normalization for features only available in bpic2012_O_DECLINED-COMPLETE_exp_prefix
if dataset_name == "bpic2012_O_DECLINED-COMPLETE_exp_prefix":
    X_dummy_no_resource['AMOUNT_REQ_prefixlen1'] = (X_dummy_no_resource['AMOUNT_REQ_prefixlen1']-min(X_dummy_no_resource['AMOUNT_REQ_prefixlen1']))/(max(X_dummy_no_resource['AMOUNT_REQ_prefixlen1'])-min(X_dummy_no_resource['AMOUNT_REQ_prefixlen1'])) # 0~1 normalization

# Normalization for features only available in BPIC17_O_Cancelled_exp_prefix
if dataset_name == "BPIC17_O_Cancelled_exp_prefix":
    X_dummy_no_resource['RequestedAmount_prefixlen1'] = (X_dummy_no_resource['RequestedAmount_prefixlen1']-min(X_dummy_no_resource['RequestedAmount_prefixlen1']))/(max(X_dummy_no_resource['RequestedAmount_prefixlen1'])-min(X_dummy_no_resource['RequestedAmount_prefixlen1'])) # 0~1 normalization
    X_dummy_no_resource['FirstWithdrawalAmount_prefixlen1'] = (X_dummy_no_resource['FirstWithdrawalAmount_prefixlen1']-min(X_dummy_no_resource['FirstWithdrawalAmount_prefixlen1']))/(max(X_dummy_no_resource['FirstWithdrawalAmount_prefixlen1'])-min(X_dummy_no_resource['FirstWithdrawalAmount_prefixlen1'])) # 0~1 normalization
    X_dummy_no_resource['MonthlyCost_prefixlen1'] = (X_dummy_no_resource['MonthlyCost_prefixlen1']-min(X_dummy_no_resource['MonthlyCost_prefixlen1']))/(max(X_dummy_no_resource['MonthlyCost_prefixlen1'])-min(X_dummy_no_resource['MonthlyCost_prefixlen1'])) # 0~1 normalization
    X_dummy_no_resource['NumberOfTerms_prefixlen1'] = (X_dummy_no_resource['NumberOfTerms_prefixlen1']-min(X_dummy_no_resource['NumberOfTerms_prefixlen1']))/(max(X_dummy_no_resource['NumberOfTerms_prefixlen1'])-min(X_dummy_no_resource['NumberOfTerms_prefixlen1'])) # 0~1 normalization
    X_dummy_no_resource['OfferedAmount_prefixlen1'] = (X_dummy_no_resource['OfferedAmount_prefixlen1']-min(X_dummy_no_resource['OfferedAmount_prefixlen1']))/(max(X_dummy_no_resource['OfferedAmount_prefixlen1'])-min(X_dummy_no_resource['OfferedAmount_prefixlen1'])) # 0~1 normalization
    X_dummy_no_resource['CreditScore_prefixlen1'] = (X_dummy_no_resource['CreditScore_prefixlen1']-min(X_dummy_no_resource['CreditScore_prefixlen1']))/(max(X_dummy_no_resource['CreditScore_prefixlen1'])-min(X_dummy_no_resource['CreditScore_prefixlen1'])) # 0~1 normalization

# Normalization for features only available in BPIC17_O_Cancelled_exp_prefix
if dataset_name == "BPIC17_O_Accepted_exp_prefix":
    X_dummy_no_resource['RequestedAmount_prefixlen1'] = (X_dummy_no_resource['RequestedAmount_prefixlen1']-min(X_dummy_no_resource['RequestedAmount_prefixlen1']))/(max(X_dummy_no_resource['RequestedAmount_prefixlen1'])-min(X_dummy_no_resource['RequestedAmount_prefixlen1'])) # 0~1 normalization
    X_dummy_no_resource['FirstWithdrawalAmount_prefixlen1'] = (X_dummy_no_resource['FirstWithdrawalAmount_prefixlen1']-min(X_dummy_no_resource['FirstWithdrawalAmount_prefixlen1']))/(max(X_dummy_no_resource['FirstWithdrawalAmount_prefixlen1'])-min(X_dummy_no_resource['FirstWithdrawalAmount_prefixlen1'])) # 0~1 normalization
    X_dummy_no_resource['MonthlyCost_prefixlen1'] = (X_dummy_no_resource['MonthlyCost_prefixlen1']-min(X_dummy_no_resource['MonthlyCost_prefixlen1']))/(max(X_dummy_no_resource['MonthlyCost_prefixlen1'])-min(X_dummy_no_resource['MonthlyCost_prefixlen1'])) # 0~1 normalization
    X_dummy_no_resource['NumberOfTerms_prefixlen1'] = (X_dummy_no_resource['NumberOfTerms_prefixlen1']-min(X_dummy_no_resource['NumberOfTerms_prefixlen1']))/(max(X_dummy_no_resource['NumberOfTerms_prefixlen1'])-min(X_dummy_no_resource['NumberOfTerms_prefixlen1'])) # 0~1 normalization
    X_dummy_no_resource['OfferedAmount_prefixlen1'] = (X_dummy_no_resource['OfferedAmount_prefixlen1']-min(X_dummy_no_resource['OfferedAmount_prefixlen1']))/(max(X_dummy_no_resource['OfferedAmount_prefixlen1'])-min(X_dummy_no_resource['OfferedAmount_prefixlen1'])) # 0~1 normalization
    X_dummy_no_resource['CreditScore_prefixlen1'] = (X_dummy_no_resource['CreditScore_prefixlen1']-min(X_dummy_no_resource['CreditScore_prefixlen1']))/(max(X_dummy_no_resource['CreditScore_prefixlen1'])-min(X_dummy_no_resource['CreditScore_prefixlen1'])) # 0~1 normalization

# Normalization for features only available in BPIC17_O_Refused_exp_prefix
if dataset_name == "BPIC17_O_Refused_exp_prefix":
    X_dummy_no_resource['RequestedAmount_prefixlen1'] = (X_dummy_no_resource['RequestedAmount_prefixlen1']-min(X_dummy_no_resource['RequestedAmount_prefixlen1']))/(max(X_dummy_no_resource['RequestedAmount_prefixlen1'])-min(X_dummy_no_resource['RequestedAmount_prefixlen1'])) # 0~1 normalization
    X_dummy_no_resource['FirstWithdrawalAmount_prefixlen1'] = (X_dummy_no_resource['FirstWithdrawalAmount_prefixlen1']-min(X_dummy_no_resource['FirstWithdrawalAmount_prefixlen1']))/(max(X_dummy_no_resource['FirstWithdrawalAmount_prefixlen1'])-min(X_dummy_no_resource['FirstWithdrawalAmount_prefixlen1'])) # 0~1 normalization
    X_dummy_no_resource['MonthlyCost_prefixlen1'] = (X_dummy_no_resource['MonthlyCost_prefixlen1']-min(X_dummy_no_resource['MonthlyCost_prefixlen1']))/(max(X_dummy_no_resource['MonthlyCost_prefixlen1'])-min(X_dummy_no_resource['MonthlyCost_prefixlen1'])) # 0~1 normalization
    X_dummy_no_resource['NumberOfTerms_prefixlen1'] = (X_dummy_no_resource['NumberOfTerms_prefixlen1']-min(X_dummy_no_resource['NumberOfTerms_prefixlen1']))/(max(X_dummy_no_resource['NumberOfTerms_prefixlen1'])-min(X_dummy_no_resource['NumberOfTerms_prefixlen1'])) # 0~1 normalization
    X_dummy_no_resource['OfferedAmount_prefixlen1'] = (X_dummy_no_resource['OfferedAmount_prefixlen1']-min(X_dummy_no_resource['OfferedAmount_prefixlen1']))/(max(X_dummy_no_resource['OfferedAmount_prefixlen1'])-min(X_dummy_no_resource['OfferedAmount_prefixlen1'])) # 0~1 normalization
    X_dummy_no_resource['CreditScore_prefixlen1'] = (X_dummy_no_resource['CreditScore_prefixlen1']-min(X_dummy_no_resource['CreditScore_prefixlen1']))/(max(X_dummy_no_resource['CreditScore_prefixlen1'])-min(X_dummy_no_resource['CreditScore_prefixlen1'])) # 0~1 normalization


print("X_dummy_no_resource after prefix shape is :" + str(X_dummy_no_resource.shape))
print(X_dummy_no_resource.columns)

print("X_dummy_yes_resource after prefix shape is :" + str(X_dummy_yes_resource.shape))
print(X_dummy_yes_resource.columns)
X_dummy_yes_resource_column_list = pd.DataFrame(X_dummy_yes_resource.columns)
X_dummy_yes_resource_column_list.to_csv('columns_X_dummy_yes_resource.csv', index = False)

X_dummy_no_resource.to_csv("X_dummy_no_resource_" + dataset_name + "_" + str(max_prefix) + ".csv")
y_dummy_no_resource.to_csv("y_dummy_no_resource_" + dataset_name + "_" + str(max_prefix) + ".csv")
X_dummy_yes_resource.to_csv("X_dummy_yes_resource_" + dataset_name + "_" + str(max_prefix) + ".csv")
y_dummy_yes_resource.to_csv("y_dummy_yes_resource_" + dataset_name + "_" + str(max_prefix) + ".csv")

X_dummy_no_resource = X_dummy_no_resource.drop("case_id_prefixlen1",1) #Case_id column was added in order to give users info about each case. From here, the column is deleted for training and testing.
X_dummy_yes_resource = X_dummy_yes_resource.drop("case_id_prefixlen1",1) #Case_id column was added in order to give users info about each case. From here, the column is deleted for training and testing.

# Deleting sparse features
threshold_value = 0.1 #choose among 0.5, 0.1, 0.05, 0.01, 0.001

no_resource_non_sparse_features_index = X_dummy_no_resource.mean() > threshold_value
print(len(X_dummy_no_resource.columns))
print(len(no_resource_non_sparse_features_index))
X_dummy_no_resource = X_dummy_no_resource[X_dummy_no_resource.columns[no_resource_non_sparse_features_index]]
X_dummy_no_resource_column_list = pd.DataFrame(X_dummy_no_resource.columns)
X_dummy_no_resource_column_list.to_csv('columns_X_dummy_no_resource.csv', index = False)

yes_resource_non_sparse_features_index = X_dummy_yes_resource.mean() > threshold_value
print(len(X_dummy_yes_resource.columns))
print(len(yes_resource_non_sparse_features_index))
X_dummy_yes_resource.to_csv('X_dummy_yes_resource.csv', index = False)
yes_resource_non_sparse_features_index.to_csv('yes_resource_non_sparse_features_index.csv', index = False)
X_dummy_yes_resource = X_dummy_yes_resource[X_dummy_yes_resource.columns[yes_resource_non_sparse_features_index]]
X_dummy_yes_resource_column_list = pd.DataFrame(X_dummy_yes_resource.columns)
X_dummy_yes_resource_column_list.to_csv('columns_X_dummy_yes_resource.csv', index = False)

## Selecting top20 important features only
if dataset_name == "BPIC15_1_f2_exp_prefix":
    # RF TOP 20 important features
    X_dummy_yes_resource = pd.concat([X_dummy_no_resource,X_dummy_yes_resource[['n_cases_recent_prefixlen1', 'n_tasks_prefixlen1', 'n_cases_prefixlen1', 'n_acts_recent_prefixlen1', 'n_tasks_recent_prefixlen1', 'n_current_handoff_recent_prefixlen1', 'n_current_act_prefixlen1', 'n_current_act_recent_prefixlen1', 'n_current_handoff_prefixlen1', 'n_handoffs_recent_prefixlen1', 'n_handoffs_prefixlen1', 'n_acts_prefixlen1', 'n_current_handoff_recent_prefixlen2', 'busyness_recent_prefixlen1', 'n_current_handoff_prefixlen2']]], axis= 1)
    # XGB TOP 20 important features
    #X_dummy_yes_resource = pd.concat([X_dummy_no_resource, X_dummy_yes_resource[['n_tasks_recent_prefixlen1', 'n_tasks_prefixlen1', 'n_current_handoff_prefixlen1', 'n_cases_prefixlen1', 'n_current_act_recent_prefixlen1', 'n_current_act_prefixlen1', 'ratio_act_case_prefixlen1', 'n_current_handoff_recent_prefixlen1', 'n_cases_recent_prefixlen1', 'n_acts_recent_prefixlen1', 'ratio_act_case_recent_prefixlen1', 'n_tasks_recent_prefixlen2', 'ratio_act_case_recent_prefixlen2', 'busyness_recent_prefixlen1']]], axis = 1)

if dataset_name == "BPIC15_2_f2_exp_prefix":
    # RF TOP 20 important features
    X_dummy_yes_resource = pd.concat([X_dummy_no_resource,X_dummy_yes_resource[['n_current_handoff_prefixlen1', 'n_tasks_prefixlen1', 'n_current_act_prefixlen1', 'n_tasks_recent_prefixlen1', 'n_cases_prefixlen1', 'n_acts_prefixlen1', 'n_acts_recent_prefixlen1', 'n_current_act_recent_prefixlen1', 'n_current_handoff_recent_prefixlen1', 'n_handoffs_recent_prefixlen1', 'busyness_prefixlen1', 'ratio_act_case_prefixlen1', 'n_cases_recent_prefixlen1']]], axis= 1)
    # XGB TOP 20 important features
    #X_dummy_yes_resource = pd.concat([X_dummy_no_resource, X_dummy_yes_resource[['n_current_handoff_recent_prefixlen1', 'n_current_handoff_prefixlen1', 'n_tasks_prefixlen1', 'n_current_act_prefixlen1', 'n_current_act_recent_prefixlen1', 'busyness_prefixlen1', 'n_cases_prefixlen1', 'n_tasks_recent_prefixlen1', 'ratio_act_case_prefixlen1']]], axis = 1)

if dataset_name == "BPIC15_3_f2_exp_prefix":
    # RF TOP 20 important features
    X_dummy_yes_resource = pd.concat([X_dummy_no_resource,X_dummy_yes_resource[['n_current_act_recent_prefixlen1', 'n_tasks_prefixlen1', 'n_tasks_recent_prefixlen1', 'n_acts_prefixlen1', 'n_current_handoff_prefixlen1', 'n_cases_prefixlen1', 'n_acts_recent_prefixlen1', 'n_handoffs_recent_prefixlen1', 'busyness_recent_prefixlen1', 'n_current_handoff_recent_prefixlen1', 'n_cases_recent_prefixlen1', 'n_tasks_recent_prefixlen2']]], axis= 1)
    # XGB TOP 20 important features
    #X_dummy_yes_resource = pd.concat([X_dummy_no_resource, X_dummy_yes_resource[['n_cases_recent_prefixlen1', 'n_acts_recent_prefixlen1', 'n_current_act_recent_prefixlen1', 'n_tasks_prefixlen1', 'n_tasks_recent_prefixlen1', 'n_current_handoff_prefixlen1', 'n_cases_prefixlen1', 'n_current_act_prefixlen1', 'ratio_act_case_prefixlen1', 'n_acts_prefixlen1', 'n_current_handoff_recent_prefixlen1']]], axis = 1)

if dataset_name == "BPIC15_4_f2_exp_prefix":
    # RF TOP 20 important features
    X_dummy_yes_resource = pd.concat([X_dummy_no_resource,X_dummy_yes_resource[['n_current_handoff_prefixlen1', 'n_tasks_recent_prefixlen1', 'n_tasks_prefixlen1', 'n_cases_prefixlen1', 'ratio_act_case_recent_prefixlen1', 'n_current_handoff_recent_prefixlen1', 'n_current_act_prefixlen1', 'n_current_act_recent_prefixlen1', 'busyness_recent_prefixlen1', 'ratio_act_case_prefixlen1', 'busyness_prefixlen1', 'n_acts_recent_prefixlen1', 'n_cases_recent_prefixlen1']]], axis= 1)
    # XGB TOP 20 important features
    #X_dummy_yes_resource = pd.concat([X_dummy_no_resource, X_dummy_yes_resource[['n_current_handoff_prefixlen1', 'n_acts_recent_prefixlen1', 'n_tasks_recent_prefixlen1', 'n_tasks_prefixlen1', 'n_cases_recent_prefixlen1', 'n_current_handoff_recent_prefixlen1', 'n_current_act_recent_prefixlen1', 'ratio_act_case_recent_prefixlen1', 'ratio_act_case_prefixlen1', 'n_current_act_prefixlen1']]], axis = 1)

if dataset_name == "BPIC15_5_f2_exp_prefix":
    # RF TOP 20 important features
    X_dummy_yes_resource = pd.concat([X_dummy_no_resource,X_dummy_yes_resource[['n_current_handoff_prefixlen1', 'n_acts_prefixlen1', 'n_cases_recent_prefixlen1', 'n_handoffs_recent_prefixlen1', 'n_cases_prefixlen1', 'n_tasks_prefixlen1', 'n_current_handoff_recent_prefixlen1', 'n_tasks_recent_prefixlen1', 'ratio_act_case_prefixlen1', 'ent_act_prefixlen1', 'n_current_act_recent_prefixlen1', 'n_current_act_prefixlen1', 'ent_case_prefixlen1', 'n_acts_recent_prefixlen1', 'ratio_act_case_recent_prefixlen1']]], axis= 1)
    # XGB TOP 20 important features
    #X_dummy_yes_resource = pd.concat([X_dummy_no_resource, X_dummy_yes_resource[['n_tasks_recent_prefixlen1', 'n_current_handoff_prefixlen1', 'n_cases_prefixlen1', 'n_current_act_prefixlen1', 'n_tasks_prefixlen1', 'n_current_act_recent_prefixlen1', 'ent_act_prefixlen1', 'ratio_act_case_recent_prefixlen1', 'ratio_act_case_prefixlen1', 'n_acts_recent_prefixlen1', 'n_acts_prefixlen1', 'n_cases_recent_prefixlen1']]], axis = 1)

if dataset_name == "traffic_fines_1_exp_prefix":
    # RF TOP 20 important features
    X_dummy_yes_resource = pd.concat([X_dummy_no_resource,X_dummy_yes_resource[['n_tasks_prefixlen1', 'n_tasks_recent_prefixlen1', 'n_current_act_recent_prefixlen1', 'polarity_current_handoff_prefixlen1', 'n_current_handoff_prefixlen1', 'n_cases_recent_prefixlen1', 'n_cases_prefixlen1', 'busyness_recent_prefixlen1', 'n_current_act_prefixlen1', 'n_current_handoff_recent_prefixlen1', 'n_handoffs_prefixlen1', 'n_cases_recent_prefixlen2']]], axis= 1)
    # XGB TOP 20 important features
    #X_dummy_yes_resource = pd.concat([X_dummy_no_resource, X_dummy_yes_resource[['n_tasks_prefixlen1', 'polarity_current_handoff_prefixlen1', 'n_cases_prefixlen1', 'n_tasks_recent_prefixlen1', 'n_current_act_recent_prefixlen1', 'n_cases_recent_prefixlen1', 'n_cases_recent_prefixlen2', 'n_handoffs_prefixlen1', 'n_current_act_recent_prefixlen2', 'ent_act_recent_prefixlen1', 'n_tasks_recent_prefixlen2']]], axis = 1)

if dataset_name == "bpic2012_O_CANCELLED-COMPLETE_exp_prefix":
    # RF TOP 20 important features
    X_dummy_yes_resource = pd.concat([X_dummy_no_resource,X_dummy_yes_resource[['ratio_act_case_prefixlen1', 'n_acts_prefixlen1', 'ratio_act_case_recent_prefixlen1', 'n_tasks_prefixlen1', 'busyness_prefixlen1', 'n_cases_recent_prefixlen1', 'n_cases_prefixlen1', 'n_acts_recent_prefixlen1', 'n_current_act_prefixlen1', 'ent_handoff_prefixlen1', 'n_current_act_recent_prefixlen1', 'n_current_handoff_recent_prefixlen1']]], axis= 1)
    # XGB TOP 20 important features
    #X_dummy_yes_resource = pd.concat([X_dummy_no_resource, X_dummy_yes_resource[['n_current_act_recent_prefixlen1', 'n_current_handoff_prefixlen1', 'n_cases_prefixlen1', 'n_cases_recent_prefixlen1', 'n_acts_prefixlen1', 'n_acts_recent_prefixlen1', 'n_tasks_recent_prefixlen1', 'busyness_prefixlen1', 'n_current_act_prefixlen1', 'ratio_act_case_recent_prefixlen1', 'n_tasks_prefixlen1']]], axis = 1)

if dataset_name == "bpic2012_O_ACCEPTED-COMPLETE_exp_prefix":
    # RF TOP 20 important features
    X_dummy_yes_resource = pd.concat([X_dummy_no_resource,X_dummy_yes_resource[['polarity_current_handoff_recent_prefixlen1', 'polarity_current_handoff_prefixlen1', 'n_current_handoff_recent_prefixlen1', 'n_tasks_prefixlen1', 'n_current_act_prefixlen1', 'n_current_handoff_prefixlen1', 'n_cases_prefixlen1', 'n_tasks_recent_prefixlen1', 'n_current_act_recent_prefixlen1', 'n_cases_recent_prefixlen1', 'n_acts_recent_prefixlen1', 'busyness_prefixlen1']]], axis= 1)
    # XGB TOP 20 important features
    #X_dummy_yes_resource = pd.concat([X_dummy_no_resource, X_dummy_yes_resource[['n_current_act_recent_prefixlen1', 'n_current_handoff_prefixlen1', 'n_cases_prefixlen1', 'n_cases_recent_prefixlen1', 'n_acts_prefixlen1', 'n_acts_recent_prefixlen1', 'n_tasks_recent_prefixlen1', 'busyness_prefixlen1', 'n_current_act_prefixlen1', 'ratio_act_case_recent_prefixlen1', 'n_tasks_prefixlen1']]], axis = 1)

if dataset_name == "bpic2012_O_DECLINED-COMPLETE_exp_prefix":
    # RF TOP 20 important features
    X_dummy_yes_resource = pd.concat([X_dummy_no_resource,X_dummy_yes_resource[['n_tasks_prefixlen1', 'n_current_handoff_recent_prefixlen1', 'polarity_current_handoff_prefixlen1', 'n_current_act_recent_prefixlen1', 'n_current_handoff_prefixlen1', 'busyness_recent_prefixlen1', 'busyness_prefixlen1', 'n_current_act_prefixlen1', 'n_tasks_recent_prefixlen1', 'n_cases_recent_prefixlen1', 'n_handoffs_prefixlen1', 'n_cases_recent_prefixlen2']]], axis= 1)
    # XGB TOP 20 important features
    #X_dummy_yes_resource = pd.concat([X_dummy_no_resource, X_dummy_yes_resource[['n_current_handoff_prefixlen1', 'n_tasks_prefixlen1', 'n_cases_prefixlen1', 'busyness_prefixlen1', 'n_tasks_recent_prefixlen1', 'n_current_handoff_recent_prefixlen1', 'n_handoffs_prefixlen1', 'n_current_act_prefixlen1', 'polarity_current_handoff_prefixlen1', 'polarity_current_handoff_recent_prefixlen1', 'ratio_act_case_prefixlen1', 'busyness_recent_prefixlen1']]], axis = 1)

if dataset_name == "BPIC17_O_Accepted_exp_prefix":
    # RF TOP 20 important features
    X_dummy_yes_resource = pd.concat([X_dummy_no_resource,X_dummy_yes_resource[['n_current_act_recent_prefixlen1', 'n_current_handoff_recent_prefixlen1', 'n_current_act_prefixlen1', 'n_current_handoff_prefixlen1', 'n_cases_prefixlen1', 'n_cases_recent_prefixlen1', 'busyness_recent_prefixlen1', 'busyness_prefixlen1', 'n_tasks_prefixlen1', 'n_handoffs_prefixlen1', 'n_tasks_recent_prefixlen1']]], axis= 1)
    # XGB TOP 20 important features
    #X_dummy_yes_resource = pd.concat([X_dummy_no_resource, X_dummy_yes_resource[['n_current_act_prefixlen1', 'n_current_act_recent_prefixlen1', 'n_cases_recent_prefixlen1', 'n_current_handoff_recent_prefixlen1', 'n_current_handoff_prefixlen1', 'n_tasks_recent_prefixlen1', 'n_handoffs_prefixlen1', 'n_acts_prefixlen1']]], axis = 1)

if dataset_name == "BPIC17_O_Cancelled_exp_prefix":
    # RF TOP 20 important features
    X_dummy_yes_resource = pd.concat([X_dummy_no_resource,X_dummy_yes_resource[['n_tasks_recent_prefixlen1', 'n_current_handoff_prefixlen1', 'n_current_act_recent_prefixlen1', 'n_current_act_prefixlen1', 'n_current_handoff_recent_prefixlen1', 'busyness_prefixlen1', 'n_cases_prefixlen1', 'busyness_recent_prefixlen1', 'n_tasks_prefixlen1', 'polarity_current_handoff_prefixlen1', 'n_cases_recent_prefixlen1']]], axis= 1)
    # XGB TOP 20 important features
    #X_dummy_yes_resource = pd.concat([X_dummy_no_resource, X_dummy_yes_resource[['n_tasks_recent_prefixlen1', 'n_current_handoff_prefixlen1', 'n_current_act_recent_prefixlen1', 'busyness_prefixlen1', 'n_current_handoff_recent_prefixlen1', 'polarity_current_handoff_prefixlen1', 'n_cases_recent_prefixlen1', 'ratio_act_case_recent_prefixlen1', 'n_handoffs_prefixlen1', 'n_current_act_prefixlen1', 'n_tasks_prefixlen1', 'n_acts_prefixlen1']]], axis = 1)

if dataset_name == "BPIC17_O_Refused_exp_prefix":
    # RF TOP 20 important features
    X_dummy_yes_resource = pd.concat([X_dummy_no_resource,X_dummy_yes_resource[['n_current_act_prefixlen1', 'n_current_handoff_recent_prefixlen1', 'n_tasks_prefixlen1', 'n_current_handoff_prefixlen1', 'n_current_act_recent_prefixlen1', 'n_cases_prefixlen1', 'busyness_prefixlen1', 'busyness_recent_prefixlen1', 'n_tasks_recent_prefixlen1', 'n_handoffs_prefixlen1', 'n_cases_recent_prefixlen1', 'n_tasks_recent_prefixlen2', 'n_acts_recent_prefixlen1']]], axis= 1)
    # XGB TOP 20 important features
    #X_dummy_yes_resource = pd.concat([X_dummy_no_resource, X_dummy_yes_resource[['n_current_handoff_recent_prefixlen1', 'n_current_act_recent_prefixlen1', 'n_current_handoff_prefixlen1', 'busyness_prefixlen1', 'n_cases_recent_prefixlen1', 'n_current_act_prefixlen1', 'n_tasks_prefixlen1', 'n_tasks_recent_prefixlen1', 'n_cases_prefixlen1']]], axis = 1)





##Pearson correlation coefficient calculation as a reference for feature selection
pearson_corr = pd.DataFrame(index = range(len(X_dummy_no_resource.columns)), columns = ["feature_name","Pearson_correlation"])
pearson_corr.iloc[:,0] = X_dummy_no_resource.columns
print(y_dummy_no_resource.columns)
for i in range(len(X_dummy_no_resource.columns)-1):
    print(pearsonr(X_dummy_no_resource[X_dummy_no_resource.columns[i]], pd.DataFrame(y_dummy_no_resource)[pd.DataFrame(y_dummy_no_resource).columns[0]]))
    pearson_corr.iloc[i,1] = pearsonr(X_dummy_no_resource[X_dummy_no_resource.columns[i]], pd.DataFrame(y_dummy_no_resource)[pd.DataFrame(y_dummy_no_resource).columns[0]])[0]
pearson_name = "pearson_correlation_coefficient_no_resource_" + dataset_name + "_" + str(max_prefix) + ".csv"
pearson_corr.to_csv(pearson_name, index = False)

##Pearson correlation coefficient calculation as a reference for feature selection
pearson_corr = pd.DataFrame(index = range(len(X_dummy_yes_resource.columns)), columns = ["feature_name","Pearson_correlation"])
pearson_corr.iloc[:,0] = X_dummy_yes_resource.columns
print(y_dummy_yes_resource.columns)
for i in range(len(X_dummy_yes_resource.columns)-1):
    print(pearsonr(X_dummy_yes_resource[X_dummy_yes_resource.columns[i]], pd.DataFrame(y_dummy_yes_resource)[pd.DataFrame(y_dummy_yes_resource).columns[0]]))
    pearson_corr.iloc[i,1] = pearsonr(X_dummy_yes_resource[X_dummy_yes_resource.columns[i]], pd.DataFrame(y_dummy_yes_resource)[pd.DataFrame(y_dummy_yes_resource).columns[0]])[0]
pearson_name = "pearson_correlation_coefficient_yes_resource_" + dataset_name + "_" + str(max_prefix) + ".csv"
pearson_corr.to_csv(pearson_name, index = False)


'''
# Mute this part if you do not want to reduce dimensionality
# Dimensionality reduction using PCA
pca_no_resource = PCA(n_components = 10)
pca_no_resource.fit(X_dummy_no_resource)
print("pca_no_resource_variance_explained:")
print(pca_no_resource.explained_variance_ratio_)
X_dummy_no_resource = pd.DataFrame(pca_no_resource.transform(X_dummy_no_resource))
X_dummy_no_resource = X_dummy_no_resource.iloc[:,0:5] #index changes depending on the variance explained.
pca_yes_resource = PCA(n_components = 10)
pca_yes_resource.fit(X_dummy_yes_resource)
print("pca_yes_resource_variance_explained:")
print(pca_yes_resource.explained_variance_ratio_)
X_dummy_yes_resource = pd.DataFrame(pca_yes_resource.transform(X_dummy_yes_resource))
X_dummy_yes_resource = X_dummy_yes_resource.iloc[:,0:5] #index changes depending on the variance explained.
'''
# [no resource] Setting k for k-fold cv
nsplits_no_resource = 5 # Set the number of k for cross validation
kf_no_resource = KFold(n_splits=nsplits_no_resource)
kf_no_resource.get_n_splits(X_dummy_no_resource)

# [yes resource] Setting k for k-fold cv
nsplits_yes_resource = 5 # Set the number of k for cross validation
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
index_no_resource = 0 #This index value is used to store auc into list created right above
for train_index, test_index in kf_no_resource.split(X_dummy_no_resource):
    # Splitting data into training and test set
    X_train_no_resource, X_test_no_resource = X_dummy_no_resource.iloc[train_index], X_dummy_no_resource.iloc[test_index]
    y_train_no_resource, y_test_no_resource = y_dummy_no_resource.iloc[train_index], y_dummy_no_resource.iloc[test_index]
    train_no_resource = pd.concat([X_train_no_resource, y_train_no_resource], axis = 1)
    test_no_resource = pd.concat([X_test_no_resource, y_test_no_resource], axis = 1)

    # Deleting columns with feature importances = 0 for no resource
    # These column indices are from Random Forest feature importances function
    cols_del_no_resource = [] # Indices with feature importances = 0 were not found yet. Please fill in the bracket with indices later.
    X_train_no_resource = X_train_no_resource.drop(X_train_no_resource.columns[cols_del_no_resource], axis = 1)
    X_test_no_resource = X_test_no_resource.drop(X_test_no_resource.columns[cols_del_no_resource], axis = 1)

    print("X_train_no_resource after prefix shape is :" + str(X_train_no_resource.shape))
    print(len(X_train_no_resource))
    print("X_test_no_resource after prefix shape is :" + str(X_test_no_resource.shape))
    print(len(X_test_no_resource))

    first_digit_parameters = [x for x in itertools.product((50, 100, 200, 500, 1000), repeat=1)]
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
    all_digit_parameters = first_digit_parameters # one hidden layer
    # all_digit_parameters = first_digit_parameters + second_digit_parameters # two hidden layers
    # all_digit_parameters = first_digit_parameters + second_digit_parameters + third_digit_parameters # three hidden layers
    learning_rate_init_parameters = [0.1, 0.01, 0.001]

    parameters = {'hidden_layer_sizes': all_digit_parameters,
		  'learning_rate_init': learning_rate_init_parameters}

        
    print("DNN started")
    start = time.time()

    #Making dnn classifier
    dnn_classifier_no_resource = MLPClassifier(max_iter = 10000, activation = 'relu')
    dnn_classifier_no_resource_clf = RandomizedSearchCV(dnn_classifier_no_resource, parameters, n_jobs = num_cpu_cores, cv = 5)

    #Fitting the dnn classifier with training set
    dnn_classifier_no_resource_clf.fit(X_train_no_resource, y_train_no_resource.values.ravel())
    print("DNN best parameter is:")
    print(dnn_classifier_no_resource_clf.best_params_)

    print("DNN time: ", time.time() - start)

    n_tree = [100, 200, 300, 400, 500]
    max_depth = [20, 40, 60]
    min_samples_split = [5, 10, 20]

    parameters = {'n_estimators': n_tree,
                  'max_depth': max_depth,
                  'min_samples_split': min_samples_split}

    print("RF started")
    start = time.time()

    #Making rf classifier
    rf_classifier_no_resource = RandomForestClassifier(n_jobs = num_cpu_cores)
    rf_classifier_no_resource_clf = RandomizedSearchCV(rf_classifier_no_resource, parameters, n_jobs = num_cpu_cores, cv=5)

    #Fitting the rf classifier with training set
    rf_classifier_no_resource_clf.fit(X_train_no_resource, y_train_no_resource.values.ravel())
    print("RF best parameter is:")
    print(rf_classifier_no_resource_clf.best_params_)

    print("RF time: ", time.time() - start)

    parameters = {'n_estimators': n_tree,
                  'learning_rate': learning_rate_init_parameters,
                  'max_depth': max_depth}

    print("XGB started")
    start = time.time()

    #Making xgb classifier
    xgb_classifier_no_resource = XGBClassifier(n_jobs = num_cpu_cores)
    # xgb_classifier_no_resource_clf = xgb_classifier_no_resource
    xgb_classifier_no_resource_clf = RandomizedSearchCV(xgb_classifier_no_resource, parameters, n_jobs = num_cpu_cores, cv=5)
    
    #Fitting the xgb classifier with training set
    xgb_classifier_no_resource_clf.fit(X_train_no_resource, y_train_no_resource.values.ravel())
    print("XGB best parameter is:")
    print(xgb_classifier_no_resource_clf.best_params_)

    print("XGB time: ", time.time() - start)

    #Prediction using dnn classifier
    dnn_pred_no_resource = dnn_classifier_no_resource_clf.predict(X_test_no_resource)

    #Prediction using rf classifier
    rf_pred_no_resource = rf_classifier_no_resource_clf.predict(X_test_no_resource)

    #Prediction using xgb classifier
    xgb_pred_no_resource = xgb_classifier_no_resource_clf.predict(X_test_no_resource)

    #Calculating auc
    dnn_auc_no_resource = roc_auc_score(y_test_no_resource, dnn_pred_no_resource)
    rf_auc_no_resource = roc_auc_score(y_test_no_resource, rf_pred_no_resource)
    xgb_auc_no_resource = roc_auc_score(y_test_no_resource, xgb_pred_no_resource)
    print("AUC of DNN classifier of" + " fold " + str(index_no_resource+1) + " no resource is: " + str(dnn_auc_no_resource))
    print("AUC of RF classifier of" + " fold " + str(index_no_resource+1) + " no resource is: " + str(rf_auc_no_resource))
    print("AUC of XGB classifier of" + " fold " + str(index_no_resource+1) + " no resource is: " + str(xgb_auc_no_resource))

    #Storing prediction results
    dnn_auc_no_resource_arr[index_no_resource] = dnn_auc_no_resource
    rf_auc_no_resource_arr[index_no_resource] = rf_auc_no_resource
    xgb_auc_no_resource_arr[index_no_resource] = xgb_auc_no_resource

    index_no_resource += 1
dnn_auc_no_resource_arr[5] = "Average" #To let users figure out that dnn_auc_no_resource[6] value is the average when seeing the csv file
dnn_auc_no_resource_arr[6] = (dnn_auc_no_resource_arr[0]+dnn_auc_no_resource_arr[1]+dnn_auc_no_resource_arr[2]+dnn_auc_no_resource_arr[3]+dnn_auc_no_resource_arr[4])/5 #Averaging the auc values from kfold cv
rf_auc_no_resource_arr[5] = "Average"
rf_auc_no_resource_arr[6] = (rf_auc_no_resource_arr[0]+rf_auc_no_resource_arr[1]+rf_auc_no_resource_arr[2]+rf_auc_no_resource_arr[3]+rf_auc_no_resource_arr[4])/5
xgb_auc_no_resource_arr[5] = "Average"
xgb_auc_no_resource_arr[6] = (xgb_auc_no_resource_arr[0]+xgb_auc_no_resource_arr[1]+xgb_auc_no_resource_arr[2]+xgb_auc_no_resource_arr[3]+xgb_auc_no_resource_arr[4])/5
print("Average AUC of DNN classifier no resource is: " + str(dnn_auc_no_resource_arr[6]))
print("Average AUC of RF classifier no resource is: " + str(rf_auc_no_resource_arr[6]))
print("Average AUC of XGB classifier no resource is: " + str(xgb_auc_no_resource_arr[6]))

#Feature importances
pd.DataFrame(X_train_no_resource.columns.tolist()).to_csv('column_names_no_resource_' + dataset_name + '_' + str(max_prefix) + '.csv')

##RF feature importances
no_resource_rf_feature_imp_name = "rf_feature_importances_no_resource_" + dataset_name + "_" + str(max_prefix) + ".csv"
np.savetxt(no_resource_rf_feature_imp_name, rf_classifier_no_resource.fit(X_train_no_resource, y_train_no_resource.values.ravel()).feature_importances_)
#np.savetxt("rf_feature_importances_index_no_resource.csv", np.argsort(rf_classifier_no_resource_clf.feature_importances_))

##XGB feature importances
no_resource_xgb_feature_imp_name = "xgb_feature_importances_no_resource_" + dataset_name + "_" + str(max_prefix) + ".csv"
np.savetxt(no_resource_xgb_feature_imp_name, xgb_classifier_no_resource.fit(X_train_no_resource, y_train_no_resource.values.ravel()).feature_importances_)
#np.savetxt("xgb_feature_importances_index_no_resource.csv", np.argsort(xgb_classifier_no_resource_clf.feature_importances_))



# [yes resource] Starting k-fold cv
index_yes_resource = 0 #This index value is used to store auc into list created right above
for train_index, test_index in kf_yes_resource.split(X_dummy_yes_resource):
    # Splitting data into training and test set
    X_train_yes_resource, X_test_yes_resource = X_dummy_yes_resource.iloc[train_index], X_dummy_yes_resource.iloc[test_index]
    y_train_yes_resource, y_test_yes_resource = y_dummy_yes_resource.iloc[train_index], y_dummy_yes_resource.iloc[test_index]
    train_yes_resource = pd.concat([X_train_yes_resource, y_train_yes_resource], axis = 1)
    test_yes_resource = pd.concat([X_test_yes_resource, y_test_yes_resource], axis = 1)

    print("X_train_yes_resource after prefix shape is :" + str(X_train_yes_resource.shape))
    print(len(X_train_yes_resource))
    print("X_test_yes_resource after prefix shape is :" + str(X_test_yes_resource.shape))
    print(len(X_test_yes_resource))

    first_digit_parameters = [x for x in itertools.product((50, 100, 200, 500, 1000), repeat=1)]
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
    all_digit_parameters = first_digit_parameters # one hidden layer
    # all_digit_parameters = first_digit_parameters + second_digit_parameters # two hidden layers
    # all_digit_parameters = first_digit_parameters + second_digit_parameters + third_digit_parameters # three hidden layers

    learning_rate_init_parameters = [0.1, 0.01, 0.001]
    

    parameters = {'hidden_layer_sizes': all_digit_parameters,
		  'learning_rate_init': learning_rate_init_parameters}

        
    print("DNN started")
    start = time.time()

    #Making dnn classifier
    dnn_classifier_yes_resource = MLPClassifier(max_iter = 10000, activation = 'relu')
    dnn_classifier_yes_resource_clf = RandomizedSearchCV(dnn_classifier_yes_resource, parameters, n_jobs = num_cpu_cores, cv = 5)

    #Fitting the dnn classifier with training set
    dnn_classifier_yes_resource_clf.fit(X_train_yes_resource, y_train_yes_resource.values.ravel())
    print("DNN best parameter is:")
    print(dnn_classifier_yes_resource_clf.best_params_)

    print("DNN time: ", time.time() - start)

    n_tree = [100, 200, 300, 400, 500]
    max_depth = [20, 40, 60]
    min_samples_split = [5, 10, 20]

    parameters = {'n_estimators': n_tree,
                  'max_depth': max_depth,
                  'min_samples_split': min_samples_split}

    print("RF started")
    start = time.time()

    #Making rf classifier
    rf_classifier_yes_resource = RandomForestClassifier(n_jobs = num_cpu_cores)
    rf_classifier_yes_resource_clf = RandomizedSearchCV(rf_classifier_yes_resource, parameters, n_jobs = num_cpu_cores, cv=5)

    #Fitting the rf classifier with training set
    rf_classifier_yes_resource_clf.fit(X_train_yes_resource, y_train_yes_resource.values.ravel())
    print("RF best parameter is:")
    print(rf_classifier_yes_resource_clf.best_params_)

    print("RF time: ", time.time() - start)

    parameters = {'n_estimators': n_tree,
                  'learning_rate': learning_rate_init_parameters,
                  'max_depth': max_depth}


    print("XGB started")
    start = time.time()

    #Making xgb classifier
    xgb_classifier_yes_resource = XGBClassifier(n_jobs = num_cpu_cores)
    xgb_classifier_yes_resource_clf = RandomizedSearchCV(xgb_classifier_yes_resource, parameters, n_jobs = num_cpu_cores, cv=5)

    #Fitting the xgb classifier with training set
    xgb_classifier_yes_resource_clf.fit(X_train_yes_resource, y_train_yes_resource.values.ravel())
    print("XGB best parameter is:")
    print(xgb_classifier_yes_resource_clf.best_params_)

    print("XGB time: ", time.time() - start)

    #Prediction using dnn classifier
    dnn_pred_yes_resource = dnn_classifier_yes_resource_clf.predict(X_test_yes_resource)

    #Prediction using rf classifier
    rf_pred_yes_resource = rf_classifier_yes_resource_clf.predict(X_test_yes_resource)

    #Prediction using xgb classifier
    xgb_pred_yes_resource = xgb_classifier_yes_resource_clf.predict(X_test_yes_resource)

    #Calculating auc
    dnn_auc_yes_resource = roc_auc_score(y_test_yes_resource, dnn_pred_yes_resource)
    rf_auc_yes_resource = roc_auc_score(y_test_yes_resource, rf_pred_yes_resource)
    xgb_auc_yes_resource = roc_auc_score(y_test_yes_resource, xgb_pred_yes_resource)
    print("AUC of DNN classifier of" + " fold " + str(index_yes_resource+1) + " yes resource is: " + str(dnn_auc_yes_resource))
    print("AUC of RF classifier of" + " fold " + str(index_yes_resource+1) + " yes resource is: " + str(rf_auc_yes_resource))
    print("AUC of XGB classifier of" + " fold " + str(index_yes_resource+1) + " yes resource is: " + str(xgb_auc_yes_resource))

    #Storing prediction results
    dnn_auc_yes_resource_arr[index_yes_resource] = dnn_auc_yes_resource
    rf_auc_yes_resource_arr[index_yes_resource] = rf_auc_yes_resource
    xgb_auc_yes_resource_arr[index_yes_resource] = xgb_auc_yes_resource

    index_yes_resource += 1
dnn_auc_yes_resource_arr[5] = "Average" #To let users figure out that dnn_auc_yes_resource[6] value is the average when seeing the csv file
dnn_auc_yes_resource_arr[6] = (dnn_auc_yes_resource_arr[0]+dnn_auc_yes_resource_arr[1]+dnn_auc_yes_resource_arr[2]+dnn_auc_yes_resource_arr[3]+dnn_auc_yes_resource_arr[4])/5 #Averaging the auc values from kfold cv
rf_auc_yes_resource_arr[5] = "Average"
rf_auc_yes_resource_arr[6] = (rf_auc_yes_resource_arr[0]+rf_auc_yes_resource_arr[1]+rf_auc_yes_resource_arr[2]+rf_auc_yes_resource_arr[3]+rf_auc_yes_resource_arr[4])/5
xgb_auc_yes_resource_arr[5] = "Average"
xgb_auc_yes_resource_arr[6] = (xgb_auc_yes_resource_arr[0]+xgb_auc_yes_resource_arr[1]+xgb_auc_yes_resource_arr[2]+xgb_auc_yes_resource_arr[3]+xgb_auc_yes_resource_arr[4])/5
print("Average AUC of DNN classifier yes resource is: " + str(dnn_auc_yes_resource_arr[6]))
print("Average AUC of RF classifier yes resource is: " + str(rf_auc_yes_resource_arr[6]))
print("Average AUC of xgb classifier yes resource is: " + str(xgb_auc_yes_resource_arr[6]))

#Feature importances
pd.DataFrame(X_train_yes_resource.columns.tolist()).to_csv('column_names_yes_resource_' + dataset_name + '_' + str(max_prefix) + '.csv')

##RF feature importances
yes_resource_rf_feature_imp_name = "rf_feature_importances_yes_resource_" + dataset_name + "_" + str(max_prefix) + ".csv"
np.savetxt(yes_resource_rf_feature_imp_name, rf_classifier_yes_resource.fit(X_train_yes_resource, y_train_yes_resource.values.ravel()).feature_importances_)
#np.savetxt("rf_feature_importances_index_yes_resource.csv", np.argsort(rf_classifier_yes_resource_clf.feature_importances_))

##XGB feature importances
yes_resource_xgb_feature_imp_name = "xgb_feature_importances_yes_resource_" + dataset_name + "_" + str(max_prefix) + ".csv"
np.savetxt(yes_resource_xgb_feature_imp_name, xgb_classifier_yes_resource.fit(X_train_yes_resource, y_train_yes_resource.values.ravel()).feature_importances_)
#np.savetxt("xgb_feature_importances_index_yes_resource.csv", np.argsort(xgb_classifier_yes_resource_clf.feature_importances_))
system_end = time.time()
print("Total time:")
print(system_end - system_start)