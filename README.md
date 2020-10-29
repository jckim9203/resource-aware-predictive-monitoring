# resource-aware-predictive-monitoring
The materials in this repository implements the experiments for the article "Encoding resource experience for predictive process monitoring" by Jongchan Kim, Marco Comuzzi, Marlon Dumas, Fabrizio Maria Maggi and Irene Teinemaa. The experiments use 13 datasets in total (4 BPIC 2011 datasets / 5 BPIC 2015 datasets / 3 BPIC 2017 datasets / 1 traffic fines dataset). Click the following dataset to have access to the website containing the original dataset: [BPIC 2011](https://www.win.tue.nl/bpi/doku.php?id=2011:challenge), [BPIC 2015](https://www.win.tue.nl/bpi/doku.php?id=2015:challenge), [BPIC 2017](https://www.win.tue.nl/bpi/doku.php?id=2017:challenge), [traffic fines](https://data.4tu.nl/articles/Road_Traffic_Fine_Management_Process/12683249). The details of the preprocessing of the datasets are available in [Teinemaa et al. (2019)](https://dl.acm.org/doi/abs/10.1145/3301300).

The materials in this repository provides implementation of extracting resource-aware features, index encoding of features, and training & testing of models to predict the outcome of an ongoing case.

## 1. extract_resource_features_dtype_conversion.py
-This code extracts 38 resource-aware features defined by four dimensions related to resource experience: Recency, Context, Target, Aspect.
-The details on four dimensions is presented as below:
* __Recency__: Long-term, Short-term
* __Context__: General, Current case, Current task, Current handoff
* __Target__: Work item, Case, Task, Handoff
* __Aspect__: Frequency, Performance, Specialization, Generalization, Busyness

## 2. <dataset_name>_prefixing_code.r
-This code adds a new variable indicating the value of the prefix that corresponding event belongs to.

## 3. train_evaluate_final_with_resource_index_encoding.py
-This code uses "index encoding" to encode the features.
-After index encoding, this code trains and evaluates the models using two decision tree-based classifiers, Random Forest and XGBoost.
