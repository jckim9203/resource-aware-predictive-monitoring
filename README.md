# resource-aware-predictive-monitoring

The experiments use 13 datasets in total (4 BPIC 2011 datasets / 5 BPIC 2015 datasets / 3 BPIC 2017 datasets / 1 traffic fines dataset). Click the following dataset to have access to the website containing the original dataset: [BPIC 2011](https://www.win.tue.nl/bpi/doku.php?id=2011:challenge), [BPIC 2015](https://www.win.tue.nl/bpi/doku.php?id=2015:challenge), [BPIC 2017](https://www.win.tue.nl/bpi/doku.php?id=2017:challenge), and [traffic fines](https://data.4tu.nl/articles/Road_Traffic_Fine_Management_Process/12683249). The details of the preprocessing of the datasets are available in [Teinemaa et al. (2019)](https://dl.acm.org/doi/abs/10.1145/3301300).


## 1. extract_resoruce_features_dtype_conversion.py
-Extracting features related to resource experience

## 2. (dataset_name)_prefixing_code.r
-Adding prefix variable

## 3. train_evaluate_final_with_resource_index_encoding.py
-Training and evaluating the models
