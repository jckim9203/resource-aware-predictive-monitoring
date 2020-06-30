#Currently for bpic17_O_Accepted
#Run the code separately for each prefix
import pandas as pd
dataset_name = "BPIC17_O_Accepted_exp_prefix"
#In below, data dummy csv file can be obtained by running the "train_evaluate_final~.py" code. The code will make memory error but anyway, will be successful in making data dummy csv file.
data_dummy = pd.read_pickle('data_dummy.pkl')
#data_dummy = pd.read_csv('data_dummy.csv', sep=",")
data_dummy_columns_list = data_dummy.columns

max_prefix = 20

##########prefix1##########
data_dummy_prefix_prefixlen1 = pd.DataFrame(index=range(len(data_dummy[data_dummy['prefix']==1])), columns=range(len(data_dummy.columns)))
for col in data_dummy_columns_list:
	print(col)
	if col not in ("case_id","Action","EventOrigin","lifecycle.transition","Accepted","Selected","AMOUNT_REQ","case_len","label"):
		data_dummy_prefix_prefixlen1[str(col) + '_prefixlen' + str(1)] = 0
		data_dummy_prefix_prefixlen1[str(col) + '_prefixlen' + str(2)] = 0
		data_dummy_prefix_prefixlen1[str(col) + '_prefixlen' + str(3)] = 0
		data_dummy_prefix_prefixlen1[str(col) + '_prefixlen' + str(4)] = 0
		data_dummy_prefix_prefixlen1[str(col) + '_prefixlen' + str(5)] = 0
		data_dummy_prefix_prefixlen1[str(col) + '_prefixlen' + str(6)] = 0
		data_dummy_prefix_prefixlen1[str(col) + '_prefixlen' + str(7)] = 0
		data_dummy_prefix_prefixlen1[str(col) + '_prefixlen' + str(8)] = 0
		data_dummy_prefix_prefixlen1[str(col) + '_prefixlen' + str(9)] = 0
		data_dummy_prefix_prefixlen1[str(col) + '_prefixlen' + str(10)] = 0
		data_dummy_prefix_prefixlen1[str(col) + '_prefixlen' + str(11)] = 0
		data_dummy_prefix_prefixlen1[str(col) + '_prefixlen' + str(12)] = 0
		data_dummy_prefix_prefixlen1[str(col) + '_prefixlen' + str(13)] = 0
		data_dummy_prefix_prefixlen1[str(col) + '_prefixlen' + str(14)] = 0
		data_dummy_prefix_prefixlen1[str(col) + '_prefixlen' + str(15)] = 0
		data_dummy_prefix_prefixlen1[str(col) + '_prefixlen' + str(16)] = 0
		data_dummy_prefix_prefixlen1[str(col) + '_prefixlen' + str(17)] = 0
		data_dummy_prefix_prefixlen1[str(col) + '_prefixlen' + str(18)] = 0
		data_dummy_prefix_prefixlen1[str(col) + '_prefixlen' + str(19)] = 0
		data_dummy_prefix_prefixlen1[str(col) + '_prefixlen' + str(20)] = 0
	else:
		data_dummy_prefix_prefixlen1[str(col) + '_prefixlen' + str(1)] = 0 #For case-level attributes, only "_prefixlen1" is attached as suffix.
#Here, stay with 761 or change 761 to other number depending on the situation. This code is to delete columns which do not have names (or having numbers as names).
#Check the column names by yourself and determine the number to replace 761. If 761 is right, stick with 761.
data_dummy_prefix_prefixlen1 = data_dummy_prefix_prefixlen1.iloc[:,761:len(data_dummy_prefix_prefixlen1.columns)]
prefix1 = data_dummy[data_dummy['prefix']==1]
prefix1 = prefix1.reset_index()
prefix1 = prefix1.drop("index",1)
prefix1 = prefix1.join(data_dummy_prefix_prefixlen1)
del data_dummy_prefix_prefixlen1
del data_dummy
for col in prefix1:
    if col[len(col)-10:len(col)] == "prefixlen1":
        prefix1[col] = prefix1[col[0:len(col)-11]]
prefix1 = prefix1.filter(regex="prefixlen")
prefix1.to_pickle('prefix1.pkl')

##########prefix2##########
data_dummy_prefix_prefixlen2 = pd.DataFrame(index=range(len(data_dummy[data_dummy['prefix']==2])), columns=range(len(data_dummy.columns)))
for col in data_dummy_columns_list:
	print(col)
	if col not in ("case_id","Action","EventOrigin","lifecycle.transition","Accepted","Selected","AMOUNT_REQ","case_len","label"):
		data_dummy_prefix_prefixlen2[str(col) + '_prefixlen' + str(1)] = 0
		data_dummy_prefix_prefixlen2[str(col) + '_prefixlen' + str(2)] = 0
		data_dummy_prefix_prefixlen2[str(col) + '_prefixlen' + str(3)] = 0
		data_dummy_prefix_prefixlen2[str(col) + '_prefixlen' + str(4)] = 0
		data_dummy_prefix_prefixlen2[str(col) + '_prefixlen' + str(5)] = 0
		data_dummy_prefix_prefixlen2[str(col) + '_prefixlen' + str(6)] = 0
		data_dummy_prefix_prefixlen2[str(col) + '_prefixlen' + str(7)] = 0
		data_dummy_prefix_prefixlen2[str(col) + '_prefixlen' + str(8)] = 0
		data_dummy_prefix_prefixlen2[str(col) + '_prefixlen' + str(9)] = 0
		data_dummy_prefix_prefixlen2[str(col) + '_prefixlen' + str(10)] = 0
		data_dummy_prefix_prefixlen2[str(col) + '_prefixlen' + str(11)] = 0
		data_dummy_prefix_prefixlen2[str(col) + '_prefixlen' + str(12)] = 0
		data_dummy_prefix_prefixlen2[str(col) + '_prefixlen' + str(13)] = 0
		data_dummy_prefix_prefixlen2[str(col) + '_prefixlen' + str(14)] = 0
		data_dummy_prefix_prefixlen2[str(col) + '_prefixlen' + str(15)] = 0
		data_dummy_prefix_prefixlen2[str(col) + '_prefixlen' + str(16)] = 0
		data_dummy_prefix_prefixlen2[str(col) + '_prefixlen' + str(17)] = 0
		data_dummy_prefix_prefixlen2[str(col) + '_prefixlen' + str(18)] = 0
		data_dummy_prefix_prefixlen2[str(col) + '_prefixlen' + str(19)] = 0
		data_dummy_prefix_prefixlen2[str(col) + '_prefixlen' + str(20)] = 0
	else:
		data_dummy_prefix_prefixlen2[str(col) + '_prefixlen' + str(1)] = 0 #For case-level attributes, only "_prefixlen1" is attached as suffix.
#Here, stay with 761 or change 761 to other number depending on the situation. This code is to delete columns which do not have names (or having numbers as names).
#Check the column names by yourself and determine the number to replace 761. If 761 is right, stick with 761.
print("Started")
data_dummy_prefix_prefixlen2 = data_dummy_prefix_prefixlen2.iloc[:,761:len(data_dummy_prefix_prefixlen2.columns)]
prefix2 = data_dummy[data_dummy['prefix']==2]
prefix2 = prefix2.reset_index()
prefix2 = prefix2.drop("index",1)
prefix2 = prefix2.join(data_dummy_prefix_prefixlen2)
del data_dummy_prefix_prefixlen2
print("data_dummy_prefix_prefixlen2 deleted")
prefix1 = data_dummy[data_dummy['prefix']==1]
del data_dummy
prefix1_ref = prefix1[prefix1['case_len']>=2]
prefix1_ref = prefix1_ref.reset_index()
prefix1_ref = prefix1_ref.drop("index",1)
del prefix1
for col in prefix2:
    if col[len(col)-10:len(col)] == "prefixlen1":
        prefix2[col] = prefix1_ref[col[0:len(col)-11]]
del prefix1_ref
print("prefix1_ref deleted")
for col in prefix2:
    if col[len(col)-10:len(col)] == "prefixlen2":
        prefix2[col] = prefix2[col[0:len(col)-11]]
prefix2 = prefix2.filter(regex="prefixlen")
prefix2.to_pickle('prefix2.pkl')

##########prefix3##########
data_dummy_prefix_prefixlen3 = pd.DataFrame(index=range(len(data_dummy[data_dummy['prefix']==3])), columns=range(len(data_dummy.columns)))
for col in data_dummy_columns_list:
	print(col)
	if col not in ("case_id","Action","EventOrigin","lifecycle.transition","Accepted","Selected","AMOUNT_REQ","case_len","label"):
		data_dummy_prefix_prefixlen3[str(col) + '_prefixlen' + str(1)] = 0
		data_dummy_prefix_prefixlen3[str(col) + '_prefixlen' + str(2)] = 0
		data_dummy_prefix_prefixlen3[str(col) + '_prefixlen' + str(3)] = 0
		data_dummy_prefix_prefixlen3[str(col) + '_prefixlen' + str(4)] = 0
		data_dummy_prefix_prefixlen3[str(col) + '_prefixlen' + str(5)] = 0
		data_dummy_prefix_prefixlen3[str(col) + '_prefixlen' + str(6)] = 0
		data_dummy_prefix_prefixlen3[str(col) + '_prefixlen' + str(7)] = 0
		data_dummy_prefix_prefixlen3[str(col) + '_prefixlen' + str(8)] = 0
		data_dummy_prefix_prefixlen3[str(col) + '_prefixlen' + str(9)] = 0
		data_dummy_prefix_prefixlen3[str(col) + '_prefixlen' + str(10)] = 0
		data_dummy_prefix_prefixlen3[str(col) + '_prefixlen' + str(11)] = 0
		data_dummy_prefix_prefixlen3[str(col) + '_prefixlen' + str(12)] = 0
		data_dummy_prefix_prefixlen3[str(col) + '_prefixlen' + str(13)] = 0
		data_dummy_prefix_prefixlen3[str(col) + '_prefixlen' + str(14)] = 0
		data_dummy_prefix_prefixlen3[str(col) + '_prefixlen' + str(15)] = 0
		data_dummy_prefix_prefixlen3[str(col) + '_prefixlen' + str(16)] = 0
		data_dummy_prefix_prefixlen3[str(col) + '_prefixlen' + str(17)] = 0
		data_dummy_prefix_prefixlen3[str(col) + '_prefixlen' + str(18)] = 0
		data_dummy_prefix_prefixlen3[str(col) + '_prefixlen' + str(19)] = 0
		data_dummy_prefix_prefixlen3[str(col) + '_prefixlen' + str(20)] = 0
	else:
		data_dummy_prefix_prefixlen3[str(col) + '_prefixlen' + str(1)] = 0 #For case-level attributes, only "_prefixlen1" is attached as suffix.
#Here, stay with 761 or change 761 to other number depending on the situation. This code is to delete columns which do not have names (or having numbers as names).
#Check the column names by yourself and determine the number to replace 761. If 761 is right, stick with 761.
data_dummy_prefix_prefixlen3 = data_dummy_prefix_prefixlen3.iloc[:,761:len(data_dummy_prefix_prefixlen3.columns)]
prefix3 = data_dummy[data_dummy['prefix']==3]
prefix3 = prefix3.reset_index()
prefix3 = prefix3.drop("index",1)
prefix3 = prefix3.join(data_dummy_prefix_prefixlen3)
del data_dummy_prefix_prefixlen3

prefix1 = data_dummy[data_dummy['prefix']==1]
prefix1_ref = prefix1[prefix1['case_len']>=3]
prefix1_ref = prefix1_ref.reset_index()
prefix1_ref = prefix1_ref.drop("index",1)
del prefix1
for col in prefix3:
    if col[len(col)-10:len(col)] == "prefixlen1":
        prefix3[col] = prefix1_ref[col[0:len(col)-11]]
del prefix1_ref

prefix2 = data_dummy[data_dummy['prefix']==2]
del data_dummy
prefix2_ref = prefix2[prefix2['case_len']>=3]
prefix2_ref = prefix2_ref.reset_index()
prefix2_ref = prefix2_ref.drop("index",1)
del prefix2
for col in prefix3:
    if col[len(col)-10:len(col)] == "prefixlen2":
        prefix3[col] = prefix2_ref[col[0:len(col)-11]]
del prefix2_ref

for col in prefix3:
    if col[len(col)-10:len(col)] == "prefixlen3":
        prefix3[col] = prefix3[col[0:len(col)-11]]
prefix3 = prefix3.filter(regex="prefixlen")
prefix3.to_pickle('prefix3.pkl')

##########prefix4##########
data_dummy_prefix_prefixlen4 = pd.DataFrame(index=range(len(data_dummy[data_dummy['prefix']==4])), columns=range(len(data_dummy.columns)))
for col in data_dummy_columns_list:
	print(col)
	if col not in ("case_id","Action","EventOrigin","lifecycle.transition","Accepted","Selected","AMOUNT_REQ","case_len","label"):
		data_dummy_prefix_prefixlen4[str(col) + '_prefixlen' + str(1)] = 0
		data_dummy_prefix_prefixlen4[str(col) + '_prefixlen' + str(2)] = 0
		data_dummy_prefix_prefixlen4[str(col) + '_prefixlen' + str(3)] = 0
		data_dummy_prefix_prefixlen4[str(col) + '_prefixlen' + str(4)] = 0
		data_dummy_prefix_prefixlen4[str(col) + '_prefixlen' + str(5)] = 0
		data_dummy_prefix_prefixlen4[str(col) + '_prefixlen' + str(6)] = 0
		data_dummy_prefix_prefixlen4[str(col) + '_prefixlen' + str(7)] = 0
		data_dummy_prefix_prefixlen4[str(col) + '_prefixlen' + str(8)] = 0
		data_dummy_prefix_prefixlen4[str(col) + '_prefixlen' + str(9)] = 0
		data_dummy_prefix_prefixlen4[str(col) + '_prefixlen' + str(10)] = 0
		data_dummy_prefix_prefixlen4[str(col) + '_prefixlen' + str(11)] = 0
		data_dummy_prefix_prefixlen4[str(col) + '_prefixlen' + str(12)] = 0
		data_dummy_prefix_prefixlen4[str(col) + '_prefixlen' + str(13)] = 0
		data_dummy_prefix_prefixlen4[str(col) + '_prefixlen' + str(14)] = 0
		data_dummy_prefix_prefixlen4[str(col) + '_prefixlen' + str(15)] = 0
		data_dummy_prefix_prefixlen4[str(col) + '_prefixlen' + str(16)] = 0
		data_dummy_prefix_prefixlen4[str(col) + '_prefixlen' + str(17)] = 0
		data_dummy_prefix_prefixlen4[str(col) + '_prefixlen' + str(18)] = 0
		data_dummy_prefix_prefixlen4[str(col) + '_prefixlen' + str(19)] = 0
		data_dummy_prefix_prefixlen4[str(col) + '_prefixlen' + str(20)] = 0
	else:
		data_dummy_prefix_prefixlen4[str(col) + '_prefixlen' + str(1)] = 0 #For case-level attributes, only "_prefixlen1" is attached as suffix.
#Here, stay with 761 or change 761 to other number depending on the situation. This code is to delete columns which do not have names (or having numbers as names).
#Check the column names by yourself and determine the number to replace 761. If 761 is right, stick with 761.
data_dummy_prefix_prefixlen4 = data_dummy_prefix_prefixlen4.iloc[:,761:len(data_dummy_prefix_prefixlen4.columns)]
prefix4 = data_dummy[data_dummy['prefix']==4]
prefix4 = prefix4.reset_index()
prefix4 = prefix4.drop("index",1)
prefix4 = prefix4.join(data_dummy_prefix_prefixlen4)
del data_dummy_prefix_prefixlen4

prefix1 = data_dummy[data_dummy['prefix']==1]
prefix1_ref = prefix1[prefix1['case_len']>=4]
prefix1_ref = prefix1_ref.reset_index()
prefix1_ref = prefix1_ref.drop("index",1)
del prefix1
for col in prefix4:
    if col[len(col)-10:len(col)] == "prefixlen1":
        prefix4[col] = prefix1_ref[col[0:len(col)-11]]
del prefix1_ref

prefix2 = data_dummy[data_dummy['prefix']==2]
prefix2_ref = prefix2[prefix2['case_len']>=4]
prefix2_ref = prefix2_ref.reset_index()
prefix2_ref = prefix2_ref.drop("index",1)
del prefix2
for col in prefix4:
    if col[len(col)-10:len(col)] == "prefixlen2":
        prefix4[col] = prefix2_ref[col[0:len(col)-11]]
del prefix2_ref

prefix3 = data_dummy[data_dummy['prefix']==3]
del data_dummy
prefix3_ref = prefix3[prefix3['case_len']>=4]
prefix3_ref = prefix3_ref.reset_index()
prefix3_ref = prefix3_ref.drop("index",1)
del prefix3
for col in prefix4:
    if col[len(col)-10:len(col)] == "prefixlen3":
        prefix4[col] = prefix3_ref[col[0:len(col)-11]]
del prefix3_ref

for col in prefix4:
    if col[len(col)-10:len(col)] == "prefixlen4":
        prefix4[col] = prefix4[col[0:len(col)-11]]
prefix4 = prefix4.filter(regex="prefixlen")
prefix4.to_pickle('prefix4.pkl')

##########prefix5##########
data_dummy_prefix_prefixlen5 = pd.DataFrame(index=range(len(data_dummy[data_dummy['prefix']==5])), columns=range(len(data_dummy.columns)))
for col in data_dummy_columns_list:
	print(col)
	if col not in ("case_id","Action","EventOrigin","lifecycle.transition","Accepted","Selected","AMOUNT_REQ","case_len","label"):
		data_dummy_prefix_prefixlen5[str(col) + '_prefixlen' + str(1)] = 0
		data_dummy_prefix_prefixlen5[str(col) + '_prefixlen' + str(2)] = 0
		data_dummy_prefix_prefixlen5[str(col) + '_prefixlen' + str(3)] = 0
		data_dummy_prefix_prefixlen5[str(col) + '_prefixlen' + str(4)] = 0
		data_dummy_prefix_prefixlen5[str(col) + '_prefixlen' + str(5)] = 0
		data_dummy_prefix_prefixlen5[str(col) + '_prefixlen' + str(6)] = 0
		data_dummy_prefix_prefixlen5[str(col) + '_prefixlen' + str(7)] = 0
		data_dummy_prefix_prefixlen5[str(col) + '_prefixlen' + str(8)] = 0
		data_dummy_prefix_prefixlen5[str(col) + '_prefixlen' + str(9)] = 0
		data_dummy_prefix_prefixlen5[str(col) + '_prefixlen' + str(10)] = 0
		data_dummy_prefix_prefixlen5[str(col) + '_prefixlen' + str(11)] = 0
		data_dummy_prefix_prefixlen5[str(col) + '_prefixlen' + str(12)] = 0
		data_dummy_prefix_prefixlen5[str(col) + '_prefixlen' + str(13)] = 0
		data_dummy_prefix_prefixlen5[str(col) + '_prefixlen' + str(14)] = 0
		data_dummy_prefix_prefixlen5[str(col) + '_prefixlen' + str(15)] = 0
		data_dummy_prefix_prefixlen5[str(col) + '_prefixlen' + str(16)] = 0
		data_dummy_prefix_prefixlen5[str(col) + '_prefixlen' + str(17)] = 0
		data_dummy_prefix_prefixlen5[str(col) + '_prefixlen' + str(18)] = 0
		data_dummy_prefix_prefixlen5[str(col) + '_prefixlen' + str(19)] = 0
		data_dummy_prefix_prefixlen5[str(col) + '_prefixlen' + str(20)] = 0
	else:
		data_dummy_prefix_prefixlen5[str(col) + '_prefixlen' + str(1)] = 0 #For case-level attributes, only "_prefixlen1" is attached as suffix.
#Here, stay with 761 or change 761 to other number depending on the situation. This code is to delete columns which do not have names (or having numbers as names).
#Check the column names by yourself and determine the number to replace 761. If 761 is right, stick with 761.
data_dummy_prefix_prefixlen5 = data_dummy_prefix_prefixlen5.iloc[:,761:len(data_dummy_prefix_prefixlen5.columns)]
prefix5 = data_dummy[data_dummy['prefix']==5]
prefix5 = prefix5.reset_index()
prefix5 = prefix5.drop("index",1)
prefix5 = prefix5.join(data_dummy_prefix_prefixlen5)
del data_dummy_prefix_prefixlen5

prefix1 = data_dummy[data_dummy['prefix']==1]
prefix1_ref = prefix1[prefix1['case_len']>=5]
prefix1_ref = prefix1_ref.reset_index()
prefix1_ref = prefix1_ref.drop("index",1)
del prefix1
for col in prefix5:
    if col[len(col)-10:len(col)] == "prefixlen1":
        prefix5[col] = prefix1_ref[col[0:len(col)-11]]
del prefix1_ref

prefix2 = data_dummy[data_dummy['prefix']==2]
prefix2_ref = prefix2[prefix2['case_len']>=5]
prefix2_ref = prefix2_ref.reset_index()
prefix2_ref = prefix2_ref.drop("index",1)
del prefix2
for col in prefix5:
    if col[len(col)-10:len(col)] == "prefixlen2":
        prefix5[col] = prefix2_ref[col[0:len(col)-11]]
del prefix2_ref

prefix3 = data_dummy[data_dummy['prefix']==3]
prefix3_ref = prefix3[prefix3['case_len']>=5]
prefix3_ref = prefix3_ref.reset_index()
prefix3_ref = prefix3_ref.drop("index",1)
del prefix3
for col in prefix5:
    if col[len(col)-10:len(col)] == "prefixlen3":
        prefix5[col] = prefix3_ref[col[0:len(col)-11]]
del prefix3_ref

prefix4 = data_dummy[data_dummy['prefix']==4]
del data_dummy
prefix4_ref = prefix4[prefix4['case_len']>=5]
prefix4_ref = prefix4_ref.reset_index()
prefix4_ref = prefix4_ref.drop("index",1)
del prefix4
for col in prefix5:
    if col[len(col)-10:len(col)] == "prefixlen4":
        prefix5[col] = prefix4_ref[col[0:len(col)-11]]
del prefix4_ref

for col in prefix5:
    if col[len(col)-10:len(col)] == "prefixlen5":
        prefix5[col] = prefix5[col[0:len(col)-11]]
prefix5 = prefix5.filter(regex="prefixlen")
prefix5.to_pickle('prefix5.pkl')

##########prefix6##########
data_dummy_prefix_prefixlen6 = pd.DataFrame(index=range(len(data_dummy[data_dummy['prefix']==6])), columns=range(len(data_dummy.columns)))
for col in data_dummy_columns_list:
	print(col)
	if col not in ("case_id","Action","EventOrigin","lifecycle.transition","Accepted","Selected","AMOUNT_REQ","case_len","label"):
		data_dummy_prefix_prefixlen6[str(col) + '_prefixlen' + str(1)] = 0
		data_dummy_prefix_prefixlen6[str(col) + '_prefixlen' + str(2)] = 0
		data_dummy_prefix_prefixlen6[str(col) + '_prefixlen' + str(3)] = 0
		data_dummy_prefix_prefixlen6[str(col) + '_prefixlen' + str(4)] = 0
		data_dummy_prefix_prefixlen6[str(col) + '_prefixlen' + str(5)] = 0
		data_dummy_prefix_prefixlen6[str(col) + '_prefixlen' + str(6)] = 0
		data_dummy_prefix_prefixlen6[str(col) + '_prefixlen' + str(7)] = 0
		data_dummy_prefix_prefixlen6[str(col) + '_prefixlen' + str(8)] = 0
		data_dummy_prefix_prefixlen6[str(col) + '_prefixlen' + str(9)] = 0
		data_dummy_prefix_prefixlen6[str(col) + '_prefixlen' + str(10)] = 0
		data_dummy_prefix_prefixlen6[str(col) + '_prefixlen' + str(11)] = 0
		data_dummy_prefix_prefixlen6[str(col) + '_prefixlen' + str(12)] = 0
		data_dummy_prefix_prefixlen6[str(col) + '_prefixlen' + str(13)] = 0
		data_dummy_prefix_prefixlen6[str(col) + '_prefixlen' + str(14)] = 0
		data_dummy_prefix_prefixlen6[str(col) + '_prefixlen' + str(15)] = 0
		data_dummy_prefix_prefixlen6[str(col) + '_prefixlen' + str(16)] = 0
		data_dummy_prefix_prefixlen6[str(col) + '_prefixlen' + str(17)] = 0
		data_dummy_prefix_prefixlen6[str(col) + '_prefixlen' + str(18)] = 0
		data_dummy_prefix_prefixlen6[str(col) + '_prefixlen' + str(19)] = 0
		data_dummy_prefix_prefixlen6[str(col) + '_prefixlen' + str(20)] = 0
	else:
		data_dummy_prefix_prefixlen6[str(col) + '_prefixlen' + str(1)] = 0 #For case-level attributes, only "_prefixlen1" is attached as suffix.
#Here, stay with 761 or change 761 to other number depending on the situation. This code is to delete columns which do not have names (or having numbers as names).
#Check the column names by yourself and determine the number to replace 761. If 761 is right, stick with 761.
data_dummy_prefix_prefixlen6 = data_dummy_prefix_prefixlen6.iloc[:,761:len(data_dummy_prefix_prefixlen6.columns)]
prefix6 = data_dummy[data_dummy['prefix']==6]
prefix6 = prefix6.reset_index()
prefix6 = prefix6.drop("index",1)
prefix6 = prefix6.join(data_dummy_prefix_prefixlen6)
del data_dummy_prefix_prefixlen6

prefix1 = data_dummy[data_dummy['prefix']==1]
prefix1_ref = prefix1[prefix1['case_len']>=6]
prefix1_ref = prefix1_ref.reset_index()
prefix1_ref = prefix1_ref.drop("index",1)
del prefix1
for col in prefix6:
    if col[len(col)-10:len(col)] == "prefixlen1":
        prefix6[col] = prefix1_ref[col[0:len(col)-11]]
del prefix1_ref

prefix2 = data_dummy[data_dummy['prefix']==2]
prefix2_ref = prefix2[prefix2['case_len']>=6]
prefix2_ref = prefix2_ref.reset_index()
prefix2_ref = prefix2_ref.drop("index",1)
del prefix2
for col in prefix6:
    if col[len(col)-10:len(col)] == "prefixlen2":
        prefix6[col] = prefix2_ref[col[0:len(col)-11]]
del prefix2_ref

prefix3 = data_dummy[data_dummy['prefix']==3]
prefix3_ref = prefix3[prefix3['case_len']>=6]
prefix3_ref = prefix3_ref.reset_index()
prefix3_ref = prefix3_ref.drop("index",1)
del prefix3
for col in prefix6:
    if col[len(col)-10:len(col)] == "prefixlen3":
        prefix6[col] = prefix3_ref[col[0:len(col)-11]]
del prefix3_ref

prefix4 = data_dummy[data_dummy['prefix']==4]
prefix4_ref = prefix4[prefix4['case_len']>=6]
prefix4_ref = prefix4_ref.reset_index()
prefix4_ref = prefix4_ref.drop("index",1)
del prefix4
for col in prefix6:
    if col[len(col)-10:len(col)] == "prefixlen4":
        prefix6[col] = prefix4_ref[col[0:len(col)-11]]
del prefix4_ref

prefix5 = data_dummy[data_dummy['prefix']==5]
del data_dummy
prefix5_ref = prefix5[prefix5['case_len']>=6]
prefix5_ref = prefix5_ref.reset_index()
prefix5_ref = prefix5_ref.drop("index",1)
del prefix5
for col in prefix6:
    if col[len(col)-10:len(col)] == "prefixlen5":
        prefix6[col] = prefix5_ref[col[0:len(col)-11]]
del prefix5_ref

for col in prefix6:
    if col[len(col)-10:len(col)] == "prefixlen6":
        prefix6[col] = prefix6[col[0:len(col)-11]]
prefix6 = prefix6.filter(regex="prefixlen")
prefix6.to_pickle('prefix6.pkl')

##########prefix7##########
data_dummy_prefix_prefixlen7 = pd.DataFrame(index=range(len(data_dummy[data_dummy['prefix']==7])), columns=range(len(data_dummy.columns)))
for col in data_dummy_columns_list:
	print(col)
	if col not in ("case_id","Action","EventOrigin","lifecycle.transition","Accepted","Selected","AMOUNT_REQ","case_len","label"):
		data_dummy_prefix_prefixlen7[str(col) + '_prefixlen' + str(1)] = 0
		data_dummy_prefix_prefixlen7[str(col) + '_prefixlen' + str(2)] = 0
		data_dummy_prefix_prefixlen7[str(col) + '_prefixlen' + str(3)] = 0
		data_dummy_prefix_prefixlen7[str(col) + '_prefixlen' + str(4)] = 0
		data_dummy_prefix_prefixlen7[str(col) + '_prefixlen' + str(5)] = 0
		data_dummy_prefix_prefixlen7[str(col) + '_prefixlen' + str(6)] = 0
		data_dummy_prefix_prefixlen7[str(col) + '_prefixlen' + str(7)] = 0
		data_dummy_prefix_prefixlen7[str(col) + '_prefixlen' + str(8)] = 0
		data_dummy_prefix_prefixlen7[str(col) + '_prefixlen' + str(9)] = 0
		data_dummy_prefix_prefixlen7[str(col) + '_prefixlen' + str(10)] = 0
		data_dummy_prefix_prefixlen7[str(col) + '_prefixlen' + str(11)] = 0
		data_dummy_prefix_prefixlen7[str(col) + '_prefixlen' + str(12)] = 0
		data_dummy_prefix_prefixlen7[str(col) + '_prefixlen' + str(13)] = 0
		data_dummy_prefix_prefixlen7[str(col) + '_prefixlen' + str(14)] = 0
		data_dummy_prefix_prefixlen7[str(col) + '_prefixlen' + str(15)] = 0
		data_dummy_prefix_prefixlen7[str(col) + '_prefixlen' + str(16)] = 0
		data_dummy_prefix_prefixlen7[str(col) + '_prefixlen' + str(17)] = 0
		data_dummy_prefix_prefixlen7[str(col) + '_prefixlen' + str(18)] = 0
		data_dummy_prefix_prefixlen7[str(col) + '_prefixlen' + str(19)] = 0
		data_dummy_prefix_prefixlen7[str(col) + '_prefixlen' + str(20)] = 0
	else:
		data_dummy_prefix_prefixlen7[str(col) + '_prefixlen' + str(1)] = 0 #For case-level attributes, only "_prefixlen1" is attached as suffix.
#Here, stay with 761 or change 761 to other number depending on the situation. This code is to delete columns which do not have names (or having numbers as names).
#Check the column names by yourself and determine the number to replace 761. If 761 is right, stick with 761.
data_dummy_prefix_prefixlen7 = data_dummy_prefix_prefixlen7.iloc[:,761:len(data_dummy_prefix_prefixlen7.columns)]
prefix7 = data_dummy[data_dummy['prefix']==7]
prefix7 = prefix7.reset_index()
prefix7 = prefix7.drop("index",1)
prefix7 = prefix7.join(data_dummy_prefix_prefixlen7)
del data_dummy_prefix_prefixlen7

prefix1 = data_dummy[data_dummy['prefix']==1]
prefix1_ref = prefix1[prefix1['case_len']>=7]
prefix1_ref = prefix1_ref.reset_index()
prefix1_ref = prefix1_ref.drop("index",1)
del prefix1
for col in prefix7:
    if col[len(col)-10:len(col)] == "prefixlen1":
        prefix7[col] = prefix1_ref[col[0:len(col)-11]]
del prefix1_ref

prefix2 = data_dummy[data_dummy['prefix']==2]
prefix2_ref = prefix2[prefix2['case_len']>=7]
prefix2_ref = prefix2_ref.reset_index()
prefix2_ref = prefix2_ref.drop("index",1)
del prefix2
for col in prefix7:
    if col[len(col)-10:len(col)] == "prefixlen2":
        prefix7[col] = prefix2_ref[col[0:len(col)-11]]
del prefix2_ref

prefix3 = data_dummy[data_dummy['prefix']==3]
prefix3_ref = prefix3[prefix3['case_len']>=7]
prefix3_ref = prefix3_ref.reset_index()
prefix3_ref = prefix3_ref.drop("index",1)
del prefix3
for col in prefix7:
    if col[len(col)-10:len(col)] == "prefixlen3":
        prefix7[col] = prefix3_ref[col[0:len(col)-11]]
del prefix3_ref

prefix4 = data_dummy[data_dummy['prefix']==4]
prefix4_ref = prefix4[prefix4['case_len']>=7]
prefix4_ref = prefix4_ref.reset_index()
prefix4_ref = prefix4_ref.drop("index",1)
del prefix4
for col in prefix7:
    if col[len(col)-10:len(col)] == "prefixlen4":
        prefix7[col] = prefix4_ref[col[0:len(col)-11]]
del prefix4_ref

prefix5 = data_dummy[data_dummy['prefix']==5]
prefix5_ref = prefix5[prefix5['case_len']>=7]
prefix5_ref = prefix5_ref.reset_index()
prefix5_ref = prefix5_ref.drop("index",1)
del prefix5
for col in prefix7:
    if col[len(col)-10:len(col)] == "prefixlen5":
        prefix7[col] = prefix5_ref[col[0:len(col)-11]]
del prefix5_ref

prefix6 = data_dummy[data_dummy['prefix']==6]
del data_dummy
prefix6_ref = prefix6[prefix6['case_len']>=7]
prefix6_ref = prefix6_ref.reset_index()
prefix6_ref = prefix6_ref.drop("index",1)
del prefix6
for col in prefix7:
    if col[len(col)-10:len(col)] == "prefixlen6":
        prefix7[col] = prefix6_ref[col[0:len(col)-11]]
del prefix6_ref

for col in prefix7:
    if col[len(col)-10:len(col)] == "prefixlen7":
        prefix7[col] = prefix7[col[0:len(col)-11]]
prefix7 = prefix7.filter(regex="prefixlen")
prefix7.to_pickle('prefix7.pkl')

##########prefix8##########
data_dummy_prefix_prefixlen8 = pd.DataFrame(index=range(len(data_dummy[data_dummy['prefix']==8])), columns=range(len(data_dummy.columns)))
for col in data_dummy_columns_list:
	print(col)
	if col not in ("case_id","Action","EventOrigin","lifecycle.transition","Accepted","Selected","AMOUNT_REQ","case_len","label"):
		data_dummy_prefix_prefixlen8[str(col) + '_prefixlen' + str(1)] = 0
		data_dummy_prefix_prefixlen8[str(col) + '_prefixlen' + str(2)] = 0
		data_dummy_prefix_prefixlen8[str(col) + '_prefixlen' + str(3)] = 0
		data_dummy_prefix_prefixlen8[str(col) + '_prefixlen' + str(4)] = 0
		data_dummy_prefix_prefixlen8[str(col) + '_prefixlen' + str(5)] = 0
		data_dummy_prefix_prefixlen8[str(col) + '_prefixlen' + str(6)] = 0
		data_dummy_prefix_prefixlen8[str(col) + '_prefixlen' + str(7)] = 0
		data_dummy_prefix_prefixlen8[str(col) + '_prefixlen' + str(8)] = 0
		data_dummy_prefix_prefixlen8[str(col) + '_prefixlen' + str(9)] = 0
		data_dummy_prefix_prefixlen8[str(col) + '_prefixlen' + str(10)] = 0
		data_dummy_prefix_prefixlen8[str(col) + '_prefixlen' + str(11)] = 0
		data_dummy_prefix_prefixlen8[str(col) + '_prefixlen' + str(12)] = 0
		data_dummy_prefix_prefixlen8[str(col) + '_prefixlen' + str(13)] = 0
		data_dummy_prefix_prefixlen8[str(col) + '_prefixlen' + str(14)] = 0
		data_dummy_prefix_prefixlen8[str(col) + '_prefixlen' + str(15)] = 0
		data_dummy_prefix_prefixlen8[str(col) + '_prefixlen' + str(16)] = 0
		data_dummy_prefix_prefixlen8[str(col) + '_prefixlen' + str(17)] = 0
		data_dummy_prefix_prefixlen8[str(col) + '_prefixlen' + str(18)] = 0
		data_dummy_prefix_prefixlen8[str(col) + '_prefixlen' + str(19)] = 0
		data_dummy_prefix_prefixlen8[str(col) + '_prefixlen' + str(20)] = 0
	else:
		data_dummy_prefix_prefixlen8[str(col) + '_prefixlen' + str(1)] = 0 #For case-level attributes, only "_prefixlen1" is attached as suffix.
#Here, stay with 761 or change 761 to other number depending on the situation. This code is to delete columns which do not have names (or having numbers as names).
#Check the column names by yourself and determine the number to replace 761. If 761 is right, stick with 761.
data_dummy_prefix_prefixlen8 = data_dummy_prefix_prefixlen8.iloc[:,761:len(data_dummy_prefix_prefixlen8.columns)]
prefix8 = data_dummy[data_dummy['prefix']==8]
prefix8 = prefix8.reset_index()
prefix8 = prefix8.drop("index",1)
prefix8 = prefix8.join(data_dummy_prefix_prefixlen8)
del data_dummy_prefix_prefixlen8

prefix1 = data_dummy[data_dummy['prefix']==1]
prefix1_ref = prefix1[prefix1['case_len']>=8]
prefix1_ref = prefix1_ref.reset_index()
prefix1_ref = prefix1_ref.drop("index",1)
del prefix1
for col in prefix8:
    if col[len(col)-10:len(col)] == "prefixlen1":
        prefix8[col] = prefix1_ref[col[0:len(col)-11]]
del prefix1_ref

prefix2 = data_dummy[data_dummy['prefix']==2]
prefix2_ref = prefix2[prefix2['case_len']>=8]
prefix2_ref = prefix2_ref.reset_index()
prefix2_ref = prefix2_ref.drop("index",1)
del prefix2
for col in prefix8:
    if col[len(col)-10:len(col)] == "prefixlen2":
        prefix8[col] = prefix2_ref[col[0:len(col)-11]]
del prefix2_ref

prefix3 = data_dummy[data_dummy['prefix']==3]
prefix3_ref = prefix3[prefix3['case_len']>=8]
prefix3_ref = prefix3_ref.reset_index()
prefix3_ref = prefix3_ref.drop("index",1)
del prefix3
for col in prefix8:
    if col[len(col)-10:len(col)] == "prefixlen3":
        prefix8[col] = prefix3_ref[col[0:len(col)-11]]
del prefix3_ref

prefix4 = data_dummy[data_dummy['prefix']==4]
prefix4_ref = prefix4[prefix4['case_len']>=8]
prefix4_ref = prefix4_ref.reset_index()
prefix4_ref = prefix4_ref.drop("index",1)
del prefix4
for col in prefix8:
    if col[len(col)-10:len(col)] == "prefixlen4":
        prefix8[col] = prefix4_ref[col[0:len(col)-11]]
del prefix4_ref

prefix5 = data_dummy[data_dummy['prefix']==5]
prefix5_ref = prefix5[prefix5['case_len']>=8]
prefix5_ref = prefix5_ref.reset_index()
prefix5_ref = prefix5_ref.drop("index",1)
del prefix5
for col in prefix8:
    if col[len(col)-10:len(col)] == "prefixlen5":
        prefix8[col] = prefix5_ref[col[0:len(col)-11]]
del prefix5_ref

prefix6 = data_dummy[data_dummy['prefix']==6]
prefix6_ref = prefix6[prefix6['case_len']>=8]
prefix6_ref = prefix6_ref.reset_index()
prefix6_ref = prefix6_ref.drop("index",1)
del prefix6
for col in prefix8:
    if col[len(col)-10:len(col)] == "prefixlen6":
        prefix8[col] = prefix6_ref[col[0:len(col)-11]]
del prefix6_ref

prefix7 = data_dummy[data_dummy['prefix']==7]
del data_dummy
prefix7_ref = prefix7[prefix7['case_len']>=8]
prefix7_ref = prefix7_ref.reset_index()
prefix7_ref = prefix7_ref.drop("index",1)
del prefix7
for col in prefix8:
    if col[len(col)-10:len(col)] == "prefixlen7":
        prefix8[col] = prefix7_ref[col[0:len(col)-11]]
del prefix7_ref

for col in prefix8:
    if col[len(col)-10:len(col)] == "prefixlen8":
        prefix8[col] = prefix8[col[0:len(col)-11]]
prefix8 = prefix8.filter(regex="prefixlen")
prefix8.to_pickle('prefix8.pkl')

##########prefix9##########
data_dummy_prefix_prefixlen9 = pd.DataFrame(index=range(len(data_dummy[data_dummy['prefix']==9])), columns=range(len(data_dummy.columns)))
for col in data_dummy_columns_list:
	print(col)
	if col not in ("case_id","Action","EventOrigin","lifecycle.transition","Accepted","Selected","AMOUNT_REQ","case_len","label"):
		data_dummy_prefix_prefixlen9[str(col) + '_prefixlen' + str(1)] = 0
		data_dummy_prefix_prefixlen9[str(col) + '_prefixlen' + str(2)] = 0
		data_dummy_prefix_prefixlen9[str(col) + '_prefixlen' + str(3)] = 0
		data_dummy_prefix_prefixlen9[str(col) + '_prefixlen' + str(4)] = 0
		data_dummy_prefix_prefixlen9[str(col) + '_prefixlen' + str(5)] = 0
		data_dummy_prefix_prefixlen9[str(col) + '_prefixlen' + str(6)] = 0
		data_dummy_prefix_prefixlen9[str(col) + '_prefixlen' + str(7)] = 0
		data_dummy_prefix_prefixlen9[str(col) + '_prefixlen' + str(8)] = 0
		data_dummy_prefix_prefixlen9[str(col) + '_prefixlen' + str(9)] = 0
		data_dummy_prefix_prefixlen9[str(col) + '_prefixlen' + str(10)] = 0
		data_dummy_prefix_prefixlen9[str(col) + '_prefixlen' + str(11)] = 0
		data_dummy_prefix_prefixlen9[str(col) + '_prefixlen' + str(12)] = 0
		data_dummy_prefix_prefixlen9[str(col) + '_prefixlen' + str(13)] = 0
		data_dummy_prefix_prefixlen9[str(col) + '_prefixlen' + str(14)] = 0
		data_dummy_prefix_prefixlen9[str(col) + '_prefixlen' + str(15)] = 0
		data_dummy_prefix_prefixlen9[str(col) + '_prefixlen' + str(16)] = 0
		data_dummy_prefix_prefixlen9[str(col) + '_prefixlen' + str(17)] = 0
		data_dummy_prefix_prefixlen9[str(col) + '_prefixlen' + str(18)] = 0
		data_dummy_prefix_prefixlen9[str(col) + '_prefixlen' + str(19)] = 0
		data_dummy_prefix_prefixlen9[str(col) + '_prefixlen' + str(20)] = 0
	else:
		data_dummy_prefix_prefixlen9[str(col) + '_prefixlen' + str(1)] = 0 #For case-level attributes, only "_prefixlen1" is attached as suffix.
#Here, stay with 761 or change 761 to other number depending on the situation. This code is to delete columns which do not have names (or having numbers as names).
#Check the column names by yourself and determine the number to replace 761. If 761 is right, stick with 761.
data_dummy_prefix_prefixlen9 = data_dummy_prefix_prefixlen9.iloc[:,761:len(data_dummy_prefix_prefixlen9.columns)]
prefix9 = data_dummy[data_dummy['prefix']==9]
prefix9 = prefix9.reset_index()
prefix9 = prefix9.drop("index",1)
prefix9 = prefix9.join(data_dummy_prefix_prefixlen9)
del data_dummy_prefix_prefixlen9

prefix1 = data_dummy[data_dummy['prefix']==1]
prefix1_ref = prefix1[prefix1['case_len']>=9]
prefix1_ref = prefix1_ref.reset_index()
prefix1_ref = prefix1_ref.drop("index",1)
del prefix1
for col in prefix9:
    if col[len(col)-10:len(col)] == "prefixlen1":
        prefix9[col] = prefix1_ref[col[0:len(col)-11]]
del prefix1_ref

prefix2 = data_dummy[data_dummy['prefix']==2]
prefix2_ref = prefix2[prefix2['case_len']>=9]
prefix2_ref = prefix2_ref.reset_index()
prefix2_ref = prefix2_ref.drop("index",1)
del prefix2
for col in prefix9:
    if col[len(col)-10:len(col)] == "prefixlen2":
        prefix9[col] = prefix2_ref[col[0:len(col)-11]]
del prefix2_ref

prefix3 = data_dummy[data_dummy['prefix']==3]
prefix3_ref = prefix3[prefix3['case_len']>=9]
prefix3_ref = prefix3_ref.reset_index()
prefix3_ref = prefix3_ref.drop("index",1)
del prefix3
for col in prefix9:
    if col[len(col)-10:len(col)] == "prefixlen3":
        prefix9[col] = prefix3_ref[col[0:len(col)-11]]
del prefix3_ref

prefix4 = data_dummy[data_dummy['prefix']==4]
prefix4_ref = prefix4[prefix4['case_len']>=9]
prefix4_ref = prefix4_ref.reset_index()
prefix4_ref = prefix4_ref.drop("index",1)
del prefix4
for col in prefix9:
    if col[len(col)-10:len(col)] == "prefixlen4":
        prefix9[col] = prefix4_ref[col[0:len(col)-11]]
del prefix4_ref

prefix5 = data_dummy[data_dummy['prefix']==5]
prefix5_ref = prefix5[prefix5['case_len']>=9]
prefix5_ref = prefix5_ref.reset_index()
prefix5_ref = prefix5_ref.drop("index",1)
del prefix5
for col in prefix9:
    if col[len(col)-10:len(col)] == "prefixlen5":
        prefix9[col] = prefix5_ref[col[0:len(col)-11]]
del prefix5_ref

prefix6 = data_dummy[data_dummy['prefix']==6]
prefix6_ref = prefix6[prefix6['case_len']>=9]
prefix6_ref = prefix6_ref.reset_index()
prefix6_ref = prefix6_ref.drop("index",1)
del prefix6
for col in prefix9:
    if col[len(col)-10:len(col)] == "prefixlen6":
        prefix9[col] = prefix6_ref[col[0:len(col)-11]]
del prefix6_ref

prefix7 = data_dummy[data_dummy['prefix']==7]
prefix7_ref = prefix7[prefix7['case_len']>=9]
prefix7_ref = prefix7_ref.reset_index()
prefix7_ref = prefix7_ref.drop("index",1)
del prefix7
for col in prefix9:
    if col[len(col)-10:len(col)] == "prefixlen7":
        prefix9[col] = prefix7_ref[col[0:len(col)-11]]
del prefix7_ref

prefix8 = data_dummy[data_dummy['prefix']==8]
del data_dummy
prefix8_ref = prefix8[prefix8['case_len']>=9]
prefix8_ref = prefix8_ref.reset_index()
prefix8_ref = prefix8_ref.drop("index",1)
del prefix8
for col in prefix9:
    if col[len(col)-10:len(col)] == "prefixlen8":
        prefix9[col] = prefix8_ref[col[0:len(col)-11]]
del prefix8_ref

for col in prefix9:
    if col[len(col)-10:len(col)] == "prefixlen9":
        prefix9[col] = prefix9[col[0:len(col)-11]]
prefix9 = prefix9.filter(regex="prefixlen")
prefix9.to_pickle('prefix9.pkl')

##########prefix10##########
data_dummy_prefix_prefixlen10 = pd.DataFrame(index=range(len(data_dummy[data_dummy['prefix']==10])), columns=range(len(data_dummy.columns)))
for col in data_dummy_columns_list:
	print(col)
	if col not in ("case_id","Action","EventOrigin","lifecycle.transition","Accepted","Selected","AMOUNT_REQ","case_len","label"):
		data_dummy_prefix_prefixlen10[str(col) + '_prefixlen' + str(1)] = 0
		data_dummy_prefix_prefixlen10[str(col) + '_prefixlen' + str(2)] = 0
		data_dummy_prefix_prefixlen10[str(col) + '_prefixlen' + str(3)] = 0
		data_dummy_prefix_prefixlen10[str(col) + '_prefixlen' + str(4)] = 0
		data_dummy_prefix_prefixlen10[str(col) + '_prefixlen' + str(5)] = 0
		data_dummy_prefix_prefixlen10[str(col) + '_prefixlen' + str(6)] = 0
		data_dummy_prefix_prefixlen10[str(col) + '_prefixlen' + str(7)] = 0
		data_dummy_prefix_prefixlen10[str(col) + '_prefixlen' + str(8)] = 0
		data_dummy_prefix_prefixlen10[str(col) + '_prefixlen' + str(9)] = 0
		data_dummy_prefix_prefixlen10[str(col) + '_prefixlen' + str(10)] = 0
		data_dummy_prefix_prefixlen10[str(col) + '_prefixlen' + str(11)] = 0
		data_dummy_prefix_prefixlen10[str(col) + '_prefixlen' + str(12)] = 0
		data_dummy_prefix_prefixlen10[str(col) + '_prefixlen' + str(13)] = 0
		data_dummy_prefix_prefixlen10[str(col) + '_prefixlen' + str(14)] = 0
		data_dummy_prefix_prefixlen10[str(col) + '_prefixlen' + str(15)] = 0
		data_dummy_prefix_prefixlen10[str(col) + '_prefixlen' + str(16)] = 0
		data_dummy_prefix_prefixlen10[str(col) + '_prefixlen' + str(17)] = 0
		data_dummy_prefix_prefixlen10[str(col) + '_prefixlen' + str(18)] = 0
		data_dummy_prefix_prefixlen10[str(col) + '_prefixlen' + str(19)] = 0
		data_dummy_prefix_prefixlen10[str(col) + '_prefixlen' + str(20)] = 0
	else:
		data_dummy_prefix_prefixlen10[str(col) + '_prefixlen' + str(1)] = 0 #For case-level attributes, only "_prefixlen1" is attached as suffix.
#Here, stay with 761 or change 761 to other number depending on the situation. This code is to delete columns which do not have names (or having numbers as names).
#Check the column names by yourself and determine the number to replace 761. If 761 is right, stick with 761.
data_dummy_prefix_prefixlen10 = data_dummy_prefix_prefixlen10.iloc[:,761:len(data_dummy_prefix_prefixlen10.columns)]
prefix10 = data_dummy[data_dummy['prefix']==10]
prefix10 = prefix10.reset_index()
prefix10 = prefix10.drop("index",1)
prefix10 = prefix10.join(data_dummy_prefix_prefixlen10)
del data_dummy_prefix_prefixlen10

prefix1 = data_dummy[data_dummy['prefix']==1]
prefix1_ref = prefix1[prefix1['case_len']>=10]
prefix1_ref = prefix1_ref.reset_index()
prefix1_ref = prefix1_ref.drop("index",1)
del prefix1
for col in prefix10:
    if col[len(col)-10:len(col)] == "prefixlen1":
        prefix10[col] = prefix1_ref[col[0:len(col)-11]]
del prefix1_ref

prefix2 = data_dummy[data_dummy['prefix']==2]
prefix2_ref = prefix2[prefix2['case_len']>=10]
prefix2_ref = prefix2_ref.reset_index()
prefix2_ref = prefix2_ref.drop("index",1)
del prefix2
for col in prefix10:
    if col[len(col)-10:len(col)] == "prefixlen2":
        prefix10[col] = prefix2_ref[col[0:len(col)-11]]
del prefix2_ref

prefix3 = data_dummy[data_dummy['prefix']==3]
prefix3_ref = prefix3[prefix3['case_len']>=10]
prefix3_ref = prefix3_ref.reset_index()
prefix3_ref = prefix3_ref.drop("index",1)
del prefix3
for col in prefix10:
    if col[len(col)-10:len(col)] == "prefixlen3":
        prefix10[col] = prefix3_ref[col[0:len(col)-11]]
del prefix3_ref

prefix4 = data_dummy[data_dummy['prefix']==4]
prefix4_ref = prefix4[prefix4['case_len']>=10]
prefix4_ref = prefix4_ref.reset_index()
prefix4_ref = prefix4_ref.drop("index",1)
del prefix4
for col in prefix10:
    if col[len(col)-10:len(col)] == "prefixlen4":
        prefix10[col] = prefix4_ref[col[0:len(col)-11]]
del prefix4_ref

prefix5 = data_dummy[data_dummy['prefix']==5]
prefix5_ref = prefix5[prefix5['case_len']>=10]
prefix5_ref = prefix5_ref.reset_index()
prefix5_ref = prefix5_ref.drop("index",1)
del prefix5
for col in prefix10:
    if col[len(col)-10:len(col)] == "prefixlen5":
        prefix10[col] = prefix5_ref[col[0:len(col)-11]]
del prefix5_ref

prefix6 = data_dummy[data_dummy['prefix']==6]
prefix6_ref = prefix6[prefix6['case_len']>=10]
prefix6_ref = prefix6_ref.reset_index()
prefix6_ref = prefix6_ref.drop("index",1)
del prefix6
for col in prefix10:
    if col[len(col)-10:len(col)] == "prefixlen6":
        prefix10[col] = prefix6_ref[col[0:len(col)-11]]
del prefix6_ref

prefix7 = data_dummy[data_dummy['prefix']==7]
prefix7_ref = prefix7[prefix7['case_len']>=10]
prefix7_ref = prefix7_ref.reset_index()
prefix7_ref = prefix7_ref.drop("index",1)
del prefix7
for col in prefix10:
    if col[len(col)-10:len(col)] == "prefixlen7":
        prefix10[col] = prefix7_ref[col[0:len(col)-11]]
del prefix7_ref

prefix8 = data_dummy[data_dummy['prefix']==8]
prefix8_ref = prefix8[prefix8['case_len']>=10]
prefix8_ref = prefix8_ref.reset_index()
prefix8_ref = prefix8_ref.drop("index",1)
del prefix8
for col in prefix10:
    if col[len(col)-10:len(col)] == "prefixlen8":
        prefix10[col] = prefix8_ref[col[0:len(col)-11]]
del prefix8_ref

prefix9 = data_dummy[data_dummy['prefix']==9]
del data_dummy
prefix9_ref = prefix9[prefix9['case_len']>=10]
prefix9_ref = prefix9_ref.reset_index()
prefix9_ref = prefix9_ref.drop("index",1)
del prefix9
for col in prefix10:
    if col[len(col)-10:len(col)] == "prefixlen9":
        prefix10[col] = prefix9_ref[col[0:len(col)-11]]
del prefix9_ref

for col in prefix10:
    if col[len(col)-11:len(col)] == "prefixlen10":
        prefix10[col] = prefix10[col[0:len(col)-12]]
prefix10 = prefix10.filter(regex="prefixlen")
prefix10.to_pickle('prefix10.pkl')

##########prefix11##########
data_dummy_prefix_prefixlen11 = pd.DataFrame(index=range(len(data_dummy[data_dummy['prefix']==11])), columns=range(len(data_dummy.columns)))
for col in data_dummy_columns_list:
	print(col)
	if col not in ("case_id","Action","EventOrigin","lifecycle.transition","Accepted","Selected","AMOUNT_REQ","case_len","label"):
		data_dummy_prefix_prefixlen11[str(col) + '_prefixlen' + str(1)] = 0
		data_dummy_prefix_prefixlen11[str(col) + '_prefixlen' + str(2)] = 0
		data_dummy_prefix_prefixlen11[str(col) + '_prefixlen' + str(3)] = 0
		data_dummy_prefix_prefixlen11[str(col) + '_prefixlen' + str(4)] = 0
		data_dummy_prefix_prefixlen11[str(col) + '_prefixlen' + str(5)] = 0
		data_dummy_prefix_prefixlen11[str(col) + '_prefixlen' + str(6)] = 0
		data_dummy_prefix_prefixlen11[str(col) + '_prefixlen' + str(7)] = 0
		data_dummy_prefix_prefixlen11[str(col) + '_prefixlen' + str(8)] = 0
		data_dummy_prefix_prefixlen11[str(col) + '_prefixlen' + str(9)] = 0
		data_dummy_prefix_prefixlen11[str(col) + '_prefixlen' + str(10)] = 0
		data_dummy_prefix_prefixlen11[str(col) + '_prefixlen' + str(11)] = 0
		data_dummy_prefix_prefixlen11[str(col) + '_prefixlen' + str(12)] = 0
		data_dummy_prefix_prefixlen11[str(col) + '_prefixlen' + str(13)] = 0
		data_dummy_prefix_prefixlen11[str(col) + '_prefixlen' + str(14)] = 0
		data_dummy_prefix_prefixlen11[str(col) + '_prefixlen' + str(15)] = 0
		data_dummy_prefix_prefixlen11[str(col) + '_prefixlen' + str(16)] = 0
		data_dummy_prefix_prefixlen11[str(col) + '_prefixlen' + str(17)] = 0
		data_dummy_prefix_prefixlen11[str(col) + '_prefixlen' + str(18)] = 0
		data_dummy_prefix_prefixlen11[str(col) + '_prefixlen' + str(19)] = 0
		data_dummy_prefix_prefixlen11[str(col) + '_prefixlen' + str(20)] = 0
	else:
		data_dummy_prefix_prefixlen11[str(col) + '_prefixlen' + str(1)] = 0 #For case-level attributes, only "_prefixlen1" is attached as suffix.
#Here, stay with 761 or change 761 to other number depending on the situation. This code is to delete columns which do not have names (or having numbers as names).
#Check the column names by yourself and determine the number to replace 761. If 761 is right, stick with 761.
data_dummy_prefix_prefixlen11 = data_dummy_prefix_prefixlen11.iloc[:,761:len(data_dummy_prefix_prefixlen11.columns)]
prefix11 = data_dummy[data_dummy['prefix']==11]
prefix11 = prefix11.reset_index()
prefix11 = prefix11.drop("index",1)
prefix11 = prefix11.join(data_dummy_prefix_prefixlen11)
del data_dummy_prefix_prefixlen11
print("data_dummy_prefix_prefixlen11 deleted")

prefix1 = data_dummy[data_dummy['prefix']==1]
prefix1_ref = prefix1[prefix1['case_len']>=11]
prefix1_ref = prefix1_ref.reset_index()
prefix1_ref = prefix1_ref.drop("index",1)
del prefix1
for col in prefix11:
    if col[len(col)-10:len(col)] == "prefixlen1":
        prefix11[col] = prefix1_ref[col[0:len(col)-11]]
del prefix1_ref
print("prefix1_ref_deleted")

prefix2 = data_dummy[data_dummy['prefix']==2]
prefix2_ref = prefix2[prefix2['case_len']>=11]
prefix2_ref = prefix2_ref.reset_index()
prefix2_ref = prefix2_ref.drop("index",1)
del prefix2
for col in prefix11:
    if col[len(col)-10:len(col)] == "prefixlen2":
        prefix11[col] = prefix2_ref[col[0:len(col)-11]]
del prefix2_ref
print("prefix2_ref_deleted")

prefix3 = data_dummy[data_dummy['prefix']==3]
prefix3_ref = prefix3[prefix3['case_len']>=11]
prefix3_ref = prefix3_ref.reset_index()
prefix3_ref = prefix3_ref.drop("index",1)
del prefix3
for col in prefix11:
    if col[len(col)-10:len(col)] == "prefixlen3":
        prefix11[col] = prefix3_ref[col[0:len(col)-11]]
del prefix3_ref
print("prefix3_ref_deleted")

prefix4 = data_dummy[data_dummy['prefix']==4]
prefix4_ref = prefix4[prefix4['case_len']>=11]
prefix4_ref = prefix4_ref.reset_index()
prefix4_ref = prefix4_ref.drop("index",1)
del prefix4
for col in prefix11:
    if col[len(col)-10:len(col)] == "prefixlen4":
        prefix11[col] = prefix4_ref[col[0:len(col)-11]]
del prefix4_ref
print("prefix4_ref_deleted")

prefix5 = data_dummy[data_dummy['prefix']==5]
prefix5_ref = prefix5[prefix5['case_len']>=11]
prefix5_ref = prefix5_ref.reset_index()
prefix5_ref = prefix5_ref.drop("index",1)
del prefix5
for col in prefix11:
    if col[len(col)-10:len(col)] == "prefixlen5":
        prefix11[col] = prefix5_ref[col[0:len(col)-11]]
del prefix5_ref
print("prefix5_ref_deleted")

prefix6 = data_dummy[data_dummy['prefix']==6]
prefix6_ref = prefix6[prefix6['case_len']>=11]
prefix6_ref = prefix6_ref.reset_index()
prefix6_ref = prefix6_ref.drop("index",1)
del prefix6
for col in prefix11:
    if col[len(col)-10:len(col)] == "prefixlen6":
        prefix11[col] = prefix6_ref[col[0:len(col)-11]]
del prefix6_ref
print("prefix6_ref_deleted")

prefix7 = data_dummy[data_dummy['prefix']==7]
prefix7_ref = prefix7[prefix7['case_len']>=11]
prefix7_ref = prefix7_ref.reset_index()
prefix7_ref = prefix7_ref.drop("index",1)
del prefix7
for col in prefix11:
    if col[len(col)-10:len(col)] == "prefixlen7":
        prefix11[col] = prefix7_ref[col[0:len(col)-11]]
del prefix7_ref
print("prefix7_ref_deleted")

prefix8 = data_dummy[data_dummy['prefix']==8]
prefix8_ref = prefix8[prefix8['case_len']>=11]
prefix8_ref = prefix8_ref.reset_index()
prefix8_ref = prefix8_ref.drop("index",1)
del prefix8
for col in prefix11:
    if col[len(col)-10:len(col)] == "prefixlen8":
        prefix11[col] = prefix8_ref[col[0:len(col)-11]]
del prefix8_ref
print("prefix8_ref_deleted")

prefix9 = data_dummy[data_dummy['prefix']==9]
prefix9_ref = prefix9[prefix9['case_len']>=11]
prefix9_ref = prefix9_ref.reset_index()
prefix9_ref = prefix9_ref.drop("index",1)
del prefix9
for col in prefix11:
    if col[len(col)-10:len(col)] == "prefixlen9":
        prefix11[col] = prefix9_ref[col[0:len(col)-11]]
del prefix9_ref
print("prefix9_ref_deleted")

prefix10 = data_dummy[data_dummy['prefix']==10]
del data_dummy
prefix10_ref = prefix10[prefix10['case_len']>=11]
prefix10_ref = prefix10_ref.reset_index()
prefix10_ref = prefix10_ref.drop("index",1)
del prefix10
for col in prefix11:
    if col[len(col)-11:len(col)] == "prefixlen10":
        prefix11[col] = prefix10_ref[col[0:len(col)-12]]
del prefix10_ref
print("prefix10_ref_deleted")

for col in prefix11:
    if col[len(col)-11:len(col)] == "prefixlen11":
        prefix11[col] = prefix11[col[0:len(col)-12]]
prefix11 = prefix11.filter(regex="prefixlen")
prefix11.to_pickle('prefix11.pkl')

##########prefix12##########
data_dummy_prefix_prefixlen12 = pd.DataFrame(index=range(len(data_dummy[data_dummy['prefix']==12])), columns=range(len(data_dummy.columns)))
for col in data_dummy_columns_list:
	print(col)
	if col not in ("case_id","Action","EventOrigin","lifecycle.transition","Accepted","Selected","AMOUNT_REQ","case_len","label"):
		data_dummy_prefix_prefixlen12[str(col) + '_prefixlen' + str(1)] = 0
		data_dummy_prefix_prefixlen12[str(col) + '_prefixlen' + str(2)] = 0
		data_dummy_prefix_prefixlen12[str(col) + '_prefixlen' + str(3)] = 0
		data_dummy_prefix_prefixlen12[str(col) + '_prefixlen' + str(4)] = 0
		data_dummy_prefix_prefixlen12[str(col) + '_prefixlen' + str(5)] = 0
		data_dummy_prefix_prefixlen12[str(col) + '_prefixlen' + str(6)] = 0
		data_dummy_prefix_prefixlen12[str(col) + '_prefixlen' + str(7)] = 0
		data_dummy_prefix_prefixlen12[str(col) + '_prefixlen' + str(8)] = 0
		data_dummy_prefix_prefixlen12[str(col) + '_prefixlen' + str(9)] = 0
		data_dummy_prefix_prefixlen12[str(col) + '_prefixlen' + str(10)] = 0
		data_dummy_prefix_prefixlen12[str(col) + '_prefixlen' + str(11)] = 0
		data_dummy_prefix_prefixlen12[str(col) + '_prefixlen' + str(12)] = 0
		data_dummy_prefix_prefixlen12[str(col) + '_prefixlen' + str(13)] = 0
		data_dummy_prefix_prefixlen12[str(col) + '_prefixlen' + str(14)] = 0
		data_dummy_prefix_prefixlen12[str(col) + '_prefixlen' + str(15)] = 0
		data_dummy_prefix_prefixlen12[str(col) + '_prefixlen' + str(16)] = 0
		data_dummy_prefix_prefixlen12[str(col) + '_prefixlen' + str(17)] = 0
		data_dummy_prefix_prefixlen12[str(col) + '_prefixlen' + str(18)] = 0
		data_dummy_prefix_prefixlen12[str(col) + '_prefixlen' + str(19)] = 0
		data_dummy_prefix_prefixlen12[str(col) + '_prefixlen' + str(20)] = 0
	else:
		data_dummy_prefix_prefixlen12[str(col) + '_prefixlen' + str(1)] = 0 #For case-level attributes, only "_prefixlen1" is attached as suffix.
#Here, stay with 761 or change 761 to other number depending on the situation. This code is to delete columns which do not have names (or having numbers as names).
#Check the column names by yourself and determine the number to replace 761. If 761 is right, stick with 761.
data_dummy_prefix_prefixlen12 = data_dummy_prefix_prefixlen12.iloc[:,761:len(data_dummy_prefix_prefixlen12.columns)]
prefix12 = data_dummy[data_dummy['prefix']==12]
prefix12 = prefix12.reset_index()
prefix12 = prefix12.drop("index",1)
prefix12 = prefix12.join(data_dummy_prefix_prefixlen12)
del data_dummy_prefix_prefixlen12

prefix1 = data_dummy[data_dummy['prefix']==1]
prefix1_ref = prefix1[prefix1['case_len']>=12]
prefix1_ref = prefix1_ref.reset_index()
prefix1_ref = prefix1_ref.drop("index",1)
del prefix1
for col in prefix12:
    if col[len(col)-10:len(col)] == "prefixlen1":
        prefix12[col] = prefix1_ref[col[0:len(col)-11]]
del prefix1_ref

prefix2 = data_dummy[data_dummy['prefix']==2]
prefix2_ref = prefix2[prefix2['case_len']>=12]
prefix2_ref = prefix2_ref.reset_index()
prefix2_ref = prefix2_ref.drop("index",1)
del prefix2
for col in prefix12:
    if col[len(col)-10:len(col)] == "prefixlen2":
        prefix12[col] = prefix2_ref[col[0:len(col)-11]]
del prefix2_ref

prefix3 = data_dummy[data_dummy['prefix']==3]
prefix3_ref = prefix3[prefix3['case_len']>=12]
prefix3_ref = prefix3_ref.reset_index()
prefix3_ref = prefix3_ref.drop("index",1)
del prefix3
for col in prefix12:
    if col[len(col)-10:len(col)] == "prefixlen3":
        prefix12[col] = prefix3_ref[col[0:len(col)-11]]
del prefix3_ref

prefix4 = data_dummy[data_dummy['prefix']==4]
prefix4_ref = prefix4[prefix4['case_len']>=12]
prefix4_ref = prefix4_ref.reset_index()
prefix4_ref = prefix4_ref.drop("index",1)
del prefix4
for col in prefix12:
    if col[len(col)-10:len(col)] == "prefixlen4":
        prefix12[col] = prefix4_ref[col[0:len(col)-11]]
del prefix4_ref

prefix5 = data_dummy[data_dummy['prefix']==5]
prefix5_ref = prefix5[prefix5['case_len']>=12]
prefix5_ref = prefix5_ref.reset_index()
prefix5_ref = prefix5_ref.drop("index",1)
del prefix5
for col in prefix12:
    if col[len(col)-10:len(col)] == "prefixlen5":
        prefix12[col] = prefix5_ref[col[0:len(col)-11]]
del prefix5_ref

prefix6 = data_dummy[data_dummy['prefix']==6]
prefix6_ref = prefix6[prefix6['case_len']>=12]
prefix6_ref = prefix6_ref.reset_index()
prefix6_ref = prefix6_ref.drop("index",1)
del prefix6
for col in prefix12:
    if col[len(col)-10:len(col)] == "prefixlen6":
        prefix12[col] = prefix6_ref[col[0:len(col)-11]]
del prefix6_ref

prefix7 = data_dummy[data_dummy['prefix']==7]
prefix7_ref = prefix7[prefix7['case_len']>=12]
prefix7_ref = prefix7_ref.reset_index()
prefix7_ref = prefix7_ref.drop("index",1)
del prefix7
for col in prefix12:
    if col[len(col)-10:len(col)] == "prefixlen7":
        prefix12[col] = prefix7_ref[col[0:len(col)-11]]
del prefix7_ref

prefix8 = data_dummy[data_dummy['prefix']==8]
prefix8_ref = prefix8[prefix8['case_len']>=12]
prefix8_ref = prefix8_ref.reset_index()
prefix8_ref = prefix8_ref.drop("index",1)
del prefix8
for col in prefix12:
    if col[len(col)-10:len(col)] == "prefixlen8":
        prefix12[col] = prefix8_ref[col[0:len(col)-11]]
del prefix8_ref

prefix9 = data_dummy[data_dummy['prefix']==9]
prefix9_ref = prefix9[prefix9['case_len']>=12]
prefix9_ref = prefix9_ref.reset_index()
prefix9_ref = prefix9_ref.drop("index",1)
del prefix9
for col in prefix12:
    if col[len(col)-10:len(col)] == "prefixlen9":
        prefix12[col] = prefix9_ref[col[0:len(col)-11]]
del prefix9_ref

prefix10 = data_dummy[data_dummy['prefix']==10]
prefix10_ref = prefix10[prefix10['case_len']>=12]
prefix10_ref = prefix10_ref.reset_index()
prefix10_ref = prefix10_ref.drop("index",1)
del prefix10
for col in prefix12:
    if col[len(col)-11:len(col)] == "prefixlen10":
        prefix12[col] = prefix10_ref[col[0:len(col)-12]]
del prefix10_ref

prefix11 = data_dummy[data_dummy['prefix']==11]
del data_dummy
prefix11_ref = prefix11[prefix11['case_len']>=12]
prefix11_ref = prefix11_ref.reset_index()
prefix11_ref = prefix11_ref.drop("index",1)
del prefix11
for col in prefix12:
    if col[len(col)-11:len(col)] == "prefixlen11":
        prefix12[col] = prefix11_ref[col[0:len(col)-12]]
del prefix11_ref

for col in prefix12:
    if col[len(col)-11:len(col)] == "prefixlen12":
        prefix12[col] = prefix12[col[0:len(col)-12]]
prefix12 = prefix12.filter(regex="prefixlen")
prefix12.to_pickle('prefix12.pkl')

##########prefix13##########
data_dummy_prefix_prefixlen13 = pd.DataFrame(index=range(len(data_dummy[data_dummy['prefix']==13])), columns=range(len(data_dummy.columns)))
for col in data_dummy_columns_list:
	print(col)
	if col not in ("case_id","Action","EventOrigin","lifecycle.transition","Accepted","Selected","AMOUNT_REQ","case_len","label"):
		data_dummy_prefix_prefixlen13[str(col) + '_prefixlen' + str(1)] = 0
		data_dummy_prefix_prefixlen13[str(col) + '_prefixlen' + str(2)] = 0
		data_dummy_prefix_prefixlen13[str(col) + '_prefixlen' + str(3)] = 0
		data_dummy_prefix_prefixlen13[str(col) + '_prefixlen' + str(4)] = 0
		data_dummy_prefix_prefixlen13[str(col) + '_prefixlen' + str(5)] = 0
		data_dummy_prefix_prefixlen13[str(col) + '_prefixlen' + str(6)] = 0
		data_dummy_prefix_prefixlen13[str(col) + '_prefixlen' + str(7)] = 0
		data_dummy_prefix_prefixlen13[str(col) + '_prefixlen' + str(8)] = 0
		data_dummy_prefix_prefixlen13[str(col) + '_prefixlen' + str(9)] = 0
		data_dummy_prefix_prefixlen13[str(col) + '_prefixlen' + str(10)] = 0
		data_dummy_prefix_prefixlen13[str(col) + '_prefixlen' + str(11)] = 0
		data_dummy_prefix_prefixlen13[str(col) + '_prefixlen' + str(12)] = 0
		data_dummy_prefix_prefixlen13[str(col) + '_prefixlen' + str(13)] = 0
		data_dummy_prefix_prefixlen13[str(col) + '_prefixlen' + str(14)] = 0
		data_dummy_prefix_prefixlen13[str(col) + '_prefixlen' + str(15)] = 0
		data_dummy_prefix_prefixlen13[str(col) + '_prefixlen' + str(16)] = 0
		data_dummy_prefix_prefixlen13[str(col) + '_prefixlen' + str(17)] = 0
		data_dummy_prefix_prefixlen13[str(col) + '_prefixlen' + str(18)] = 0
		data_dummy_prefix_prefixlen13[str(col) + '_prefixlen' + str(19)] = 0
		data_dummy_prefix_prefixlen13[str(col) + '_prefixlen' + str(20)] = 0
	else:
		data_dummy_prefix_prefixlen13[str(col) + '_prefixlen' + str(1)] = 0 #For case-level attributes, only "_prefixlen1" is attached as suffix.
#Here, stay with 761 or change 761 to other number depending on the situation. This code is to delete columns which do not have names (or having numbers as names).
#Check the column names by yourself and determine the number to replace 761. If 761 is right, stick with 761.
data_dummy_prefix_prefixlen13 = data_dummy_prefix_prefixlen13.iloc[:,761:len(data_dummy_prefix_prefixlen13.columns)]
prefix13 = data_dummy[data_dummy['prefix']==13]
prefix13 = prefix13.reset_index()
prefix13 = prefix13.drop("index",1)
prefix13 = prefix13.join(data_dummy_prefix_prefixlen13)
del data_dummy_prefix_prefixlen13

prefix1 = data_dummy[data_dummy['prefix']==1]
prefix1_ref = prefix1[prefix1['case_len']>=13]
prefix1_ref = prefix1_ref.reset_index()
prefix1_ref = prefix1_ref.drop("index",1)
del prefix1
for col in prefix13:
    if col[len(col)-10:len(col)] == "prefixlen1":
        prefix13[col] = prefix1_ref[col[0:len(col)-11]]
del prefix1_ref

prefix2 = data_dummy[data_dummy['prefix']==2]
prefix2_ref = prefix2[prefix2['case_len']>=13]
prefix2_ref = prefix2_ref.reset_index()
prefix2_ref = prefix2_ref.drop("index",1)
del prefix2
for col in prefix13:
    if col[len(col)-10:len(col)] == "prefixlen2":
        prefix13[col] = prefix2_ref[col[0:len(col)-11]]
del prefix2_ref

prefix3 = data_dummy[data_dummy['prefix']==3]
prefix3_ref = prefix3[prefix3['case_len']>=13]
prefix3_ref = prefix3_ref.reset_index()
prefix3_ref = prefix3_ref.drop("index",1)
del prefix3
for col in prefix13:
    if col[len(col)-10:len(col)] == "prefixlen3":
        prefix13[col] = prefix3_ref[col[0:len(col)-11]]
del prefix3_ref

prefix4 = data_dummy[data_dummy['prefix']==4]
prefix4_ref = prefix4[prefix4['case_len']>=13]
prefix4_ref = prefix4_ref.reset_index()
prefix4_ref = prefix4_ref.drop("index",1)
del prefix4
for col in prefix13:
    if col[len(col)-10:len(col)] == "prefixlen4":
        prefix13[col] = prefix4_ref[col[0:len(col)-11]]
del prefix4_ref

prefix5 = data_dummy[data_dummy['prefix']==5]
prefix5_ref = prefix5[prefix5['case_len']>=13]
prefix5_ref = prefix5_ref.reset_index()
prefix5_ref = prefix5_ref.drop("index",1)
del prefix5
for col in prefix13:
    if col[len(col)-10:len(col)] == "prefixlen5":
        prefix13[col] = prefix5_ref[col[0:len(col)-11]]
del prefix5_ref

prefix6 = data_dummy[data_dummy['prefix']==6]
prefix6_ref = prefix6[prefix6['case_len']>=13]
prefix6_ref = prefix6_ref.reset_index()
prefix6_ref = prefix6_ref.drop("index",1)
del prefix6
for col in prefix13:
    if col[len(col)-10:len(col)] == "prefixlen6":
        prefix13[col] = prefix6_ref[col[0:len(col)-11]]
del prefix6_ref

prefix7 = data_dummy[data_dummy['prefix']==7]
prefix7_ref = prefix7[prefix7['case_len']>=13]
prefix7_ref = prefix7_ref.reset_index()
prefix7_ref = prefix7_ref.drop("index",1)
del prefix7
for col in prefix13:
    if col[len(col)-10:len(col)] == "prefixlen7":
        prefix13[col] = prefix7_ref[col[0:len(col)-11]]
del prefix7_ref

prefix8 = data_dummy[data_dummy['prefix']==8]
prefix8_ref = prefix8[prefix8['case_len']>=13]
prefix8_ref = prefix8_ref.reset_index()
prefix8_ref = prefix8_ref.drop("index",1)
del prefix8
for col in prefix13:
    if col[len(col)-10:len(col)] == "prefixlen8":
        prefix13[col] = prefix8_ref[col[0:len(col)-11]]
del prefix8_ref

prefix9 = data_dummy[data_dummy['prefix']==9]
prefix9_ref = prefix9[prefix9['case_len']>=13]
prefix9_ref = prefix9_ref.reset_index()
prefix9_ref = prefix9_ref.drop("index",1)
del prefix9
for col in prefix13:
    if col[len(col)-10:len(col)] == "prefixlen9":
        prefix13[col] = prefix9_ref[col[0:len(col)-11]]
del prefix9_ref

prefix10 = data_dummy[data_dummy['prefix']==10]
prefix10_ref = prefix10[prefix10['case_len']>=13]
prefix10_ref = prefix10_ref.reset_index()
prefix10_ref = prefix10_ref.drop("index",1)
del prefix10
for col in prefix13:
    if col[len(col)-11:len(col)] == "prefixlen10":
        prefix13[col] = prefix10_ref[col[0:len(col)-12]]
del prefix10_ref

prefix11 = data_dummy[data_dummy['prefix']==11]
prefix11_ref = prefix11[prefix11['case_len']>=13]
prefix11_ref = prefix11_ref.reset_index()
prefix11_ref = prefix11_ref.drop("index",1)
del prefix11
for col in prefix13:
    if col[len(col)-11:len(col)] == "prefixlen11":
        prefix13[col] = prefix11_ref[col[0:len(col)-12]]
del prefix11_ref

prefix12 = data_dummy[data_dummy['prefix']==12]
del data_dummy
prefix12_ref = prefix12[prefix12['case_len']>=13]
prefix12_ref = prefix12_ref.reset_index()
prefix12_ref = prefix12_ref.drop("index",1)
del prefix12
for col in prefix13:
    if col[len(col)-11:len(col)] == "prefixlen12":
        prefix13[col] = prefix12_ref[col[0:len(col)-12]]
del prefix12_ref

for col in prefix13:
    if col[len(col)-11:len(col)] == "prefixlen13":
        prefix13[col] = prefix13[col[0:len(col)-12]]
prefix13 = prefix13.filter(regex="prefixlen")
prefix13.to_pickle('prefix13.pkl')

##########prefix14##########
data_dummy_prefix_prefixlen14 = pd.DataFrame(index=range(len(data_dummy[data_dummy['prefix']==14])), columns=range(len(data_dummy.columns)))
for col in data_dummy_columns_list:
	print(col)
	if col not in ("case_id","Action","EventOrigin","lifecycle.transition","Accepted","Selected","AMOUNT_REQ","case_len","label"):
		data_dummy_prefix_prefixlen14[str(col) + '_prefixlen' + str(1)] = 0
		data_dummy_prefix_prefixlen14[str(col) + '_prefixlen' + str(2)] = 0
		data_dummy_prefix_prefixlen14[str(col) + '_prefixlen' + str(3)] = 0
		data_dummy_prefix_prefixlen14[str(col) + '_prefixlen' + str(4)] = 0
		data_dummy_prefix_prefixlen14[str(col) + '_prefixlen' + str(5)] = 0
		data_dummy_prefix_prefixlen14[str(col) + '_prefixlen' + str(6)] = 0
		data_dummy_prefix_prefixlen14[str(col) + '_prefixlen' + str(7)] = 0
		data_dummy_prefix_prefixlen14[str(col) + '_prefixlen' + str(8)] = 0
		data_dummy_prefix_prefixlen14[str(col) + '_prefixlen' + str(9)] = 0
		data_dummy_prefix_prefixlen14[str(col) + '_prefixlen' + str(10)] = 0
		data_dummy_prefix_prefixlen14[str(col) + '_prefixlen' + str(11)] = 0
		data_dummy_prefix_prefixlen14[str(col) + '_prefixlen' + str(12)] = 0
		data_dummy_prefix_prefixlen14[str(col) + '_prefixlen' + str(13)] = 0
		data_dummy_prefix_prefixlen14[str(col) + '_prefixlen' + str(14)] = 0
		data_dummy_prefix_prefixlen14[str(col) + '_prefixlen' + str(15)] = 0
		data_dummy_prefix_prefixlen14[str(col) + '_prefixlen' + str(16)] = 0
		data_dummy_prefix_prefixlen14[str(col) + '_prefixlen' + str(17)] = 0
		data_dummy_prefix_prefixlen14[str(col) + '_prefixlen' + str(18)] = 0
		data_dummy_prefix_prefixlen14[str(col) + '_prefixlen' + str(19)] = 0
		data_dummy_prefix_prefixlen14[str(col) + '_prefixlen' + str(20)] = 0
	else:
		data_dummy_prefix_prefixlen14[str(col) + '_prefixlen' + str(1)] = 0 #For case-level attributes, only "_prefixlen1" is attached as suffix.
#Here, stay with 761 or change 761 to other number depending on the situation. This code is to delete columns which do not have names (or having numbers as names).
#Check the column names by yourself and determine the number to replace 761. If 761 is right, stick with 761.
data_dummy_prefix_prefixlen14 = data_dummy_prefix_prefixlen14.iloc[:,761:len(data_dummy_prefix_prefixlen14.columns)]
prefix14 = data_dummy[data_dummy['prefix']==14]
prefix14 = prefix14.reset_index()
prefix14 = prefix14.drop("index",1)
prefix14 = prefix14.join(data_dummy_prefix_prefixlen14)
del data_dummy_prefix_prefixlen14

prefix1 = data_dummy[data_dummy['prefix']==1]
prefix1_ref = prefix1[prefix1['case_len']>=14]
prefix1_ref = prefix1_ref.reset_index()
prefix1_ref = prefix1_ref.drop("index",1)
del prefix1
for col in prefix14:
    if col[len(col)-10:len(col)] == "prefixlen1":
        prefix14[col] = prefix1_ref[col[0:len(col)-11]]
del prefix1_ref

prefix2 = data_dummy[data_dummy['prefix']==2]
prefix2_ref = prefix2[prefix2['case_len']>=14]
prefix2_ref = prefix2_ref.reset_index()
prefix2_ref = prefix2_ref.drop("index",1)
del prefix2
for col in prefix14:
    if col[len(col)-10:len(col)] == "prefixlen2":
        prefix14[col] = prefix2_ref[col[0:len(col)-11]]
del prefix2_ref

prefix3 = data_dummy[data_dummy['prefix']==3]
prefix3_ref = prefix3[prefix3['case_len']>=14]
prefix3_ref = prefix3_ref.reset_index()
prefix3_ref = prefix3_ref.drop("index",1)
del prefix3
for col in prefix14:
    if col[len(col)-10:len(col)] == "prefixlen3":
        prefix14[col] = prefix3_ref[col[0:len(col)-11]]
del prefix3_ref

prefix4 = data_dummy[data_dummy['prefix']==4]
prefix4_ref = prefix4[prefix4['case_len']>=14]
prefix4_ref = prefix4_ref.reset_index()
prefix4_ref = prefix4_ref.drop("index",1)
del prefix4
for col in prefix14:
    if col[len(col)-10:len(col)] == "prefixlen4":
        prefix14[col] = prefix4_ref[col[0:len(col)-11]]
del prefix4_ref

prefix5 = data_dummy[data_dummy['prefix']==5]
prefix5_ref = prefix5[prefix5['case_len']>=14]
prefix5_ref = prefix5_ref.reset_index()
prefix5_ref = prefix5_ref.drop("index",1)
del prefix5
for col in prefix14:
    if col[len(col)-10:len(col)] == "prefixlen5":
        prefix14[col] = prefix5_ref[col[0:len(col)-11]]
del prefix5_ref

prefix6 = data_dummy[data_dummy['prefix']==6]
prefix6_ref = prefix6[prefix6['case_len']>=14]
prefix6_ref = prefix6_ref.reset_index()
prefix6_ref = prefix6_ref.drop("index",1)
del prefix6
for col in prefix14:
    if col[len(col)-10:len(col)] == "prefixlen6":
        prefix14[col] = prefix6_ref[col[0:len(col)-11]]
del prefix6_ref

prefix7 = data_dummy[data_dummy['prefix']==7]
prefix7_ref = prefix7[prefix7['case_len']>=14]
prefix7_ref = prefix7_ref.reset_index()
prefix7_ref = prefix7_ref.drop("index",1)
del prefix7
for col in prefix14:
    if col[len(col)-10:len(col)] == "prefixlen7":
        prefix14[col] = prefix7_ref[col[0:len(col)-11]]
del prefix7_ref

prefix8 = data_dummy[data_dummy['prefix']==8]
prefix8_ref = prefix8[prefix8['case_len']>=14]
prefix8_ref = prefix8_ref.reset_index()
prefix8_ref = prefix8_ref.drop("index",1)
del prefix8
for col in prefix14:
    if col[len(col)-10:len(col)] == "prefixlen8":
        prefix14[col] = prefix8_ref[col[0:len(col)-11]]
del prefix8_ref

prefix9 = data_dummy[data_dummy['prefix']==9]
prefix9_ref = prefix9[prefix9['case_len']>=14]
prefix9_ref = prefix9_ref.reset_index()
prefix9_ref = prefix9_ref.drop("index",1)
del prefix9
for col in prefix14:
    if col[len(col)-10:len(col)] == "prefixlen9":
        prefix14[col] = prefix9_ref[col[0:len(col)-11]]
del prefix9_ref

prefix10 = data_dummy[data_dummy['prefix']==10]
prefix10_ref = prefix10[prefix10['case_len']>=14]
prefix10_ref = prefix10_ref.reset_index()
prefix10_ref = prefix10_ref.drop("index",1)
del prefix10
for col in prefix14:
    if col[len(col)-11:len(col)] == "prefixlen10":
        prefix14[col] = prefix10_ref[col[0:len(col)-12]]
del prefix10_ref

prefix11 = data_dummy[data_dummy['prefix']==11]
prefix11_ref = prefix11[prefix11['case_len']>=14]
prefix11_ref = prefix11_ref.reset_index()
prefix11_ref = prefix11_ref.drop("index",1)
del prefix11
for col in prefix14:
    if col[len(col)-11:len(col)] == "prefixlen11":
        prefix14[col] = prefix11_ref[col[0:len(col)-12]]
del prefix11_ref

prefix12 = data_dummy[data_dummy['prefix']==12]
prefix12_ref = prefix12[prefix12['case_len']>=14]
prefix12_ref = prefix12_ref.reset_index()
prefix12_ref = prefix12_ref.drop("index",1)
del prefix12
for col in prefix14:
    if col[len(col)-11:len(col)] == "prefixlen12":
        prefix14[col] = prefix12_ref[col[0:len(col)-12]]
del prefix12_ref

prefix13 = data_dummy[data_dummy['prefix']==13]
del data_dummy
prefix13_ref = prefix13[prefix13['case_len']>=14]
prefix13_ref = prefix13_ref.reset_index()
prefix13_ref = prefix13_ref.drop("index",1)
del prefix13
for col in prefix14:
    if col[len(col)-11:len(col)] == "prefixlen13":
        prefix14[col] = prefix13_ref[col[0:len(col)-12]]
del prefix13_ref

for col in prefix14:
    if col[len(col)-11:len(col)] == "prefixlen14":
        prefix14[col] = prefix14[col[0:len(col)-12]]
prefix14 = prefix14.filter(regex="prefixlen")
prefix14.to_pickle('prefix14.pkl')

##########prefix15##########
data_dummy_prefix_prefixlen15 = pd.DataFrame(index=range(len(data_dummy[data_dummy['prefix']==15])), columns=range(len(data_dummy.columns)))
for col in data_dummy_columns_list:
	print(col)
	if col not in ("case_id","Action","EventOrigin","lifecycle.transition","Accepted","Selected","AMOUNT_REQ","case_len","label"):
		data_dummy_prefix_prefixlen15[str(col) + '_prefixlen' + str(1)] = 0
		data_dummy_prefix_prefixlen15[str(col) + '_prefixlen' + str(2)] = 0
		data_dummy_prefix_prefixlen15[str(col) + '_prefixlen' + str(3)] = 0
		data_dummy_prefix_prefixlen15[str(col) + '_prefixlen' + str(4)] = 0
		data_dummy_prefix_prefixlen15[str(col) + '_prefixlen' + str(5)] = 0
		data_dummy_prefix_prefixlen15[str(col) + '_prefixlen' + str(6)] = 0
		data_dummy_prefix_prefixlen15[str(col) + '_prefixlen' + str(7)] = 0
		data_dummy_prefix_prefixlen15[str(col) + '_prefixlen' + str(8)] = 0
		data_dummy_prefix_prefixlen15[str(col) + '_prefixlen' + str(9)] = 0
		data_dummy_prefix_prefixlen15[str(col) + '_prefixlen' + str(10)] = 0
		data_dummy_prefix_prefixlen15[str(col) + '_prefixlen' + str(11)] = 0
		data_dummy_prefix_prefixlen15[str(col) + '_prefixlen' + str(12)] = 0
		data_dummy_prefix_prefixlen15[str(col) + '_prefixlen' + str(13)] = 0
		data_dummy_prefix_prefixlen15[str(col) + '_prefixlen' + str(14)] = 0
		data_dummy_prefix_prefixlen15[str(col) + '_prefixlen' + str(15)] = 0
		data_dummy_prefix_prefixlen15[str(col) + '_prefixlen' + str(16)] = 0
		data_dummy_prefix_prefixlen15[str(col) + '_prefixlen' + str(17)] = 0
		data_dummy_prefix_prefixlen15[str(col) + '_prefixlen' + str(18)] = 0
		data_dummy_prefix_prefixlen15[str(col) + '_prefixlen' + str(19)] = 0
		data_dummy_prefix_prefixlen15[str(col) + '_prefixlen' + str(20)] = 0
	else:
		data_dummy_prefix_prefixlen15[str(col) + '_prefixlen' + str(1)] = 0 #For case-level attributes, only "_prefixlen1" is attached as suffix.
#Here, stay with 761 or change 761 to other number depending on the situation. This code is to delete columns which do not have names (or having numbers as names).
#Check the column names by yourself and determine the number to replace 761. If 761 is right, stick with 761.
data_dummy_prefix_prefixlen15 = data_dummy_prefix_prefixlen15.iloc[:,761:len(data_dummy_prefix_prefixlen15.columns)]
prefix15 = data_dummy[data_dummy['prefix']==15]
prefix15 = prefix15.reset_index()
prefix15 = prefix15.drop("index",1)
prefix15 = prefix15.join(data_dummy_prefix_prefixlen15)
del data_dummy_prefix_prefixlen15

prefix1 = data_dummy[data_dummy['prefix']==1]
prefix1_ref = prefix1[prefix1['case_len']>=15]
prefix1_ref = prefix1_ref.reset_index()
prefix1_ref = prefix1_ref.drop("index",1)
del prefix1
for col in prefix15:
    if col[len(col)-10:len(col)] == "prefixlen1":
        prefix15[col] = prefix1_ref[col[0:len(col)-11]]
del prefix1_ref

prefix2 = data_dummy[data_dummy['prefix']==2]
prefix2_ref = prefix2[prefix2['case_len']>=15]
prefix2_ref = prefix2_ref.reset_index()
prefix2_ref = prefix2_ref.drop("index",1)
del prefix2
for col in prefix15:
    if col[len(col)-10:len(col)] == "prefixlen2":
        prefix15[col] = prefix2_ref[col[0:len(col)-11]]
del prefix2_ref

prefix3 = data_dummy[data_dummy['prefix']==3]
prefix3_ref = prefix3[prefix3['case_len']>=15]
prefix3_ref = prefix3_ref.reset_index()
prefix3_ref = prefix3_ref.drop("index",1)
del prefix3
for col in prefix15:
    if col[len(col)-10:len(col)] == "prefixlen3":
        prefix15[col] = prefix3_ref[col[0:len(col)-11]]
del prefix3_ref

prefix4 = data_dummy[data_dummy['prefix']==4]
prefix4_ref = prefix4[prefix4['case_len']>=15]
prefix4_ref = prefix4_ref.reset_index()
prefix4_ref = prefix4_ref.drop("index",1)
del prefix4
for col in prefix15:
    if col[len(col)-10:len(col)] == "prefixlen4":
        prefix15[col] = prefix4_ref[col[0:len(col)-11]]
del prefix4_ref

prefix5 = data_dummy[data_dummy['prefix']==5]
prefix5_ref = prefix5[prefix5['case_len']>=15]
prefix5_ref = prefix5_ref.reset_index()
prefix5_ref = prefix5_ref.drop("index",1)
del prefix5
for col in prefix15:
    if col[len(col)-10:len(col)] == "prefixlen5":
        prefix15[col] = prefix5_ref[col[0:len(col)-11]]
del prefix5_ref

prefix6 = data_dummy[data_dummy['prefix']==6]
prefix6_ref = prefix6[prefix6['case_len']>=15]
prefix6_ref = prefix6_ref.reset_index()
prefix6_ref = prefix6_ref.drop("index",1)
del prefix6
for col in prefix15:
    if col[len(col)-10:len(col)] == "prefixlen6":
        prefix15[col] = prefix6_ref[col[0:len(col)-11]]
del prefix6_ref

prefix7 = data_dummy[data_dummy['prefix']==7]
prefix7_ref = prefix7[prefix7['case_len']>=15]
prefix7_ref = prefix7_ref.reset_index()
prefix7_ref = prefix7_ref.drop("index",1)
del prefix7
for col in prefix15:
    if col[len(col)-10:len(col)] == "prefixlen7":
        prefix15[col] = prefix7_ref[col[0:len(col)-11]]
del prefix7_ref

prefix8 = data_dummy[data_dummy['prefix']==8]
prefix8_ref = prefix8[prefix8['case_len']>=15]
prefix8_ref = prefix8_ref.reset_index()
prefix8_ref = prefix8_ref.drop("index",1)
del prefix8
for col in prefix15:
    if col[len(col)-10:len(col)] == "prefixlen8":
        prefix15[col] = prefix8_ref[col[0:len(col)-11]]
del prefix8_ref

prefix9 = data_dummy[data_dummy['prefix']==9]
prefix9_ref = prefix9[prefix9['case_len']>=15]
prefix9_ref = prefix9_ref.reset_index()
prefix9_ref = prefix9_ref.drop("index",1)
del prefix9
for col in prefix15:
    if col[len(col)-10:len(col)] == "prefixlen9":
        prefix15[col] = prefix9_ref[col[0:len(col)-11]]
del prefix9_ref

prefix10 = data_dummy[data_dummy['prefix']==10]
prefix10_ref = prefix10[prefix10['case_len']>=15]
prefix10_ref = prefix10_ref.reset_index()
prefix10_ref = prefix10_ref.drop("index",1)
del prefix10
for col in prefix15:
    if col[len(col)-11:len(col)] == "prefixlen10":
        prefix15[col] = prefix10_ref[col[0:len(col)-12]]
del prefix10_ref

prefix11 = data_dummy[data_dummy['prefix']==11]
prefix11_ref = prefix11[prefix11['case_len']>=15]
prefix11_ref = prefix11_ref.reset_index()
prefix11_ref = prefix11_ref.drop("index",1)
del prefix11
for col in prefix15:
    if col[len(col)-11:len(col)] == "prefixlen11":
        prefix15[col] = prefix11_ref[col[0:len(col)-12]]
del prefix11_ref

prefix12 = data_dummy[data_dummy['prefix']==12]
prefix12_ref = prefix12[prefix12['case_len']>=15]
prefix12_ref = prefix12_ref.reset_index()
prefix12_ref = prefix12_ref.drop("index",1)
del prefix12
for col in prefix15:
    if col[len(col)-11:len(col)] == "prefixlen12":
        prefix15[col] = prefix12_ref[col[0:len(col)-12]]
del prefix12_ref

prefix13 = data_dummy[data_dummy['prefix']==13]
prefix13_ref = prefix13[prefix13['case_len']>=15]
prefix13_ref = prefix13_ref.reset_index()
prefix13_ref = prefix13_ref.drop("index",1)
del prefix13
for col in prefix15:
    if col[len(col)-11:len(col)] == "prefixlen13":
        prefix15[col] = prefix13_ref[col[0:len(col)-12]]
del prefix13_ref

prefix14 = data_dummy[data_dummy['prefix']==14]
del data_dummy
prefix14_ref = prefix14[prefix14['case_len']>=15]
prefix14_ref = prefix14_ref.reset_index()
prefix14_ref = prefix14_ref.drop("index",1)
del prefix14
for col in prefix15:
    if col[len(col)-11:len(col)] == "prefixlen14":
        prefix15[col] = prefix14_ref[col[0:len(col)-12]]
del prefix14_ref

for col in prefix15:
    if col[len(col)-11:len(col)] == "prefixlen15":
        prefix15[col] = prefix15[col[0:len(col)-12]]
prefix15 = prefix15.filter(regex="prefixlen")
prefix15.to_pickle('prefix15.pkl')

##########prefix16##########
data_dummy_prefix_prefixlen16 = pd.DataFrame(index=range(len(data_dummy[data_dummy['prefix']==16])), columns=range(len(data_dummy.columns)))
for col in data_dummy_columns_list:
	print(col)
	if col not in ("case_id","Action","EventOrigin","lifecycle.transition","Accepted","Selected","AMOUNT_REQ","case_len","label"):
		data_dummy_prefix_prefixlen16[str(col) + '_prefixlen' + str(1)] = 0
		data_dummy_prefix_prefixlen16[str(col) + '_prefixlen' + str(2)] = 0
		data_dummy_prefix_prefixlen16[str(col) + '_prefixlen' + str(3)] = 0
		data_dummy_prefix_prefixlen16[str(col) + '_prefixlen' + str(4)] = 0
		data_dummy_prefix_prefixlen16[str(col) + '_prefixlen' + str(5)] = 0
		data_dummy_prefix_prefixlen16[str(col) + '_prefixlen' + str(6)] = 0
		data_dummy_prefix_prefixlen16[str(col) + '_prefixlen' + str(7)] = 0
		data_dummy_prefix_prefixlen16[str(col) + '_prefixlen' + str(8)] = 0
		data_dummy_prefix_prefixlen16[str(col) + '_prefixlen' + str(9)] = 0
		data_dummy_prefix_prefixlen16[str(col) + '_prefixlen' + str(10)] = 0
		data_dummy_prefix_prefixlen16[str(col) + '_prefixlen' + str(11)] = 0
		data_dummy_prefix_prefixlen16[str(col) + '_prefixlen' + str(12)] = 0
		data_dummy_prefix_prefixlen16[str(col) + '_prefixlen' + str(13)] = 0
		data_dummy_prefix_prefixlen16[str(col) + '_prefixlen' + str(14)] = 0
		data_dummy_prefix_prefixlen16[str(col) + '_prefixlen' + str(15)] = 0
		data_dummy_prefix_prefixlen16[str(col) + '_prefixlen' + str(16)] = 0
		data_dummy_prefix_prefixlen16[str(col) + '_prefixlen' + str(17)] = 0
		data_dummy_prefix_prefixlen16[str(col) + '_prefixlen' + str(18)] = 0
		data_dummy_prefix_prefixlen16[str(col) + '_prefixlen' + str(19)] = 0
		data_dummy_prefix_prefixlen16[str(col) + '_prefixlen' + str(20)] = 0
	else:
		data_dummy_prefix_prefixlen16[str(col) + '_prefixlen' + str(1)] = 0 #For case-level attributes, only "_prefixlen1" is attached as suffix.
#Here, stay with 761 or change 761 to other number depending on the situation. This code is to delete columns which do not have names (or having numbers as names).
#Check the column names by yourself and determine the number to replace 761. If 761 is right, stick with 761.
data_dummy_prefix_prefixlen16 = data_dummy_prefix_prefixlen16.iloc[:,761:len(data_dummy_prefix_prefixlen16.columns)]
prefix16 = data_dummy[data_dummy['prefix']==16]
prefix16 = prefix16.reset_index()
prefix16 = prefix16.drop("index",1)
prefix16 = prefix16.join(data_dummy_prefix_prefixlen16)
del data_dummy_prefix_prefixlen16

prefix1 = data_dummy[data_dummy['prefix']==1]
prefix1_ref = prefix1[prefix1['case_len']>=16]
prefix1_ref = prefix1_ref.reset_index()
prefix1_ref = prefix1_ref.drop("index",1)
del prefix1
for col in prefix16:
    if col[len(col)-10:len(col)] == "prefixlen1":
        prefix16[col] = prefix1_ref[col[0:len(col)-11]]
del prefix1_ref

prefix2 = data_dummy[data_dummy['prefix']==2]
prefix2_ref = prefix2[prefix2['case_len']>=16]
prefix2_ref = prefix2_ref.reset_index()
prefix2_ref = prefix2_ref.drop("index",1)
del prefix2
for col in prefix16:
    if col[len(col)-10:len(col)] == "prefixlen2":
        prefix16[col] = prefix2_ref[col[0:len(col)-11]]
del prefix2_ref

prefix3 = data_dummy[data_dummy['prefix']==3]
prefix3_ref = prefix3[prefix3['case_len']>=16]
prefix3_ref = prefix3_ref.reset_index()
prefix3_ref = prefix3_ref.drop("index",1)
del prefix3
for col in prefix16:
    if col[len(col)-10:len(col)] == "prefixlen3":
        prefix16[col] = prefix3_ref[col[0:len(col)-11]]
del prefix3_ref

prefix4 = data_dummy[data_dummy['prefix']==4]
prefix4_ref = prefix4[prefix4['case_len']>=16]
prefix4_ref = prefix4_ref.reset_index()
prefix4_ref = prefix4_ref.drop("index",1)
del prefix4
for col in prefix16:
    if col[len(col)-10:len(col)] == "prefixlen4":
        prefix16[col] = prefix4_ref[col[0:len(col)-11]]
del prefix4_ref

prefix5 = data_dummy[data_dummy['prefix']==5]
prefix5_ref = prefix5[prefix5['case_len']>=16]
prefix5_ref = prefix5_ref.reset_index()
prefix5_ref = prefix5_ref.drop("index",1)
del prefix5
for col in prefix16:
    if col[len(col)-10:len(col)] == "prefixlen5":
        prefix16[col] = prefix5_ref[col[0:len(col)-11]]
del prefix5_ref

prefix6 = data_dummy[data_dummy['prefix']==6]
prefix6_ref = prefix6[prefix6['case_len']>=16]
prefix6_ref = prefix6_ref.reset_index()
prefix6_ref = prefix6_ref.drop("index",1)
del prefix6
for col in prefix16:
    if col[len(col)-10:len(col)] == "prefixlen6":
        prefix16[col] = prefix6_ref[col[0:len(col)-11]]
del prefix6_ref

prefix7 = data_dummy[data_dummy['prefix']==7]
prefix7_ref = prefix7[prefix7['case_len']>=16]
prefix7_ref = prefix7_ref.reset_index()
prefix7_ref = prefix7_ref.drop("index",1)
del prefix7
for col in prefix16:
    if col[len(col)-10:len(col)] == "prefixlen7":
        prefix16[col] = prefix7_ref[col[0:len(col)-11]]
del prefix7_ref

prefix8 = data_dummy[data_dummy['prefix']==8]
prefix8_ref = prefix8[prefix8['case_len']>=16]
prefix8_ref = prefix8_ref.reset_index()
prefix8_ref = prefix8_ref.drop("index",1)
del prefix8
for col in prefix16:
    if col[len(col)-10:len(col)] == "prefixlen8":
        prefix16[col] = prefix8_ref[col[0:len(col)-11]]
del prefix8_ref

prefix9 = data_dummy[data_dummy['prefix']==9]
prefix9_ref = prefix9[prefix9['case_len']>=16]
prefix9_ref = prefix9_ref.reset_index()
prefix9_ref = prefix9_ref.drop("index",1)
del prefix9
for col in prefix16:
    if col[len(col)-10:len(col)] == "prefixlen9":
        prefix16[col] = prefix9_ref[col[0:len(col)-11]]
del prefix9_ref

prefix10 = data_dummy[data_dummy['prefix']==10]
prefix10_ref = prefix10[prefix10['case_len']>=16]
prefix10_ref = prefix10_ref.reset_index()
prefix10_ref = prefix10_ref.drop("index",1)
del prefix10
for col in prefix16:
    if col[len(col)-11:len(col)] == "prefixlen10":
        prefix16[col] = prefix10_ref[col[0:len(col)-12]]
del prefix10_ref

prefix11 = data_dummy[data_dummy['prefix']==11]
prefix11_ref = prefix11[prefix11['case_len']>=16]
prefix11_ref = prefix11_ref.reset_index()
prefix11_ref = prefix11_ref.drop("index",1)
del prefix11
for col in prefix16:
    if col[len(col)-11:len(col)] == "prefixlen11":
        prefix16[col] = prefix11_ref[col[0:len(col)-12]]
del prefix11_ref

prefix12 = data_dummy[data_dummy['prefix']==12]
prefix12_ref = prefix12[prefix12['case_len']>=16]
prefix12_ref = prefix12_ref.reset_index()
prefix12_ref = prefix12_ref.drop("index",1)
del prefix12
for col in prefix16:
    if col[len(col)-11:len(col)] == "prefixlen12":
        prefix16[col] = prefix12_ref[col[0:len(col)-12]]
del prefix12_ref

prefix13 = data_dummy[data_dummy['prefix']==13]
prefix13_ref = prefix13[prefix13['case_len']>=16]
prefix13_ref = prefix13_ref.reset_index()
prefix13_ref = prefix13_ref.drop("index",1)
del prefix13
for col in prefix16:
    if col[len(col)-11:len(col)] == "prefixlen13":
        prefix16[col] = prefix13_ref[col[0:len(col)-12]]
del prefix13_ref

prefix14 = data_dummy[data_dummy['prefix']==14]
prefix14_ref = prefix14[prefix14['case_len']>=16]
prefix14_ref = prefix14_ref.reset_index()
prefix14_ref = prefix14_ref.drop("index",1)
del prefix14
for col in prefix16:
    if col[len(col)-11:len(col)] == "prefixlen14":
        prefix16[col] = prefix14_ref[col[0:len(col)-12]]
del prefix14_ref

prefix15 = data_dummy[data_dummy['prefix']==15]
del data_dummy
prefix15_ref = prefix15[prefix15['case_len']>=16]
prefix15_ref = prefix15_ref.reset_index()
prefix15_ref = prefix15_ref.drop("index",1)
del prefix15
for col in prefix16:
    if col[len(col)-11:len(col)] == "prefixlen15":
        prefix16[col] = prefix15_ref[col[0:len(col)-12]]
del prefix15_ref

for col in prefix16:
    if col[len(col)-11:len(col)] == "prefixlen16":
        prefix16[col] = prefix16[col[0:len(col)-12]]
prefix16 = prefix16.filter(regex="prefixlen")
prefix16.to_pickle('prefix16.pkl')

##########prefix17##########
data_dummy_prefix_prefixlen17 = pd.DataFrame(index=range(len(data_dummy[data_dummy['prefix']==17])), columns=range(len(data_dummy.columns)))
for col in data_dummy_columns_list:
	print(col)
	if col not in ("case_id","Action","EventOrigin","lifecycle.transition","Accepted","Selected","AMOUNT_REQ","case_len","label"):
		data_dummy_prefix_prefixlen17[str(col) + '_prefixlen' + str(1)] = 0
		data_dummy_prefix_prefixlen17[str(col) + '_prefixlen' + str(2)] = 0
		data_dummy_prefix_prefixlen17[str(col) + '_prefixlen' + str(3)] = 0
		data_dummy_prefix_prefixlen17[str(col) + '_prefixlen' + str(4)] = 0
		data_dummy_prefix_prefixlen17[str(col) + '_prefixlen' + str(5)] = 0
		data_dummy_prefix_prefixlen17[str(col) + '_prefixlen' + str(6)] = 0
		data_dummy_prefix_prefixlen17[str(col) + '_prefixlen' + str(7)] = 0
		data_dummy_prefix_prefixlen17[str(col) + '_prefixlen' + str(8)] = 0
		data_dummy_prefix_prefixlen17[str(col) + '_prefixlen' + str(9)] = 0
		data_dummy_prefix_prefixlen17[str(col) + '_prefixlen' + str(10)] = 0
		data_dummy_prefix_prefixlen17[str(col) + '_prefixlen' + str(11)] = 0
		data_dummy_prefix_prefixlen17[str(col) + '_prefixlen' + str(12)] = 0
		data_dummy_prefix_prefixlen17[str(col) + '_prefixlen' + str(13)] = 0
		data_dummy_prefix_prefixlen17[str(col) + '_prefixlen' + str(14)] = 0
		data_dummy_prefix_prefixlen17[str(col) + '_prefixlen' + str(15)] = 0
		data_dummy_prefix_prefixlen17[str(col) + '_prefixlen' + str(16)] = 0
		data_dummy_prefix_prefixlen17[str(col) + '_prefixlen' + str(17)] = 0
		data_dummy_prefix_prefixlen17[str(col) + '_prefixlen' + str(18)] = 0
		data_dummy_prefix_prefixlen17[str(col) + '_prefixlen' + str(19)] = 0
		data_dummy_prefix_prefixlen17[str(col) + '_prefixlen' + str(20)] = 0
	else:
		data_dummy_prefix_prefixlen17[str(col) + '_prefixlen' + str(1)] = 0 #For case-level attributes, only "_prefixlen1" is attached as suffix.
#Here, stay with 761 or change 761 to other number depending on the situation. This code is to delete columns which do not have names (or having numbers as names).
#Check the column names by yourself and determine the number to replace 761. If 761 is right, stick with 761.
data_dummy_prefix_prefixlen17 = data_dummy_prefix_prefixlen17.iloc[:,761:len(data_dummy_prefix_prefixlen17.columns)]
prefix17 = data_dummy[data_dummy['prefix']==17]
prefix17 = prefix17.reset_index()
prefix17 = prefix17.drop("index",1)
prefix17 = prefix17.join(data_dummy_prefix_prefixlen17)
del data_dummy_prefix_prefixlen17

prefix1 = data_dummy[data_dummy['prefix']==1]
prefix1_ref = prefix1[prefix1['case_len']>=17]
prefix1_ref = prefix1_ref.reset_index()
prefix1_ref = prefix1_ref.drop("index",1)
del prefix1
for col in prefix17:
    if col[len(col)-10:len(col)] == "prefixlen1":
        prefix17[col] = prefix1_ref[col[0:len(col)-11]]
del prefix1_ref

prefix2 = data_dummy[data_dummy['prefix']==2]
prefix2_ref = prefix2[prefix2['case_len']>=17]
prefix2_ref = prefix2_ref.reset_index()
prefix2_ref = prefix2_ref.drop("index",1)
del prefix2
for col in prefix17:
    if col[len(col)-10:len(col)] == "prefixlen2":
        prefix17[col] = prefix2_ref[col[0:len(col)-11]]
del prefix2_ref

prefix3 = data_dummy[data_dummy['prefix']==3]
prefix3_ref = prefix3[prefix3['case_len']>=17]
prefix3_ref = prefix3_ref.reset_index()
prefix3_ref = prefix3_ref.drop("index",1)
del prefix3
for col in prefix17:
    if col[len(col)-10:len(col)] == "prefixlen3":
        prefix17[col] = prefix3_ref[col[0:len(col)-11]]
del prefix3_ref

prefix4 = data_dummy[data_dummy['prefix']==4]
prefix4_ref = prefix4[prefix4['case_len']>=17]
prefix4_ref = prefix4_ref.reset_index()
prefix4_ref = prefix4_ref.drop("index",1)
del prefix4
for col in prefix17:
    if col[len(col)-10:len(col)] == "prefixlen4":
        prefix17[col] = prefix4_ref[col[0:len(col)-11]]
del prefix4_ref

prefix5 = data_dummy[data_dummy['prefix']==5]
prefix5_ref = prefix5[prefix5['case_len']>=17]
prefix5_ref = prefix5_ref.reset_index()
prefix5_ref = prefix5_ref.drop("index",1)
del prefix5
for col in prefix17:
    if col[len(col)-10:len(col)] == "prefixlen5":
        prefix17[col] = prefix5_ref[col[0:len(col)-11]]
del prefix5_ref

prefix6 = data_dummy[data_dummy['prefix']==6]
prefix6_ref = prefix6[prefix6['case_len']>=17]
prefix6_ref = prefix6_ref.reset_index()
prefix6_ref = prefix6_ref.drop("index",1)
del prefix6
for col in prefix17:
    if col[len(col)-10:len(col)] == "prefixlen6":
        prefix17[col] = prefix6_ref[col[0:len(col)-11]]
del prefix6_ref

prefix7 = data_dummy[data_dummy['prefix']==7]
prefix7_ref = prefix7[prefix7['case_len']>=17]
prefix7_ref = prefix7_ref.reset_index()
prefix7_ref = prefix7_ref.drop("index",1)
del prefix7
for col in prefix17:
    if col[len(col)-10:len(col)] == "prefixlen7":
        prefix17[col] = prefix7_ref[col[0:len(col)-11]]
del prefix7_ref

prefix8 = data_dummy[data_dummy['prefix']==8]
prefix8_ref = prefix8[prefix8['case_len']>=17]
prefix8_ref = prefix8_ref.reset_index()
prefix8_ref = prefix8_ref.drop("index",1)
del prefix8
for col in prefix17:
    if col[len(col)-10:len(col)] == "prefixlen8":
        prefix17[col] = prefix8_ref[col[0:len(col)-11]]
del prefix8_ref

prefix9 = data_dummy[data_dummy['prefix']==9]
prefix9_ref = prefix9[prefix9['case_len']>=17]
prefix9_ref = prefix9_ref.reset_index()
prefix9_ref = prefix9_ref.drop("index",1)
del prefix9
for col in prefix17:
    if col[len(col)-10:len(col)] == "prefixlen9":
        prefix17[col] = prefix9_ref[col[0:len(col)-11]]
del prefix9_ref

prefix10 = data_dummy[data_dummy['prefix']==10]
prefix10_ref = prefix10[prefix10['case_len']>=17]
prefix10_ref = prefix10_ref.reset_index()
prefix10_ref = prefix10_ref.drop("index",1)
del prefix10
for col in prefix17:
    if col[len(col)-11:len(col)] == "prefixlen10":
        prefix17[col] = prefix10_ref[col[0:len(col)-12]]
del prefix10_ref

prefix11 = data_dummy[data_dummy['prefix']==11]
prefix11_ref = prefix11[prefix11['case_len']>=17]
prefix11_ref = prefix11_ref.reset_index()
prefix11_ref = prefix11_ref.drop("index",1)
del prefix11
for col in prefix17:
    if col[len(col)-11:len(col)] == "prefixlen11":
        prefix17[col] = prefix11_ref[col[0:len(col)-12]]
del prefix11_ref

prefix12 = data_dummy[data_dummy['prefix']==12]
prefix12_ref = prefix12[prefix12['case_len']>=17]
prefix12_ref = prefix12_ref.reset_index()
prefix12_ref = prefix12_ref.drop("index",1)
del prefix12
for col in prefix17:
    if col[len(col)-11:len(col)] == "prefixlen12":
        prefix17[col] = prefix12_ref[col[0:len(col)-12]]
del prefix12_ref

prefix13 = data_dummy[data_dummy['prefix']==13]
prefix13_ref = prefix13[prefix13['case_len']>=17]
prefix13_ref = prefix13_ref.reset_index()
prefix13_ref = prefix13_ref.drop("index",1)
del prefix13
for col in prefix17:
    if col[len(col)-11:len(col)] == "prefixlen13":
        prefix17[col] = prefix13_ref[col[0:len(col)-12]]
del prefix13_ref

prefix14 = data_dummy[data_dummy['prefix']==14]
prefix14_ref = prefix14[prefix14['case_len']>=17]
prefix14_ref = prefix14_ref.reset_index()
prefix14_ref = prefix14_ref.drop("index",1)
del prefix14
for col in prefix17:
    if col[len(col)-11:len(col)] == "prefixlen14":
        prefix17[col] = prefix14_ref[col[0:len(col)-12]]
del prefix14_ref

prefix15 = data_dummy[data_dummy['prefix']==15]
prefix15_ref = prefix15[prefix15['case_len']>=17]
prefix15_ref = prefix15_ref.reset_index()
prefix15_ref = prefix15_ref.drop("index",1)
del prefix15
for col in prefix17:
    if col[len(col)-11:len(col)] == "prefixlen15":
        prefix17[col] = prefix15_ref[col[0:len(col)-12]]
del prefix15_ref

prefix16 = data_dummy[data_dummy['prefix']==16]
del data_dummy
prefix16_ref = prefix16[prefix16['case_len']>=17]
prefix16_ref = prefix16_ref.reset_index()
prefix16_ref = prefix16_ref.drop("index",1)
del prefix16
for col in prefix17:
    if col[len(col)-11:len(col)] == "prefixlen16":
        prefix17[col] = prefix16_ref[col[0:len(col)-12]]
del prefix16_ref

for col in prefix17:
    if col[len(col)-11:len(col)] == "prefixlen17":
        prefix17[col] = prefix17[col[0:len(col)-12]]
prefix17 = prefix17.filter(regex="prefixlen")
prefix17.to_pickle('prefix17.pkl')

##########prefix18##########
data_dummy_prefix_prefixlen18 = pd.DataFrame(index=range(len(data_dummy[data_dummy['prefix']==18])), columns=range(len(data_dummy.columns)))
for col in data_dummy_columns_list:
	print(col)
	if col not in ("case_id","Action","EventOrigin","lifecycle.transition","Accepted","Selected","AMOUNT_REQ","case_len","label"):
		data_dummy_prefix_prefixlen18[str(col) + '_prefixlen' + str(1)] = 0
		data_dummy_prefix_prefixlen18[str(col) + '_prefixlen' + str(2)] = 0
		data_dummy_prefix_prefixlen18[str(col) + '_prefixlen' + str(3)] = 0
		data_dummy_prefix_prefixlen18[str(col) + '_prefixlen' + str(4)] = 0
		data_dummy_prefix_prefixlen18[str(col) + '_prefixlen' + str(5)] = 0
		data_dummy_prefix_prefixlen18[str(col) + '_prefixlen' + str(6)] = 0
		data_dummy_prefix_prefixlen18[str(col) + '_prefixlen' + str(7)] = 0
		data_dummy_prefix_prefixlen18[str(col) + '_prefixlen' + str(8)] = 0
		data_dummy_prefix_prefixlen18[str(col) + '_prefixlen' + str(9)] = 0
		data_dummy_prefix_prefixlen18[str(col) + '_prefixlen' + str(10)] = 0
		data_dummy_prefix_prefixlen18[str(col) + '_prefixlen' + str(11)] = 0
		data_dummy_prefix_prefixlen18[str(col) + '_prefixlen' + str(12)] = 0
		data_dummy_prefix_prefixlen18[str(col) + '_prefixlen' + str(13)] = 0
		data_dummy_prefix_prefixlen18[str(col) + '_prefixlen' + str(14)] = 0
		data_dummy_prefix_prefixlen18[str(col) + '_prefixlen' + str(15)] = 0
		data_dummy_prefix_prefixlen18[str(col) + '_prefixlen' + str(16)] = 0
		data_dummy_prefix_prefixlen18[str(col) + '_prefixlen' + str(17)] = 0
		data_dummy_prefix_prefixlen18[str(col) + '_prefixlen' + str(18)] = 0
		data_dummy_prefix_prefixlen18[str(col) + '_prefixlen' + str(19)] = 0
		data_dummy_prefix_prefixlen18[str(col) + '_prefixlen' + str(20)] = 0
	else:
		data_dummy_prefix_prefixlen18[str(col) + '_prefixlen' + str(1)] = 0 #For case-level attributes, only "_prefixlen1" is attached as suffix.
#Here, stay with 761 or change 761 to other number depending on the situation. This code is to delete columns which do not have names (or having numbers as names).
#Check the column names by yourself and determine the number to replace 761. If 761 is right, stick with 761.
data_dummy_prefix_prefixlen18 = data_dummy_prefix_prefixlen18.iloc[:,761:len(data_dummy_prefix_prefixlen18.columns)]
prefix18 = data_dummy[data_dummy['prefix']==18]
prefix18 = prefix18.reset_index()
prefix18 = prefix18.drop("index",1)
prefix18 = prefix18.join(data_dummy_prefix_prefixlen18)
del data_dummy_prefix_prefixlen18

prefix1 = data_dummy[data_dummy['prefix']==1]
prefix1_ref = prefix1[prefix1['case_len']>=18]
prefix1_ref = prefix1_ref.reset_index()
prefix1_ref = prefix1_ref.drop("index",1)
del prefix1
for col in prefix18:
    if col[len(col)-10:len(col)] == "prefixlen1":
        prefix18[col] = prefix1_ref[col[0:len(col)-11]]
del prefix1_ref

prefix2 = data_dummy[data_dummy['prefix']==2]
prefix2_ref = prefix2[prefix2['case_len']>=18]
prefix2_ref = prefix2_ref.reset_index()
prefix2_ref = prefix2_ref.drop("index",1)
del prefix2
for col in prefix18:
    if col[len(col)-10:len(col)] == "prefixlen2":
        prefix18[col] = prefix2_ref[col[0:len(col)-11]]
del prefix2_ref

prefix3 = data_dummy[data_dummy['prefix']==3]
prefix3_ref = prefix3[prefix3['case_len']>=18]
prefix3_ref = prefix3_ref.reset_index()
prefix3_ref = prefix3_ref.drop("index",1)
del prefix3
for col in prefix18:
    if col[len(col)-10:len(col)] == "prefixlen3":
        prefix18[col] = prefix3_ref[col[0:len(col)-11]]
del prefix3_ref

prefix4 = data_dummy[data_dummy['prefix']==4]
prefix4_ref = prefix4[prefix4['case_len']>=18]
prefix4_ref = prefix4_ref.reset_index()
prefix4_ref = prefix4_ref.drop("index",1)
del prefix4
for col in prefix18:
    if col[len(col)-10:len(col)] == "prefixlen4":
        prefix18[col] = prefix4_ref[col[0:len(col)-11]]
del prefix4_ref

prefix5 = data_dummy[data_dummy['prefix']==5]
prefix5_ref = prefix5[prefix5['case_len']>=18]
prefix5_ref = prefix5_ref.reset_index()
prefix5_ref = prefix5_ref.drop("index",1)
del prefix5
for col in prefix18:
    if col[len(col)-10:len(col)] == "prefixlen5":
        prefix18[col] = prefix5_ref[col[0:len(col)-11]]
del prefix5_ref

prefix6 = data_dummy[data_dummy['prefix']==6]
prefix6_ref = prefix6[prefix6['case_len']>=18]
prefix6_ref = prefix6_ref.reset_index()
prefix6_ref = prefix6_ref.drop("index",1)
del prefix6
for col in prefix18:
    if col[len(col)-10:len(col)] == "prefixlen6":
        prefix18[col] = prefix6_ref[col[0:len(col)-11]]
del prefix6_ref

prefix7 = data_dummy[data_dummy['prefix']==7]
prefix7_ref = prefix7[prefix7['case_len']>=18]
prefix7_ref = prefix7_ref.reset_index()
prefix7_ref = prefix7_ref.drop("index",1)
del prefix7
for col in prefix18:
    if col[len(col)-10:len(col)] == "prefixlen7":
        prefix18[col] = prefix7_ref[col[0:len(col)-11]]
del prefix7_ref

prefix8 = data_dummy[data_dummy['prefix']==8]
prefix8_ref = prefix8[prefix8['case_len']>=18]
prefix8_ref = prefix8_ref.reset_index()
prefix8_ref = prefix8_ref.drop("index",1)
del prefix8
for col in prefix18:
    if col[len(col)-10:len(col)] == "prefixlen8":
        prefix18[col] = prefix8_ref[col[0:len(col)-11]]
del prefix8_ref

prefix9 = data_dummy[data_dummy['prefix']==9]
prefix9_ref = prefix9[prefix9['case_len']>=18]
prefix9_ref = prefix9_ref.reset_index()
prefix9_ref = prefix9_ref.drop("index",1)
del prefix9
for col in prefix18:
    if col[len(col)-10:len(col)] == "prefixlen9":
        prefix18[col] = prefix9_ref[col[0:len(col)-11]]
del prefix9_ref

prefix10 = data_dummy[data_dummy['prefix']==10]
prefix10_ref = prefix10[prefix10['case_len']>=18]
prefix10_ref = prefix10_ref.reset_index()
prefix10_ref = prefix10_ref.drop("index",1)
del prefix10
for col in prefix18:
    if col[len(col)-11:len(col)] == "prefixlen10":
        prefix18[col] = prefix10_ref[col[0:len(col)-12]]
del prefix10_ref

prefix11 = data_dummy[data_dummy['prefix']==11]
prefix11_ref = prefix11[prefix11['case_len']>=18]
prefix11_ref = prefix11_ref.reset_index()
prefix11_ref = prefix11_ref.drop("index",1)
del prefix11
for col in prefix18:
    if col[len(col)-11:len(col)] == "prefixlen11":
        prefix18[col] = prefix11_ref[col[0:len(col)-12]]
del prefix11_ref

prefix12 = data_dummy[data_dummy['prefix']==12]
prefix12_ref = prefix12[prefix12['case_len']>=18]
prefix12_ref = prefix12_ref.reset_index()
prefix12_ref = prefix12_ref.drop("index",1)
del prefix12
for col in prefix18:
    if col[len(col)-11:len(col)] == "prefixlen12":
        prefix18[col] = prefix12_ref[col[0:len(col)-12]]
del prefix12_ref

prefix13 = data_dummy[data_dummy['prefix']==13]
prefix13_ref = prefix13[prefix13['case_len']>=18]
prefix13_ref = prefix13_ref.reset_index()
prefix13_ref = prefix13_ref.drop("index",1)
del prefix13
for col in prefix18:
    if col[len(col)-11:len(col)] == "prefixlen13":
        prefix18[col] = prefix13_ref[col[0:len(col)-12]]
del prefix13_ref

prefix14 = data_dummy[data_dummy['prefix']==14]
prefix14_ref = prefix14[prefix14['case_len']>=18]
prefix14_ref = prefix14_ref.reset_index()
prefix14_ref = prefix14_ref.drop("index",1)
del prefix14
for col in prefix18:
    if col[len(col)-11:len(col)] == "prefixlen14":
        prefix18[col] = prefix14_ref[col[0:len(col)-12]]
del prefix14_ref

prefix15 = data_dummy[data_dummy['prefix']==15]
prefix15_ref = prefix15[prefix15['case_len']>=18]
prefix15_ref = prefix15_ref.reset_index()
prefix15_ref = prefix15_ref.drop("index",1)
del prefix15
for col in prefix18:
    if col[len(col)-11:len(col)] == "prefixlen15":
        prefix18[col] = prefix15_ref[col[0:len(col)-12]]
del prefix15_ref

prefix16 = data_dummy[data_dummy['prefix']==16]
prefix16_ref = prefix16[prefix16['case_len']>=18]
prefix16_ref = prefix16_ref.reset_index()
prefix16_ref = prefix16_ref.drop("index",1)
del prefix16
for col in prefix18:
    if col[len(col)-11:len(col)] == "prefixlen16":
        prefix18[col] = prefix16_ref[col[0:len(col)-12]]
del prefix16_ref

prefix17 = data_dummy[data_dummy['prefix']==17]
del data_dummy
prefix17_ref = prefix17[prefix17['case_len']>=18]
prefix17_ref = prefix17_ref.reset_index()
prefix17_ref = prefix17_ref.drop("index",1)
del prefix17
for col in prefix18:
    if col[len(col)-11:len(col)] == "prefixlen17":
        prefix18[col] = prefix17_ref[col[0:len(col)-12]]
del prefix17_ref

for col in prefix18:
    if col[len(col)-11:len(col)] == "prefixlen18":
        prefix18[col] = prefix18[col[0:len(col)-12]]
prefix18 = prefix18.filter(regex="prefixlen")
prefix18.to_pickle('prefix18.pkl')

##########prefix19##########
data_dummy_prefix_prefixlen19 = pd.DataFrame(index=range(len(data_dummy[data_dummy['prefix']==19])), columns=range(len(data_dummy.columns)))
for col in data_dummy_columns_list:
	print(col)
	if col not in ("case_id","Action","EventOrigin","lifecycle.transition","Accepted","Selected","AMOUNT_REQ","case_len","label"):
		data_dummy_prefix_prefixlen19[str(col) + '_prefixlen' + str(1)] = 0
		data_dummy_prefix_prefixlen19[str(col) + '_prefixlen' + str(2)] = 0
		data_dummy_prefix_prefixlen19[str(col) + '_prefixlen' + str(3)] = 0
		data_dummy_prefix_prefixlen19[str(col) + '_prefixlen' + str(4)] = 0
		data_dummy_prefix_prefixlen19[str(col) + '_prefixlen' + str(5)] = 0
		data_dummy_prefix_prefixlen19[str(col) + '_prefixlen' + str(6)] = 0
		data_dummy_prefix_prefixlen19[str(col) + '_prefixlen' + str(7)] = 0
		data_dummy_prefix_prefixlen19[str(col) + '_prefixlen' + str(8)] = 0
		data_dummy_prefix_prefixlen19[str(col) + '_prefixlen' + str(9)] = 0
		data_dummy_prefix_prefixlen19[str(col) + '_prefixlen' + str(10)] = 0
		data_dummy_prefix_prefixlen19[str(col) + '_prefixlen' + str(11)] = 0
		data_dummy_prefix_prefixlen19[str(col) + '_prefixlen' + str(12)] = 0
		data_dummy_prefix_prefixlen19[str(col) + '_prefixlen' + str(13)] = 0
		data_dummy_prefix_prefixlen19[str(col) + '_prefixlen' + str(14)] = 0
		data_dummy_prefix_prefixlen19[str(col) + '_prefixlen' + str(15)] = 0
		data_dummy_prefix_prefixlen19[str(col) + '_prefixlen' + str(16)] = 0
		data_dummy_prefix_prefixlen19[str(col) + '_prefixlen' + str(17)] = 0
		data_dummy_prefix_prefixlen19[str(col) + '_prefixlen' + str(18)] = 0
		data_dummy_prefix_prefixlen19[str(col) + '_prefixlen' + str(19)] = 0
		data_dummy_prefix_prefixlen19[str(col) + '_prefixlen' + str(20)] = 0
	else:
		data_dummy_prefix_prefixlen19[str(col) + '_prefixlen' + str(1)] = 0 #For case-level attributes, only "_prefixlen1" is attached as suffix.
#Here, stay with 761 or change 761 to other number depending on the situation. This code is to delete columns which do not have names (or having numbers as names).
#Check the column names by yourself and determine the number to replace 761. If 761 is right, stick with 761.
data_dummy_prefix_prefixlen19 = data_dummy_prefix_prefixlen19.iloc[:,761:len(data_dummy_prefix_prefixlen19.columns)]
prefix19 = data_dummy[data_dummy['prefix']==19]
prefix19 = prefix19.reset_index()
prefix19 = prefix19.drop("index",1)
prefix19 = prefix19.join(data_dummy_prefix_prefixlen19)
del data_dummy_prefix_prefixlen19

prefix1 = data_dummy[data_dummy['prefix']==1]
prefix1_ref = prefix1[prefix1['case_len']>=19]
prefix1_ref = prefix1_ref.reset_index()
prefix1_ref = prefix1_ref.drop("index",1)
del prefix1
for col in prefix19:
    if col[len(col)-10:len(col)] == "prefixlen1":
        prefix19[col] = prefix1_ref[col[0:len(col)-11]]
del prefix1_ref

prefix2 = data_dummy[data_dummy['prefix']==2]
prefix2_ref = prefix2[prefix2['case_len']>=19]
prefix2_ref = prefix2_ref.reset_index()
prefix2_ref = prefix2_ref.drop("index",1)
del prefix2
for col in prefix19:
    if col[len(col)-10:len(col)] == "prefixlen2":
        prefix19[col] = prefix2_ref[col[0:len(col)-11]]
del prefix2_ref

prefix3 = data_dummy[data_dummy['prefix']==3]
prefix3_ref = prefix3[prefix3['case_len']>=19]
prefix3_ref = prefix3_ref.reset_index()
prefix3_ref = prefix3_ref.drop("index",1)
del prefix3
for col in prefix19:
    if col[len(col)-10:len(col)] == "prefixlen3":
        prefix19[col] = prefix3_ref[col[0:len(col)-11]]
del prefix3_ref

prefix4 = data_dummy[data_dummy['prefix']==4]
prefix4_ref = prefix4[prefix4['case_len']>=19]
prefix4_ref = prefix4_ref.reset_index()
prefix4_ref = prefix4_ref.drop("index",1)
del prefix4
for col in prefix19:
    if col[len(col)-10:len(col)] == "prefixlen4":
        prefix19[col] = prefix4_ref[col[0:len(col)-11]]
del prefix4_ref

prefix5 = data_dummy[data_dummy['prefix']==5]
prefix5_ref = prefix5[prefix5['case_len']>=19]
prefix5_ref = prefix5_ref.reset_index()
prefix5_ref = prefix5_ref.drop("index",1)
del prefix5
for col in prefix19:
    if col[len(col)-10:len(col)] == "prefixlen5":
        prefix19[col] = prefix5_ref[col[0:len(col)-11]]
del prefix5_ref

prefix6 = data_dummy[data_dummy['prefix']==6]
prefix6_ref = prefix6[prefix6['case_len']>=19]
prefix6_ref = prefix6_ref.reset_index()
prefix6_ref = prefix6_ref.drop("index",1)
del prefix6
for col in prefix19:
    if col[len(col)-10:len(col)] == "prefixlen6":
        prefix19[col] = prefix6_ref[col[0:len(col)-11]]
del prefix6_ref

prefix7 = data_dummy[data_dummy['prefix']==7]
prefix7_ref = prefix7[prefix7['case_len']>=19]
prefix7_ref = prefix7_ref.reset_index()
prefix7_ref = prefix7_ref.drop("index",1)
del prefix7
for col in prefix19:
    if col[len(col)-10:len(col)] == "prefixlen7":
        prefix19[col] = prefix7_ref[col[0:len(col)-11]]
del prefix7_ref

prefix8 = data_dummy[data_dummy['prefix']==8]
prefix8_ref = prefix8[prefix8['case_len']>=19]
prefix8_ref = prefix8_ref.reset_index()
prefix8_ref = prefix8_ref.drop("index",1)
del prefix8
for col in prefix19:
    if col[len(col)-10:len(col)] == "prefixlen8":
        prefix19[col] = prefix8_ref[col[0:len(col)-11]]
del prefix8_ref

prefix9 = data_dummy[data_dummy['prefix']==9]
prefix9_ref = prefix9[prefix9['case_len']>=19]
prefix9_ref = prefix9_ref.reset_index()
prefix9_ref = prefix9_ref.drop("index",1)
del prefix9
for col in prefix19:
    if col[len(col)-10:len(col)] == "prefixlen9":
        prefix19[col] = prefix9_ref[col[0:len(col)-11]]
del prefix9_ref

prefix10 = data_dummy[data_dummy['prefix']==10]
prefix10_ref = prefix10[prefix10['case_len']>=19]
prefix10_ref = prefix10_ref.reset_index()
prefix10_ref = prefix10_ref.drop("index",1)
del prefix10
for col in prefix19:
    if col[len(col)-11:len(col)] == "prefixlen10":
        prefix19[col] = prefix10_ref[col[0:len(col)-12]]
del prefix10_ref

prefix11 = data_dummy[data_dummy['prefix']==11]
prefix11_ref = prefix11[prefix11['case_len']>=19]
prefix11_ref = prefix11_ref.reset_index()
prefix11_ref = prefix11_ref.drop("index",1)
del prefix11
for col in prefix19:
    if col[len(col)-11:len(col)] == "prefixlen11":
        prefix19[col] = prefix11_ref[col[0:len(col)-12]]
del prefix11_ref

prefix12 = data_dummy[data_dummy['prefix']==12]
prefix12_ref = prefix12[prefix12['case_len']>=19]
prefix12_ref = prefix12_ref.reset_index()
prefix12_ref = prefix12_ref.drop("index",1)
del prefix12
for col in prefix19:
    if col[len(col)-11:len(col)] == "prefixlen12":
        prefix19[col] = prefix12_ref[col[0:len(col)-12]]
del prefix12_ref

prefix13 = data_dummy[data_dummy['prefix']==13]
prefix13_ref = prefix13[prefix13['case_len']>=19]
prefix13_ref = prefix13_ref.reset_index()
prefix13_ref = prefix13_ref.drop("index",1)
del prefix13
for col in prefix19:
    if col[len(col)-11:len(col)] == "prefixlen13":
        prefix19[col] = prefix13_ref[col[0:len(col)-12]]
del prefix13_ref

prefix14 = data_dummy[data_dummy['prefix']==14]
prefix14_ref = prefix14[prefix14['case_len']>=19]
prefix14_ref = prefix14_ref.reset_index()
prefix14_ref = prefix14_ref.drop("index",1)
del prefix14
for col in prefix19:
    if col[len(col)-11:len(col)] == "prefixlen14":
        prefix19[col] = prefix14_ref[col[0:len(col)-12]]
del prefix14_ref

prefix15 = data_dummy[data_dummy['prefix']==15]
prefix15_ref = prefix15[prefix15['case_len']>=19]
prefix15_ref = prefix15_ref.reset_index()
prefix15_ref = prefix15_ref.drop("index",1)
del prefix15
for col in prefix19:
    if col[len(col)-11:len(col)] == "prefixlen15":
        prefix19[col] = prefix15_ref[col[0:len(col)-12]]
del prefix15_ref

prefix16 = data_dummy[data_dummy['prefix']==16]
prefix16_ref = prefix16[prefix16['case_len']>=19]
prefix16_ref = prefix16_ref.reset_index()
prefix16_ref = prefix16_ref.drop("index",1)
del prefix16
for col in prefix19:
    if col[len(col)-11:len(col)] == "prefixlen16":
        prefix19[col] = prefix16_ref[col[0:len(col)-12]]
del prefix16_ref

prefix17 = data_dummy[data_dummy['prefix']==17]
prefix17_ref = prefix17[prefix17['case_len']>=19]
prefix17_ref = prefix17_ref.reset_index()
prefix17_ref = prefix17_ref.drop("index",1)
del prefix17
for col in prefix19:
    if col[len(col)-11:len(col)] == "prefixlen17":
        prefix19[col] = prefix17_ref[col[0:len(col)-12]]
del prefix17_ref

prefix18 = data_dummy[data_dummy['prefix']==18]
del data_dummy
prefix18_ref = prefix18[prefix18['case_len']>=19]
prefix18_ref = prefix18_ref.reset_index()
prefix18_ref = prefix18_ref.drop("index",1)
del prefix18
for col in prefix19:
    if col[len(col)-11:len(col)] == "prefixlen18":
        prefix19[col] = prefix18_ref[col[0:len(col)-12]]
del prefix18_ref

for col in prefix19:
    if col[len(col)-11:len(col)] == "prefixlen19":
        prefix19[col] = prefix19[col[0:len(col)-12]]
prefix19 = prefix19.filter(regex="prefixlen")
prefix19.to_pickle('prefix19.pkl')

##########prefix20##########
data_dummy_prefix_prefixlen20 = pd.DataFrame(index=range(len(data_dummy[data_dummy['prefix']==20])), columns=range(len(data_dummy.columns)))
for col in data_dummy_columns_list:
	print(col)
	if col not in ("case_id","Action","EventOrigin","lifecycle.transition","Accepted","Selected","AMOUNT_REQ","case_len","label"):
		data_dummy_prefix_prefixlen20[str(col) + '_prefixlen' + str(1)] = 0
		data_dummy_prefix_prefixlen20[str(col) + '_prefixlen' + str(2)] = 0
		data_dummy_prefix_prefixlen20[str(col) + '_prefixlen' + str(3)] = 0
		data_dummy_prefix_prefixlen20[str(col) + '_prefixlen' + str(4)] = 0
		data_dummy_prefix_prefixlen20[str(col) + '_prefixlen' + str(5)] = 0
		data_dummy_prefix_prefixlen20[str(col) + '_prefixlen' + str(6)] = 0
		data_dummy_prefix_prefixlen20[str(col) + '_prefixlen' + str(7)] = 0
		data_dummy_prefix_prefixlen20[str(col) + '_prefixlen' + str(8)] = 0
		data_dummy_prefix_prefixlen20[str(col) + '_prefixlen' + str(9)] = 0
		data_dummy_prefix_prefixlen20[str(col) + '_prefixlen' + str(10)] = 0
		data_dummy_prefix_prefixlen20[str(col) + '_prefixlen' + str(11)] = 0
		data_dummy_prefix_prefixlen20[str(col) + '_prefixlen' + str(12)] = 0
		data_dummy_prefix_prefixlen20[str(col) + '_prefixlen' + str(13)] = 0
		data_dummy_prefix_prefixlen20[str(col) + '_prefixlen' + str(14)] = 0
		data_dummy_prefix_prefixlen20[str(col) + '_prefixlen' + str(15)] = 0
		data_dummy_prefix_prefixlen20[str(col) + '_prefixlen' + str(16)] = 0
		data_dummy_prefix_prefixlen20[str(col) + '_prefixlen' + str(17)] = 0
		data_dummy_prefix_prefixlen20[str(col) + '_prefixlen' + str(18)] = 0
		data_dummy_prefix_prefixlen20[str(col) + '_prefixlen' + str(19)] = 0
		data_dummy_prefix_prefixlen20[str(col) + '_prefixlen' + str(20)] = 0
	else:
		data_dummy_prefix_prefixlen20[str(col) + '_prefixlen' + str(1)] = 0 #For case-level attributes, only "_prefixlen1" is attached as suffix.
#Here, stay with 761 or change 761 to other number depending on the situation. This code is to delete columns which do not have names (or having numbers as names).
#Check the column names by yourself and determine the number to replace 761. If 761 is right, stick with 761.
data_dummy_prefix_prefixlen20 = data_dummy_prefix_prefixlen20.iloc[:,761:len(data_dummy_prefix_prefixlen20.columns)]
prefix20 = data_dummy[data_dummy['prefix']==20]
prefix20 = prefix20.reset_index()
prefix20 = prefix20.drop("index",1)
prefix20 = prefix20.join(data_dummy_prefix_prefixlen20)
del data_dummy_prefix_prefixlen20

prefix1 = data_dummy[data_dummy['prefix']==1]
prefix1_ref = prefix1[prefix1['case_len']>=20]
prefix1_ref = prefix1_ref.reset_index()
prefix1_ref = prefix1_ref.drop("index",1)
del prefix1
for col in prefix20:
    if col[len(col)-10:len(col)] == "prefixlen1":
        prefix20[col] = prefix1_ref[col[0:len(col)-11]]
del prefix1_ref

prefix2 = data_dummy[data_dummy['prefix']==2]
prefix2_ref = prefix2[prefix2['case_len']>=20]
prefix2_ref = prefix2_ref.reset_index()
prefix2_ref = prefix2_ref.drop("index",1)
del prefix2
for col in prefix20:
    if col[len(col)-10:len(col)] == "prefixlen2":
        prefix20[col] = prefix2_ref[col[0:len(col)-11]]
del prefix2_ref

prefix3 = data_dummy[data_dummy['prefix']==3]
prefix3_ref = prefix3[prefix3['case_len']>=20]
prefix3_ref = prefix3_ref.reset_index()
prefix3_ref = prefix3_ref.drop("index",1)
del prefix3
for col in prefix20:
    if col[len(col)-10:len(col)] == "prefixlen3":
        prefix20[col] = prefix3_ref[col[0:len(col)-11]]
del prefix3_ref

prefix4 = data_dummy[data_dummy['prefix']==4]
prefix4_ref = prefix4[prefix4['case_len']>=20]
prefix4_ref = prefix4_ref.reset_index()
prefix4_ref = prefix4_ref.drop("index",1)
del prefix4
for col in prefix20:
    if col[len(col)-10:len(col)] == "prefixlen4":
        prefix20[col] = prefix4_ref[col[0:len(col)-11]]
del prefix4_ref

prefix5 = data_dummy[data_dummy['prefix']==5]
prefix5_ref = prefix5[prefix5['case_len']>=20]
prefix5_ref = prefix5_ref.reset_index()
prefix5_ref = prefix5_ref.drop("index",1)
del prefix5
for col in prefix20:
    if col[len(col)-10:len(col)] == "prefixlen5":
        prefix20[col] = prefix5_ref[col[0:len(col)-11]]
del prefix5_ref

prefix6 = data_dummy[data_dummy['prefix']==6]
prefix6_ref = prefix6[prefix6['case_len']>=20]
prefix6_ref = prefix6_ref.reset_index()
prefix6_ref = prefix6_ref.drop("index",1)
del prefix6
for col in prefix20:
    if col[len(col)-10:len(col)] == "prefixlen6":
        prefix20[col] = prefix6_ref[col[0:len(col)-11]]
del prefix6_ref

prefix7 = data_dummy[data_dummy['prefix']==7]
prefix7_ref = prefix7[prefix7['case_len']>=20]
prefix7_ref = prefix7_ref.reset_index()
prefix7_ref = prefix7_ref.drop("index",1)
del prefix7
for col in prefix20:
    if col[len(col)-10:len(col)] == "prefixlen7":
        prefix20[col] = prefix7_ref[col[0:len(col)-11]]
del prefix7_ref

prefix8 = data_dummy[data_dummy['prefix']==8]
prefix8_ref = prefix8[prefix8['case_len']>=20]
prefix8_ref = prefix8_ref.reset_index()
prefix8_ref = prefix8_ref.drop("index",1)
del prefix8
for col in prefix20:
    if col[len(col)-10:len(col)] == "prefixlen8":
        prefix20[col] = prefix8_ref[col[0:len(col)-11]]
del prefix8_ref

prefix9 = data_dummy[data_dummy['prefix']==9]
prefix9_ref = prefix9[prefix9['case_len']>=20]
prefix9_ref = prefix9_ref.reset_index()
prefix9_ref = prefix9_ref.drop("index",1)
del prefix9
for col in prefix20:
    if col[len(col)-10:len(col)] == "prefixlen9":
        prefix20[col] = prefix9_ref[col[0:len(col)-11]]
del prefix9_ref

prefix10 = data_dummy[data_dummy['prefix']==10]
prefix10_ref = prefix10[prefix10['case_len']>=20]
prefix10_ref = prefix10_ref.reset_index()
prefix10_ref = prefix10_ref.drop("index",1)
del prefix10
for col in prefix20:
    if col[len(col)-11:len(col)] == "prefixlen10":
        prefix20[col] = prefix10_ref[col[0:len(col)-12]]
del prefix10_ref

prefix11 = data_dummy[data_dummy['prefix']==11]
prefix11_ref = prefix11[prefix11['case_len']>=20]
prefix11_ref = prefix11_ref.reset_index()
prefix11_ref = prefix11_ref.drop("index",1)
del prefix11
for col in prefix20:
    if col[len(col)-11:len(col)] == "prefixlen11":
        prefix20[col] = prefix11_ref[col[0:len(col)-12]]
del prefix11_ref

prefix12 = data_dummy[data_dummy['prefix']==12]
prefix12_ref = prefix12[prefix12['case_len']>=20]
prefix12_ref = prefix12_ref.reset_index()
prefix12_ref = prefix12_ref.drop("index",1)
del prefix12
for col in prefix20:
    if col[len(col)-11:len(col)] == "prefixlen12":
        prefix20[col] = prefix12_ref[col[0:len(col)-12]]
del prefix12_ref

prefix13 = data_dummy[data_dummy['prefix']==13]
prefix13_ref = prefix13[prefix13['case_len']>=20]
prefix13_ref = prefix13_ref.reset_index()
prefix13_ref = prefix13_ref.drop("index",1)
del prefix13
for col in prefix20:
    if col[len(col)-11:len(col)] == "prefixlen13":
        prefix20[col] = prefix13_ref[col[0:len(col)-12]]
del prefix13_ref

prefix14 = data_dummy[data_dummy['prefix']==14]
prefix14_ref = prefix14[prefix14['case_len']>=20]
prefix14_ref = prefix14_ref.reset_index()
prefix14_ref = prefix14_ref.drop("index",1)
del prefix14
for col in prefix20:
    if col[len(col)-11:len(col)] == "prefixlen14":
        prefix20[col] = prefix14_ref[col[0:len(col)-12]]
del prefix14_ref

prefix15 = data_dummy[data_dummy['prefix']==15]
prefix15_ref = prefix15[prefix15['case_len']>=20]
prefix15_ref = prefix15_ref.reset_index()
prefix15_ref = prefix15_ref.drop("index",1)
del prefix15
for col in prefix20:
    if col[len(col)-11:len(col)] == "prefixlen15":
        prefix20[col] = prefix15_ref[col[0:len(col)-12]]
del prefix15_ref

prefix16 = data_dummy[data_dummy['prefix']==16]
prefix16_ref = prefix16[prefix16['case_len']>=20]
prefix16_ref = prefix16_ref.reset_index()
prefix16_ref = prefix16_ref.drop("index",1)
del prefix16
for col in prefix20:
    if col[len(col)-11:len(col)] == "prefixlen16":
        prefix20[col] = prefix16_ref[col[0:len(col)-12]]
del prefix16_ref

prefix17 = data_dummy[data_dummy['prefix']==17]
prefix17_ref = prefix17[prefix17['case_len']>=20]
prefix17_ref = prefix17_ref.reset_index()
prefix17_ref = prefix17_ref.drop("index",1)
del prefix17
for col in prefix20:
    if col[len(col)-11:len(col)] == "prefixlen17":
        prefix20[col] = prefix17_ref[col[0:len(col)-12]]
del prefix17_ref

prefix18 = data_dummy[data_dummy['prefix']==18]
prefix18_ref = prefix18[prefix18['case_len']>=20]
prefix18_ref = prefix18_ref.reset_index()
prefix18_ref = prefix18_ref.drop("index",1)
del prefix18
for col in prefix20:
    if col[len(col)-11:len(col)] == "prefixlen18":
        prefix20[col] = prefix18_ref[col[0:len(col)-12]]
del prefix18_ref

prefix19 = data_dummy[data_dummy['prefix']==19]
del data_dummy
prefix19_ref = prefix19[prefix19['case_len']>=20]
prefix19_ref = prefix19_ref.reset_index()
prefix19_ref = prefix19_ref.drop("index",1)
del prefix19
for col in prefix20:
    if col[len(col)-11:len(col)] == "prefixlen19":
        prefix20[col] = prefix19_ref[col[0:len(col)-12]]
del prefix19_ref

for col in prefix20:
    if col[len(col)-11:len(col)] == "prefixlen20":
        prefix20[col] = prefix20[col[0:len(col)-12]]
prefix20 = prefix20.filter(regex="prefixlen")

prefix20.to_pickle('prefix20.pkl')
