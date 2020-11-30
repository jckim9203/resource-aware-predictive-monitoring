import pandas as pd

def variable_setting(dataset_name, prefix_index_encoding):
    if dataset_name == "BPIC11_f1_exp_prefix":
        data_dummy_prefix_yes_resource = prefix_index_encoding
        prefix_index_encoding_case_id = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("case_id")]
        prefix_index_encoding_activity = prefix_index_encoding.iloc[:,
                                         prefix_index_encoding.columns.str.contains("Activity.code")]
        prefix_index_encoding_producer = prefix_index_encoding.iloc[:,
                                         prefix_index_encoding.columns.str.contains("Producer.code")]
        prefix_index_encoding_timesincelastevent = prefix_index_encoding.iloc[:,
                                                   prefix_index_encoding.columns.str.contains("timesincelastevent")]
        prefix_index_encoding_timesincecasestart = prefix_index_encoding.iloc[:,
                                                   prefix_index_encoding.columns.str.contains("timesincecasestart")]
        prefix_index_encoding_timesincemidnight = prefix_index_encoding.iloc[:,
                                                  prefix_index_encoding.columns.str.contains("timesincemidnight")]
        prefix_index_encoding_event_nr = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains(
            "event_nr")]  # 'prefix' column and 'event_nr' column have same values here
        prefix_index_encoding_month = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("month")]
        prefix_index_encoding_weekday = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("weekday")]
        prefix_index_encoding_hour = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("hour")]
        prefix_index_encoding_open_cases = prefix_index_encoding.iloc[:,
                                           prefix_index_encoding.columns.str.contains("open_cases")]
        prefix_index_encoding_label = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains(
            "label_prefixlen1")]  # Should be "label_prefixlen1" all the time
        prefix_index_encoding_Age = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("Age")]
        prefix_index_encoding_Treatment_code = prefix_index_encoding.iloc[:,
                                               prefix_index_encoding.columns.str.contains("Treatment.code")]
        prefix_index_encoding_Diagnosis_code = prefix_index_encoding.iloc[:,
                                               prefix_index_encoding.columns.str.contains("Diagnosis.code")]
        prefix_index_encoding_Specialism_code_1 = prefix_index_encoding.iloc[:,
                                                  prefix_index_encoding.columns.str.contains("Specialism.code.1")]
        prefix_index_encoding_Section = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("Section")]
        prefix_index_encoding_Specialism_code_2 = prefix_index_encoding.iloc[:,
                                                  prefix_index_encoding.columns.str.contains("Specialism.code.2")]
        prefix_index_encoding_group = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("group")]
        prefix_index_encoding_num_of_exec = prefix_index_encoding.iloc[:,
                                            prefix_index_encoding.columns.str.contains("Number.of.executions")]
        data_dummy_prefix_no_resource = pd.concat(
            [prefix_index_encoding_activity, prefix_index_encoding_producer,
             prefix_index_encoding_timesincelastevent, prefix_index_encoding_timesincecasestart,
             prefix_index_encoding_timesincemidnight, prefix_index_encoding_event_nr, prefix_index_encoding_month,
             prefix_index_encoding_weekday, prefix_index_encoding_hour, prefix_index_encoding_open_cases,
             prefix_index_encoding_label, prefix_index_encoding_Age, prefix_index_encoding_Treatment_code,
             prefix_index_encoding_Section, prefix_index_encoding_Specialism_code_1,
             prefix_index_encoding_Specialism_code_2, prefix_index_encoding_group, prefix_index_encoding_num_of_exec],
            axis=1)
    
    if dataset_name == "BPIC11_f2_exp_prefix":
        data_dummy_prefix_yes_resource = prefix_index_encoding
        prefix_index_encoding_case_id = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("case_id")]
        prefix_index_encoding_activity = prefix_index_encoding.iloc[:,
                                         prefix_index_encoding.columns.str.contains("Activity.code")]
        prefix_index_encoding_producer = prefix_index_encoding.iloc[:,
                                         prefix_index_encoding.columns.str.contains("Producer.code")]
        prefix_index_encoding_timesincelastevent = prefix_index_encoding.iloc[:,
                                                   prefix_index_encoding.columns.str.contains("timesincelastevent")]
        prefix_index_encoding_timesincecasestart = prefix_index_encoding.iloc[:,
                                                   prefix_index_encoding.columns.str.contains("timesincecasestart")]
        prefix_index_encoding_timesincemidnight = prefix_index_encoding.iloc[:,
                                                  prefix_index_encoding.columns.str.contains("timesincemidnight")]
        prefix_index_encoding_event_nr = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains(
            "event_nr")]  # 'prefix' column and 'event_nr' column have same values here
        prefix_index_encoding_month = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("month")]
        prefix_index_encoding_weekday = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("weekday")]
        prefix_index_encoding_hour = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("hour")]
        prefix_index_encoding_open_cases = prefix_index_encoding.iloc[:,
                                           prefix_index_encoding.columns.str.contains("open_cases")]
        prefix_index_encoding_label = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains(
            "label_prefixlen1")]  # Should be "label_prefixlen1" all the time
        prefix_index_encoding_Age = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("Age")]
        prefix_index_encoding_Treatment_code = prefix_index_encoding.iloc[:,
                                               prefix_index_encoding.columns.str.contains("Treatment.code")]
        prefix_index_encoding_Diagnosis_code = prefix_index_encoding.iloc[:,
                                               prefix_index_encoding.columns.str.contains("Diagnosis.code")]
        prefix_index_encoding_Specialism_code_1 = prefix_index_encoding.iloc[:,
                                                  prefix_index_encoding.columns.str.contains("Specialism.code.1")]
        prefix_index_encoding_Section = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("Section")]
        prefix_index_encoding_Specialism_code_2 = prefix_index_encoding.iloc[:,
                                                  prefix_index_encoding.columns.str.contains("Specialism.code.2")]
        prefix_index_encoding_group = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("group")]
        prefix_index_encoding_num_of_exec = prefix_index_encoding.iloc[:,
                                            prefix_index_encoding.columns.str.contains("Number.of.executions")]
        data_dummy_prefix_no_resource = pd.concat(
            [prefix_index_encoding_activity, prefix_index_encoding_producer,
             prefix_index_encoding_timesincelastevent, prefix_index_encoding_timesincecasestart,
             prefix_index_encoding_timesincemidnight, prefix_index_encoding_event_nr, prefix_index_encoding_month,
             prefix_index_encoding_weekday, prefix_index_encoding_hour, prefix_index_encoding_open_cases,
             prefix_index_encoding_label, prefix_index_encoding_Age, prefix_index_encoding_Treatment_code,
             prefix_index_encoding_Section, prefix_index_encoding_Specialism_code_1,
             prefix_index_encoding_Specialism_code_2, prefix_index_encoding_group, prefix_index_encoding_num_of_exec],
            axis=1)
    
    if dataset_name == "BPIC11_f3_exp_prefix":
        data_dummy_prefix_yes_resource = prefix_index_encoding
        prefix_index_encoding_case_id = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("case_id")]
        prefix_index_encoding_activity = prefix_index_encoding.iloc[:,
                                         prefix_index_encoding.columns.str.contains("Activity.code")]
        prefix_index_encoding_producer = prefix_index_encoding.iloc[:,
                                         prefix_index_encoding.columns.str.contains("Producer.code")]
        prefix_index_encoding_timesincelastevent = prefix_index_encoding.iloc[:,
                                                   prefix_index_encoding.columns.str.contains("timesincelastevent")]
        prefix_index_encoding_timesincecasestart = prefix_index_encoding.iloc[:,
                                                   prefix_index_encoding.columns.str.contains("timesincecasestart")]
        prefix_index_encoding_timesincemidnight = prefix_index_encoding.iloc[:,
                                                  prefix_index_encoding.columns.str.contains("timesincemidnight")]
        prefix_index_encoding_event_nr = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains(
            "event_nr")]  # 'prefix' column and 'event_nr' column have same values here
        prefix_index_encoding_month = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("month")]
        prefix_index_encoding_weekday = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("weekday")]
        prefix_index_encoding_hour = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("hour")]
        prefix_index_encoding_open_cases = prefix_index_encoding.iloc[:,
                                           prefix_index_encoding.columns.str.contains("open_cases")]
        prefix_index_encoding_label = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains(
            "label_prefixlen1")]  # Should be "label_prefixlen1" all the time
        prefix_index_encoding_Age = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("Age")]
        prefix_index_encoding_Treatment_code = prefix_index_encoding.iloc[:,
                                               prefix_index_encoding.columns.str.contains("Treatment.code")]
        prefix_index_encoding_Diagnosis_code = prefix_index_encoding.iloc[:,
                                               prefix_index_encoding.columns.str.contains("Diagnosis.code")]
        prefix_index_encoding_Specialism_code_1 = prefix_index_encoding.iloc[:,
                                                  prefix_index_encoding.columns.str.contains("Specialism.code.1")]
        prefix_index_encoding_Section = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("Section")]
        prefix_index_encoding_Specialism_code_2 = prefix_index_encoding.iloc[:,
                                                  prefix_index_encoding.columns.str.contains("Specialism.code.2")]
        prefix_index_encoding_group = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("group")]
        prefix_index_encoding_num_of_exec = prefix_index_encoding.iloc[:,
                                            prefix_index_encoding.columns.str.contains("Number.of.executions")]
        data_dummy_prefix_no_resource = pd.concat(
            [prefix_index_encoding_activity, prefix_index_encoding_producer,
             prefix_index_encoding_timesincelastevent, prefix_index_encoding_timesincecasestart,
             prefix_index_encoding_timesincemidnight, prefix_index_encoding_event_nr, prefix_index_encoding_month,
             prefix_index_encoding_weekday, prefix_index_encoding_hour, prefix_index_encoding_open_cases,
             prefix_index_encoding_label, prefix_index_encoding_Age, prefix_index_encoding_Treatment_code,
             prefix_index_encoding_Section, prefix_index_encoding_Specialism_code_1,
             prefix_index_encoding_Specialism_code_2, prefix_index_encoding_group, prefix_index_encoding_num_of_exec],
            axis=1)
    
    if dataset_name == "BPIC11_f4_exp_prefix":
        data_dummy_prefix_yes_resource = prefix_index_encoding
        prefix_index_encoding_case_id = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("case_id")]
        prefix_index_encoding_activity = prefix_index_encoding.iloc[:,
                                         prefix_index_encoding.columns.str.contains("Activity.code")]
        prefix_index_encoding_producer = prefix_index_encoding.iloc[:,
                                         prefix_index_encoding.columns.str.contains("Producer.code")]
        prefix_index_encoding_timesincelastevent = prefix_index_encoding.iloc[:,
                                                   prefix_index_encoding.columns.str.contains("timesincelastevent")]
        prefix_index_encoding_timesincecasestart = prefix_index_encoding.iloc[:,
                                                   prefix_index_encoding.columns.str.contains("timesincecasestart")]
        prefix_index_encoding_timesincemidnight = prefix_index_encoding.iloc[:,
                                                  prefix_index_encoding.columns.str.contains("timesincemidnight")]
        prefix_index_encoding_event_nr = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains(
            "event_nr")]  # 'prefix' column and 'event_nr' column have same values here
        prefix_index_encoding_month = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("month")]
        prefix_index_encoding_weekday = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("weekday")]
        prefix_index_encoding_hour = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("hour")]
        prefix_index_encoding_open_cases = prefix_index_encoding.iloc[:,
                                           prefix_index_encoding.columns.str.contains("open_cases")]
        prefix_index_encoding_label = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains(
            "label_prefixlen1")]  # Should be "label_prefixlen1" all the time
        prefix_index_encoding_Age = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("Age")]
        prefix_index_encoding_Treatment_code = prefix_index_encoding.iloc[:,
                                               prefix_index_encoding.columns.str.contains("Treatment.code")]
        prefix_index_encoding_Diagnosis_code = prefix_index_encoding.iloc[:,
                                               prefix_index_encoding.columns.str.contains("Diagnosis.code")]
        prefix_index_encoding_Specialism_code_1 = prefix_index_encoding.iloc[:,
                                                  prefix_index_encoding.columns.str.contains("Specialism.code.1")]
        prefix_index_encoding_Section = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("Section")]
        prefix_index_encoding_Specialism_code_2 = prefix_index_encoding.iloc[:,
                                                  prefix_index_encoding.columns.str.contains("Specialism.code.2")]
        prefix_index_encoding_group = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("group")]
        prefix_index_encoding_num_of_exec = prefix_index_encoding.iloc[:,
                                            prefix_index_encoding.columns.str.contains("Number.of.executions")]
        data_dummy_prefix_no_resource = pd.concat(
            [prefix_index_encoding_activity, prefix_index_encoding_producer,
             prefix_index_encoding_timesincelastevent, prefix_index_encoding_timesincecasestart,
             prefix_index_encoding_timesincemidnight, prefix_index_encoding_event_nr, prefix_index_encoding_month,
             prefix_index_encoding_weekday, prefix_index_encoding_hour, prefix_index_encoding_open_cases,
             prefix_index_encoding_label, prefix_index_encoding_Age, prefix_index_encoding_Treatment_code,
             prefix_index_encoding_Section, prefix_index_encoding_Specialism_code_1,
             prefix_index_encoding_Specialism_code_2, prefix_index_encoding_group, prefix_index_encoding_num_of_exec],
            axis=1)
    
    if dataset_name == "BPIC15_1_f2_exp_prefix":
        data_dummy_prefix_yes_resource = prefix_index_encoding
        prefix_index_encoding_case_id = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("case_id")]
        prefix_index_encoding_activity = prefix_index_encoding.iloc[:,
                                         prefix_index_encoding.columns.str.contains("Activity")]
        prefix_index_encoding_monitoringResource = prefix_index_encoding.iloc[:,
                                                   prefix_index_encoding.columns.str.contains("monitoringResource")]
        prefix_index_encoding_question = prefix_index_encoding.iloc[:,
                                         prefix_index_encoding.columns.str.contains("question")]
        prefix_index_encoding_responsible_actor = prefix_index_encoding.iloc[:,
                                                  prefix_index_encoding.columns.str.contains("Responsible_actor")]
        prefix_index_encoding_timesincelastevent = prefix_index_encoding.iloc[:,
                                                   prefix_index_encoding.columns.str.contains("timesincelastevent")]
        prefix_index_encoding_timesincecasestart = prefix_index_encoding.iloc[:,
                                                   prefix_index_encoding.columns.str.contains("timesincecasestart")]
        prefix_index_encoding_timesincemidnight = prefix_index_encoding.iloc[:,
                                                  prefix_index_encoding.columns.str.contains("timesincemidnight")]
        prefix_index_encoding_event_nr = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains(
            "event_nr")]  # 'prefix' column and 'event_nr' column have same values here
        prefix_index_encoding_month = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("month")]
        prefix_index_encoding_weekday = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("weekday")]
        prefix_index_encoding_hour = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("hour")]
        prefix_index_encoding_open_cases = prefix_index_encoding.iloc[:,
                                           prefix_index_encoding.columns.str.contains("open_cases")]
        prefix_index_encoding_label = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains(
            "label_prefixlen1")]  # Should be "label_prefixlen1" all the time
        prefix_index_encoding_SUMleges = prefix_index_encoding.iloc[:,
                                         prefix_index_encoding.columns.str.contains("SUMleges")]
        prefix_index_encoding_Aanleg = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("Aanleg")]
        prefix_index_encoding_Bouw = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("Bouw")]
        prefix_index_encoding_Brandveilig = prefix_index_encoding.iloc[:,
                                            prefix_index_encoding.columns.str.contains("Brandveilig")]
        prefix_index_encoding_Handelen = prefix_index_encoding.iloc[:,
                                         prefix_index_encoding.columns.str.contains("Handelen")]
        prefix_index_encoding_Kap = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("Kap")]
        prefix_index_encoding_Milieu = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("Milieu")]
        prefix_index_encoding_Monument = prefix_index_encoding.iloc[:,
                                         prefix_index_encoding.columns.str.contains("Monument")]
        prefix_index_encoding_Reclame = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("Reclame")]
        prefix_index_encoding_Sloop = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("Sloop")]
        prefix_index_encoding_resource = prefix_index_encoding.iloc[:,
                                         prefix_index_encoding.columns.str.contains("org.resource")]
        data_dummy_prefix_no_resource = pd.concat(
            [prefix_index_encoding_activity, prefix_index_encoding_monitoringResource,
             prefix_index_encoding_question, prefix_index_encoding_responsible_actor,
             prefix_index_encoding_timesincelastevent, prefix_index_encoding_timesincecasestart,
             prefix_index_encoding_timesincemidnight, prefix_index_encoding_event_nr, prefix_index_encoding_month,
             prefix_index_encoding_weekday, prefix_index_encoding_hour, prefix_index_encoding_open_cases,
             prefix_index_encoding_label, prefix_index_encoding_SUMleges, prefix_index_encoding_Aanleg,
             prefix_index_encoding_Bouw, prefix_index_encoding_Brandveilig, prefix_index_encoding_Handelen,
             prefix_index_encoding_Kap, prefix_index_encoding_Milieu, prefix_index_encoding_Monument,
             prefix_index_encoding_Reclame, prefix_index_encoding_Sloop, prefix_index_encoding_resource], axis=1)
    
    if dataset_name == "BPIC15_2_f2_exp_prefix":
        data_dummy_prefix_yes_resource = prefix_index_encoding
        prefix_index_encoding_case_id = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("case_id")]
        prefix_index_encoding_activity = prefix_index_encoding.iloc[:,
                                         prefix_index_encoding.columns.str.contains("Activity")]
        prefix_index_encoding_monitoringResource = prefix_index_encoding.iloc[:,
                                                   prefix_index_encoding.columns.str.contains("monitoringResource")]
        prefix_index_encoding_question = prefix_index_encoding.iloc[:,
                                         prefix_index_encoding.columns.str.contains("question")]
        prefix_index_encoding_responsible_actor = prefix_index_encoding.iloc[:,
                                                  prefix_index_encoding.columns.str.contains("Responsible_actor")]
        prefix_index_encoding_timesincelastevent = prefix_index_encoding.iloc[:,
                                                   prefix_index_encoding.columns.str.contains("timesincelastevent")]
        prefix_index_encoding_timesincecasestart = prefix_index_encoding.iloc[:,
                                                   prefix_index_encoding.columns.str.contains("timesincecasestart")]
        prefix_index_encoding_timesincemidnight = prefix_index_encoding.iloc[:,
                                                  prefix_index_encoding.columns.str.contains("timesincemidnight")]
        prefix_index_encoding_event_nr = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains(
            "event_nr")]  # 'prefix' column and 'event_nr' column have same values here
        prefix_index_encoding_month = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("month")]
        prefix_index_encoding_weekday = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("weekday")]
        prefix_index_encoding_hour = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("hour")]
        prefix_index_encoding_open_cases = prefix_index_encoding.iloc[:,
                                           prefix_index_encoding.columns.str.contains("open_cases")]
        prefix_index_encoding_label = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains(
            "label_prefixlen1")]  # Should be "label_prefixlen1" all the time
        prefix_index_encoding_SUMleges = prefix_index_encoding.iloc[:,
                                         prefix_index_encoding.columns.str.contains("SUMleges")]
        prefix_index_encoding_Aanleg = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("Aanleg")]
        prefix_index_encoding_Bouw = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("Bouw")]
        prefix_index_encoding_Brandveilig = prefix_index_encoding.iloc[:,
                                            prefix_index_encoding.columns.str.contains("Brandveilig")]
        prefix_index_encoding_Handelen = prefix_index_encoding.iloc[:,
                                         prefix_index_encoding.columns.str.contains("Handelen")]
        prefix_index_encoding_Kap = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("Kap")]
        prefix_index_encoding_Milieu = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("Milieu")]
        prefix_index_encoding_Monument = prefix_index_encoding.iloc[:,
                                         prefix_index_encoding.columns.str.contains("Monument")]
        prefix_index_encoding_Reclame = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("Reclame")]
        prefix_index_encoding_Sloop = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("Sloop")]
        prefix_index_encoding_resource = prefix_index_encoding.iloc[:,
                                         prefix_index_encoding.columns.str.contains("org.resource")]
        data_dummy_prefix_no_resource = pd.concat(
            [prefix_index_encoding_activity, prefix_index_encoding_monitoringResource,
             prefix_index_encoding_question, prefix_index_encoding_responsible_actor,
             prefix_index_encoding_timesincelastevent, prefix_index_encoding_timesincecasestart,
             prefix_index_encoding_timesincemidnight, prefix_index_encoding_event_nr, prefix_index_encoding_month,
             prefix_index_encoding_weekday, prefix_index_encoding_hour, prefix_index_encoding_open_cases,
             prefix_index_encoding_label, prefix_index_encoding_SUMleges, prefix_index_encoding_Aanleg,
             prefix_index_encoding_Bouw, prefix_index_encoding_Brandveilig, prefix_index_encoding_Handelen,
             prefix_index_encoding_Kap, prefix_index_encoding_Milieu, prefix_index_encoding_Monument,
             prefix_index_encoding_Reclame, prefix_index_encoding_Sloop, prefix_index_encoding_resource], axis=1)
    
    if dataset_name == "BPIC15_3_f2_exp_prefix":
        data_dummy_prefix_yes_resource = prefix_index_encoding
        prefix_index_encoding_case_id = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("case_id")]
        prefix_index_encoding_activity = prefix_index_encoding.iloc[:,
                                         prefix_index_encoding.columns.str.contains("Activity")]
        prefix_index_encoding_monitoringResource = prefix_index_encoding.iloc[:,
                                                   prefix_index_encoding.columns.str.contains("monitoringResource")]
        prefix_index_encoding_question = prefix_index_encoding.iloc[:,
                                         prefix_index_encoding.columns.str.contains("question")]
        prefix_index_encoding_responsible_actor = prefix_index_encoding.iloc[:,
                                                  prefix_index_encoding.columns.str.contains("Responsible_actor")]
        prefix_index_encoding_timesincelastevent = prefix_index_encoding.iloc[:,
                                                   prefix_index_encoding.columns.str.contains("timesincelastevent")]
        prefix_index_encoding_timesincecasestart = prefix_index_encoding.iloc[:,
                                                   prefix_index_encoding.columns.str.contains("timesincecasestart")]
        prefix_index_encoding_timesincemidnight = prefix_index_encoding.iloc[:,
                                                  prefix_index_encoding.columns.str.contains("timesincemidnight")]
        prefix_index_encoding_event_nr = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains(
            "event_nr")]  # 'prefix' column and 'event_nr' column have same values here
        prefix_index_encoding_month = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("month")]
        prefix_index_encoding_weekday = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("weekday")]
        prefix_index_encoding_hour = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("hour")]
        prefix_index_encoding_open_cases = prefix_index_encoding.iloc[:,
                                           prefix_index_encoding.columns.str.contains("open_cases")]
        prefix_index_encoding_label = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains(
            "label_prefixlen1")]  # Should be "label_prefixlen1" all the time
        prefix_index_encoding_SUMleges = prefix_index_encoding.iloc[:,
                                         prefix_index_encoding.columns.str.contains("SUMleges")]
        prefix_index_encoding_Aanleg = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("Aanleg")]
        prefix_index_encoding_Bouw = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("Bouw")]
        prefix_index_encoding_Brandveilig = prefix_index_encoding.iloc[:,
                                            prefix_index_encoding.columns.str.contains("Brandveilig")]
        prefix_index_encoding_Flora = prefix_index_encoding.iloc[:,
                                         prefix_index_encoding.columns.str.contains("Flora")]
        prefix_index_encoding_Handelen = prefix_index_encoding.iloc[:,
                                         prefix_index_encoding.columns.str.contains("Handelen")]
        prefix_index_encoding_Kap = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("Kap")]
        prefix_index_encoding_Milieu = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("Milieu")]
        prefix_index_encoding_Monument = prefix_index_encoding.iloc[:,
                                         prefix_index_encoding.columns.str.contains("Monument")]
        prefix_index_encoding_Reclame = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("Reclame")]
        prefix_index_encoding_Sloop = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("Sloop")]
        prefix_index_encoding_resource = prefix_index_encoding.iloc[:,
                                         prefix_index_encoding.columns.str.contains("org.resource")]
        data_dummy_prefix_no_resource = pd.concat(
            [prefix_index_encoding_activity, prefix_index_encoding_monitoringResource,
             prefix_index_encoding_question, prefix_index_encoding_responsible_actor,
             prefix_index_encoding_timesincelastevent, prefix_index_encoding_timesincecasestart,
             prefix_index_encoding_timesincemidnight, prefix_index_encoding_event_nr, prefix_index_encoding_month,
             prefix_index_encoding_weekday, prefix_index_encoding_hour, prefix_index_encoding_open_cases,
             prefix_index_encoding_label, prefix_index_encoding_SUMleges, prefix_index_encoding_Aanleg,
             prefix_index_encoding_Bouw, prefix_index_encoding_Brandveilig, prefix_index_encoding_Flora, prefix_index_encoding_Handelen,
             prefix_index_encoding_Kap, prefix_index_encoding_Milieu, prefix_index_encoding_Monument,
             prefix_index_encoding_Reclame, prefix_index_encoding_Sloop, prefix_index_encoding_resource], axis=1)
    
    if dataset_name == "BPIC15_4_f2_exp_prefix":
        data_dummy_prefix_yes_resource = prefix_index_encoding
        prefix_index_encoding_case_id = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("case_id")]
        prefix_index_encoding_activity = prefix_index_encoding.iloc[:,
                                         prefix_index_encoding.columns.str.contains("Activity")]
        prefix_index_encoding_monitoringResource = prefix_index_encoding.iloc[:,
                                                   prefix_index_encoding.columns.str.contains("monitoringResource")]
        prefix_index_encoding_question = prefix_index_encoding.iloc[:,
                                         prefix_index_encoding.columns.str.contains("question")]
        prefix_index_encoding_responsible_actor = prefix_index_encoding.iloc[:,
                                                  prefix_index_encoding.columns.str.contains("Responsible_actor")]
        prefix_index_encoding_timesincelastevent = prefix_index_encoding.iloc[:,
                                                   prefix_index_encoding.columns.str.contains("timesincelastevent")]
        prefix_index_encoding_timesincecasestart = prefix_index_encoding.iloc[:,
                                                   prefix_index_encoding.columns.str.contains("timesincecasestart")]
        prefix_index_encoding_timesincemidnight = prefix_index_encoding.iloc[:,
                                                  prefix_index_encoding.columns.str.contains("timesincemidnight")]
        prefix_index_encoding_event_nr = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains(
            "event_nr")]  # 'prefix' column and 'event_nr' column have same values here
        prefix_index_encoding_month = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("month")]
        prefix_index_encoding_weekday = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("weekday")]
        prefix_index_encoding_hour = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("hour")]
        prefix_index_encoding_open_cases = prefix_index_encoding.iloc[:,
                                           prefix_index_encoding.columns.str.contains("open_cases")]
        prefix_index_encoding_label = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains(
            "label_prefixlen1")]  # Should be "label_prefixlen1" all the time
        prefix_index_encoding_SUMleges = prefix_index_encoding.iloc[:,
                                         prefix_index_encoding.columns.str.contains("SUMleges")]
        prefix_index_encoding_Aanleg = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("Aanleg")]
        prefix_index_encoding_Bouw = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("Bouw")]
        prefix_index_encoding_Brandveilig = prefix_index_encoding.iloc[:,
                                            prefix_index_encoding.columns.str.contains("Brandveilig")]
        prefix_index_encoding_Handelen = prefix_index_encoding.iloc[:,
                                         prefix_index_encoding.columns.str.contains("Handelen")]
        prefix_index_encoding_Kap = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("Kap")]
        prefix_index_encoding_Milieu = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("Milieu")]
        prefix_index_encoding_Monument = prefix_index_encoding.iloc[:,
                                         prefix_index_encoding.columns.str.contains("Monument")]
        prefix_index_encoding_Reclame = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("Reclame")]
        prefix_index_encoding_Sloop = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("Sloop")]
        prefix_index_encoding_resource = prefix_index_encoding.iloc[:,
                                         prefix_index_encoding.columns.str.contains("org.resource")]
        data_dummy_prefix_no_resource = pd.concat(
            [prefix_index_encoding_activity, prefix_index_encoding_monitoringResource,
             prefix_index_encoding_question, prefix_index_encoding_responsible_actor,
             prefix_index_encoding_timesincelastevent, prefix_index_encoding_timesincecasestart,
             prefix_index_encoding_timesincemidnight, prefix_index_encoding_event_nr, prefix_index_encoding_month,
             prefix_index_encoding_weekday, prefix_index_encoding_hour, prefix_index_encoding_open_cases,
             prefix_index_encoding_label, prefix_index_encoding_SUMleges, prefix_index_encoding_Aanleg,
             prefix_index_encoding_Bouw, prefix_index_encoding_Brandveilig, prefix_index_encoding_Handelen,
             prefix_index_encoding_Kap, prefix_index_encoding_Milieu, prefix_index_encoding_Monument,
             prefix_index_encoding_Reclame, prefix_index_encoding_Sloop, prefix_index_encoding_resource], axis=1)
    
    if dataset_name == "BPIC15_5_f2_exp_prefix":
        data_dummy_prefix_yes_resource = prefix_index_encoding
        prefix_index_encoding_case_id = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("case_id")]
        prefix_index_encoding_activity = prefix_index_encoding.iloc[:,
                                         prefix_index_encoding.columns.str.contains("Activity")]
        prefix_index_encoding_monitoringResource = prefix_index_encoding.iloc[:,
                                                   prefix_index_encoding.columns.str.contains("monitoringResource")]
        prefix_index_encoding_question = prefix_index_encoding.iloc[:,
                                         prefix_index_encoding.columns.str.contains("question")]
        prefix_index_encoding_responsible_actor = prefix_index_encoding.iloc[:,
                                                  prefix_index_encoding.columns.str.contains("Responsible_actor")]
        prefix_index_encoding_timesincelastevent = prefix_index_encoding.iloc[:,
                                                   prefix_index_encoding.columns.str.contains("timesincelastevent")]
        prefix_index_encoding_timesincecasestart = prefix_index_encoding.iloc[:,
                                                   prefix_index_encoding.columns.str.contains("timesincecasestart")]
        prefix_index_encoding_timesincemidnight = prefix_index_encoding.iloc[:,
                                                  prefix_index_encoding.columns.str.contains("timesincemidnight")]
        prefix_index_encoding_event_nr = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains(
            "event_nr")]  # 'prefix' column and 'event_nr' column have same values here
        prefix_index_encoding_month = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("month")]
        prefix_index_encoding_weekday = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("weekday")]
        prefix_index_encoding_hour = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("hour")]
        prefix_index_encoding_open_cases = prefix_index_encoding.iloc[:,
                                           prefix_index_encoding.columns.str.contains("open_cases")]
        prefix_index_encoding_label = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains(
            "label_prefixlen1")]  # Should be "label_prefixlen1" all the time
        prefix_index_encoding_SUMleges = prefix_index_encoding.iloc[:,
                                         prefix_index_encoding.columns.str.contains("SUMleges")]
        prefix_index_encoding_Aanleg = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("Aanleg")]
        prefix_index_encoding_Bouw = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("Bouw")]
        prefix_index_encoding_Brandveilig = prefix_index_encoding.iloc[:,
                                            prefix_index_encoding.columns.str.contains("Brandveilig")]
        prefix_index_encoding_Flora = prefix_index_encoding.iloc[:,
                                            prefix_index_encoding.columns.str.contains("Flora")]
        prefix_index_encoding_Handelen = prefix_index_encoding.iloc[:,
                                         prefix_index_encoding.columns.str.contains("Handelen")]
        prefix_index_encoding_Integraal = prefix_index_encoding.iloc[:,
                                         prefix_index_encoding.columns.str.contains("Integraal")]
        prefix_index_encoding_Kap = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("Kap")]
        prefix_index_encoding_Milieu = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("Milieu")]
        prefix_index_encoding_Monument = prefix_index_encoding.iloc[:,
                                         prefix_index_encoding.columns.str.contains("Monument")]
        prefix_index_encoding_Reclame = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("Reclame")]
        prefix_index_encoding_Sloop = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("Sloop")]
        prefix_index_encoding_resource = prefix_index_encoding.iloc[:,
                                         prefix_index_encoding.columns.str.contains("org.resource")]
        data_dummy_prefix_no_resource = pd.concat(
            [prefix_index_encoding_activity, prefix_index_encoding_monitoringResource,
             prefix_index_encoding_question, prefix_index_encoding_responsible_actor,
             prefix_index_encoding_timesincelastevent, prefix_index_encoding_timesincecasestart,
             prefix_index_encoding_timesincemidnight, prefix_index_encoding_event_nr, prefix_index_encoding_month,
             prefix_index_encoding_weekday, prefix_index_encoding_hour, prefix_index_encoding_open_cases,
             prefix_index_encoding_label, prefix_index_encoding_SUMleges, prefix_index_encoding_Aanleg,
             prefix_index_encoding_Bouw, prefix_index_encoding_Brandveilig, prefix_index_encoding_Flora, prefix_index_encoding_Handelen,
             prefix_index_encoding_Integraal, prefix_index_encoding_Kap, prefix_index_encoding_Milieu, prefix_index_encoding_Monument,
             prefix_index_encoding_Reclame, prefix_index_encoding_Sloop, prefix_index_encoding_resource], axis=1)
    
    if dataset_name == "traffic_fines_1_exp_prefix":
        data_dummy_prefix_yes_resource = prefix_index_encoding
        prefix_index_encoding_case_id = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("case_id")]
        prefix_index_encoding_activity = prefix_index_encoding.iloc[:,
                                         prefix_index_encoding.columns.str.contains("Activity")]
        prefix_index_encoding_article = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("article")]
        prefix_index_encoding_amount = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("amount")]
        prefix_index_encoding_vehicleClass = prefix_index_encoding.iloc[:,
                                             prefix_index_encoding.columns.str.contains("vehicleClass")]
        prefix_index_encoding_points = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("points")]
        prefix_index_encoding_lastSent = prefix_index_encoding.iloc[:,
                                         prefix_index_encoding.columns.str.contains("lastSent")]
        prefix_index_encoding_notificationType = prefix_index_encoding.iloc[:,
                                                 prefix_index_encoding.columns.str.contains("notificationType")]
        prefix_index_encoding_dismissal = prefix_index_encoding.iloc[:,
                                          prefix_index_encoding.columns.str.contains("dismissal")]
        prefix_index_encoding_expense = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("expense")]
        prefix_index_encoding_monitoringResource = prefix_index_encoding.iloc[:,
                                                   prefix_index_encoding.columns.str.contains("monitoringResource")]
        prefix_index_encoding_responsible_actor = prefix_index_encoding.iloc[:,
                                                  prefix_index_encoding.columns.str.contains("Responsible_actor")]
        prefix_index_encoding_timesincelastevent = prefix_index_encoding.iloc[:,
                                                   prefix_index_encoding.columns.str.contains("timesincelastevent")]
        prefix_index_encoding_timesincecasestart = prefix_index_encoding.iloc[:,
                                                   prefix_index_encoding.columns.str.contains("timesincecasestart")]
        prefix_index_encoding_timesincemidnight = prefix_index_encoding.iloc[:,
                                                  prefix_index_encoding.columns.str.contains("timesincemidnight")]
        prefix_index_encoding_event_nr = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains(
            "event_nr")]  # 'prefix' column and 'event_nr' column have same values here
        prefix_index_encoding_month = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("month")]
        prefix_index_encoding_weekday = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("weekday")]
        prefix_index_encoding_hour = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("hour")]
        prefix_index_encoding_open_cases = prefix_index_encoding.iloc[:,
                                           prefix_index_encoding.columns.str.contains("open_cases")]
        prefix_index_encoding_label = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains(
            "label_prefixlen1")]  # Should be "label_prefixlen1" all the time
        prefix_index_encoding_resource = prefix_index_encoding.iloc[:,
                                         prefix_index_encoding.columns.str.contains("Resource")]
        data_dummy_prefix_no_resource = pd.concat(
            [prefix_index_encoding_activity, prefix_index_encoding_article,
             prefix_index_encoding_amount, prefix_index_encoding_vehicleClass, prefix_index_encoding_points,
             prefix_index_encoding_lastSent, prefix_index_encoding_notificationType, prefix_index_encoding_dismissal,
             prefix_index_encoding_expense, prefix_index_encoding_monitoringResource,
             prefix_index_encoding_responsible_actor, prefix_index_encoding_timesincelastevent,
             prefix_index_encoding_timesincecasestart, prefix_index_encoding_timesincemidnight,
             prefix_index_encoding_event_nr, prefix_index_encoding_month, prefix_index_encoding_weekday,
             prefix_index_encoding_hour, prefix_index_encoding_open_cases, prefix_index_encoding_label,
             prefix_index_encoding_resource], axis=1)
    
    if dataset_name == "bpic2012_O_CANCELLED-COMPLETE_exp_prefix":
        data_dummy_prefix_yes_resource = prefix_index_encoding
        prefix_index_encoding_case_id = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("case_id")]
        prefix_index_encoding_activity = prefix_index_encoding.iloc[:,
                                         prefix_index_encoding.columns.str.contains("Activity")]
        prefix_index_encoding_amount_req = prefix_index_encoding.iloc[:,
                                           prefix_index_encoding.columns.str.contains("AMOUNT_REQ")]
        prefix_index_encoding_timesincelastevent = prefix_index_encoding.iloc[:,
                                                   prefix_index_encoding.columns.str.contains("timesincelastevent")]
        prefix_index_encoding_timesincecasestart = prefix_index_encoding.iloc[:,
                                                   prefix_index_encoding.columns.str.contains("timesincecasestart")]
        prefix_index_encoding_timesincemidnight = prefix_index_encoding.iloc[:,
                                                  prefix_index_encoding.columns.str.contains("timesincemidnight")]
        prefix_index_encoding_event_nr = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains(
            "event_nr")]  # 'prefix' column and 'event_nr' column have same values here
        prefix_index_encoding_month = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("month")]
        prefix_index_encoding_weekday = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("weekday")]
        prefix_index_encoding_hour = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("hour")]
        prefix_index_encoding_open_cases = prefix_index_encoding.iloc[:,
                                           prefix_index_encoding.columns.str.contains("open_cases")]
        prefix_index_encoding_label = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains(
            "label_prefixlen1")]  # Should be "label_prefixlen1" all the time
        prefix_index_encoding_resource = prefix_index_encoding.iloc[:,
                                         prefix_index_encoding.columns.str.contains("Resource")]
        data_dummy_prefix_no_resource = pd.concat(
            [prefix_index_encoding_activity, prefix_index_encoding_amount_req,
             prefix_index_encoding_timesincelastevent, prefix_index_encoding_timesincecasestart,
             prefix_index_encoding_timesincemidnight, prefix_index_encoding_event_nr, prefix_index_encoding_month,
             prefix_index_encoding_weekday, prefix_index_encoding_hour, prefix_index_encoding_open_cases,
             prefix_index_encoding_label, prefix_index_encoding_resource], axis=1)
    
    if dataset_name == "bpic2012_O_ACCEPTED-COMPLETE_exp_prefix":
        data_dummy_prefix_yes_resource = prefix_index_encoding
        prefix_index_encoding_case_id = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("case_id")]
        prefix_index_encoding_activity = prefix_index_encoding.iloc[:,
                                         prefix_index_encoding.columns.str.contains("Activity")]
        prefix_index_encoding_amount_req = prefix_index_encoding.iloc[:,
                                           prefix_index_encoding.columns.str.contains("AMOUNT_REQ")]
        prefix_index_encoding_timesincelastevent = prefix_index_encoding.iloc[:,
                                                   prefix_index_encoding.columns.str.contains("timesincelastevent")]
        prefix_index_encoding_timesincecasestart = prefix_index_encoding.iloc[:,
                                                   prefix_index_encoding.columns.str.contains("timesincecasestart")]
        prefix_index_encoding_timesincemidnight = prefix_index_encoding.iloc[:,
                                                  prefix_index_encoding.columns.str.contains("timesincemidnight")]
        prefix_index_encoding_event_nr = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains(
            "event_nr")]  # 'prefix' column and 'event_nr' column have same values here
        prefix_index_encoding_month = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("month")]
        prefix_index_encoding_weekday = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("weekday")]
        prefix_index_encoding_hour = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("hour")]
        prefix_index_encoding_open_cases = prefix_index_encoding.iloc[:,
                                           prefix_index_encoding.columns.str.contains("open_cases")]
        prefix_index_encoding_label = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains(
            "label_prefixlen1")]  # Should be "label_prefixlen1" all the time
        prefix_index_encoding_resource = prefix_index_encoding.iloc[:,
                                         prefix_index_encoding.columns.str.contains("Resource")]
        data_dummy_prefix_no_resource = pd.concat(
            [prefix_index_encoding_activity, prefix_index_encoding_amount_req,
             prefix_index_encoding_timesincelastevent, prefix_index_encoding_timesincecasestart,
             prefix_index_encoding_timesincemidnight, prefix_index_encoding_event_nr, prefix_index_encoding_month,
             prefix_index_encoding_weekday, prefix_index_encoding_hour, prefix_index_encoding_open_cases,
             prefix_index_encoding_label, prefix_index_encoding_resource], axis=1)
    
    if dataset_name == "bpic2012_O_DECLINED-COMPLETE_exp_prefix":
        data_dummy_prefix_yes_resource = prefix_index_encoding
        prefix_index_encoding_case_id = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("case_id")]
        prefix_index_encoding_activity = prefix_index_encoding.iloc[:,
                                         prefix_index_encoding.columns.str.contains("Activity")]
        prefix_index_encoding_amount_req = prefix_index_encoding.iloc[:,
                                           prefix_index_encoding.columns.str.contains("AMOUNT_REQ")]
        prefix_index_encoding_timesincelastevent = prefix_index_encoding.iloc[:,
                                                   prefix_index_encoding.columns.str.contains("timesincelastevent")]
        prefix_index_encoding_timesincecasestart = prefix_index_encoding.iloc[:,
                                                   prefix_index_encoding.columns.str.contains("timesincecasestart")]
        prefix_index_encoding_timesincemidnight = prefix_index_encoding.iloc[:,
                                                  prefix_index_encoding.columns.str.contains("timesincemidnight")]
        prefix_index_encoding_event_nr = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains(
            "event_nr")]  # 'prefix' column and 'event_nr' column have same values here
        prefix_index_encoding_month = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("month")]
        prefix_index_encoding_weekday = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("weekday")]
        prefix_index_encoding_hour = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("hour")]
        prefix_index_encoding_open_cases = prefix_index_encoding.iloc[:,
                                           prefix_index_encoding.columns.str.contains("open_cases")]
        prefix_index_encoding_label = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains(
            "label_prefixlen1")]  # Should be "label_prefixlen1" all the time
        prefix_index_encoding_resource = prefix_index_encoding.iloc[:,
                                         prefix_index_encoding.columns.str.contains("Resource")]
        data_dummy_prefix_no_resource = pd.concat(
            [prefix_index_encoding_activity, prefix_index_encoding_amount_req,
             prefix_index_encoding_timesincelastevent, prefix_index_encoding_timesincecasestart,
             prefix_index_encoding_timesincemidnight, prefix_index_encoding_event_nr, prefix_index_encoding_month,
             prefix_index_encoding_weekday, prefix_index_encoding_hour, prefix_index_encoding_open_cases,
             prefix_index_encoding_label, prefix_index_encoding_resource], axis=1)
    
    if dataset_name == "BPIC17_O_Cancelled_exp_prefix":
        data_dummy_prefix_yes_resource = prefix_index_encoding
        prefix_index_encoding_case_id = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("case_id")]
        prefix_index_encoding_activity_a = prefix_index_encoding.iloc[:,
                                           prefix_index_encoding.columns.str.contains("Activity_A")]
        prefix_index_encoding_activity_o = prefix_index_encoding.iloc[:,
                                           prefix_index_encoding.columns.str.contains("Activity_O")]
        prefix_index_encoding_activity_w = prefix_index_encoding.iloc[:,
                                           prefix_index_encoding.columns.str.contains("Activity_W")]
        prefix_index_encoding_application = prefix_index_encoding.iloc[:,
                                            prefix_index_encoding.columns.str.contains("ApplicationType")]
        prefix_index_encoding_loangoal = prefix_index_encoding.iloc[:,
                                         prefix_index_encoding.columns.str.contains("LoanGoal")]
        prefix_index_encoding_action = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("Action")]
        prefix_index_encoding_eventorigin = prefix_index_encoding.iloc[:,
                                            prefix_index_encoding.columns.str.contains("EventOrigin")]
        prefix_index_encoding_lifecycle_transition = prefix_index_encoding.iloc[:,
                                                     prefix_index_encoding.columns.str.contains("lifecycle.transition")]
        prefix_index_encoding_accepted_false = prefix_index_encoding.iloc[:,
                                               prefix_index_encoding.columns.str.contains("Accepted_FALSE")]
        prefix_index_encoding_accepted_true = prefix_index_encoding.iloc[:,
                                              prefix_index_encoding.columns.str.contains("Accepted_TRUE")]
        prefix_index_encoding_accepted_missing = prefix_index_encoding.iloc[:,
                                                 prefix_index_encoding.columns.str.contains("Accepted_missing")]
        prefix_index_encoding_selected_false = prefix_index_encoding.iloc[:,
                                               prefix_index_encoding.columns.str.contains("Selected_FALSE")]
        prefix_index_encoding_selected_true = prefix_index_encoding.iloc[:,
                                              prefix_index_encoding.columns.str.contains("Selected_TRUE")]
        prefix_index_encoding_selected_missing = prefix_index_encoding.iloc[:,
                                                 prefix_index_encoding.columns.str.contains("Selected_missing")]
        prefix_index_encoding_firstwithdrawalamount = prefix_index_encoding.iloc[:,
                                                      prefix_index_encoding.columns.str.contains("FirstWithdrawalAmount")]
        prefix_index_encoding_monthlycost = prefix_index_encoding.iloc[:,
                                            prefix_index_encoding.columns.str.contains("MonthlyCost")]
        prefix_index_encoding_numberofterms = prefix_index_encoding.iloc[:,
                                              prefix_index_encoding.columns.str.contains("NumberOfTerms")]
        prefix_index_encoding_offeredamount = prefix_index_encoding.iloc[:,
                                              prefix_index_encoding.columns.str.contains("OfferedAmount")]
        prefix_index_encoding_creditscore = prefix_index_encoding.iloc[:,
                                            prefix_index_encoding.columns.str.contains("CreditScore")]
        prefix_index_encoding_requestedamount = prefix_index_encoding.iloc[:,
                                                prefix_index_encoding.columns.str.contains("RequestedAmount")]
        prefix_index_encoding_timesincelastevent = prefix_index_encoding.iloc[:,
                                                   prefix_index_encoding.columns.str.contains("timesincelastevent")]
        prefix_index_encoding_timesincecasestart = prefix_index_encoding.iloc[:,
                                                   prefix_index_encoding.columns.str.contains("timesincecasestart")]
        prefix_index_encoding_timesincemidnight = prefix_index_encoding.iloc[:,
                                                  prefix_index_encoding.columns.str.contains("timesincemidnight")]
        prefix_index_encoding_event_nr = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains(
            "event_nr")]  # 'prefix' column and 'event_nr' column have same values here
        prefix_index_encoding_month = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("month")]
        prefix_index_encoding_weekday = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("weekday")]
        prefix_index_encoding_hour = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("hour")]
        prefix_index_encoding_open_cases = prefix_index_encoding.iloc[:,
                                           prefix_index_encoding.columns.str.contains("open_cases")]
        prefix_index_encoding_label = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains(
            "label_prefixlen1")]  # Should be "label_prefixlen1" all the time
        prefix_index_encoding_resource = prefix_index_encoding.iloc[:,
                                         prefix_index_encoding.columns.str.contains("org.resource")]
        data_dummy_prefix_no_resource = pd.concat(
            [prefix_index_encoding_activity_a, prefix_index_encoding_activity_o,
             prefix_index_encoding_activity_w, prefix_index_encoding_application, prefix_index_encoding_loangoal,
             prefix_index_encoding_action, prefix_index_encoding_eventorigin, prefix_index_encoding_lifecycle_transition,
             prefix_index_encoding_accepted_false, prefix_index_encoding_accepted_true,
             prefix_index_encoding_accepted_missing, prefix_index_encoding_selected_false,
             prefix_index_encoding_selected_true, prefix_index_encoding_selected_missing,
             prefix_index_encoding_firstwithdrawalamount, prefix_index_encoding_monthlycost,
             prefix_index_encoding_numberofterms, prefix_index_encoding_offeredamount, prefix_index_encoding_creditscore,
             prefix_index_encoding_requestedamount, prefix_index_encoding_timesincelastevent,
             prefix_index_encoding_timesincecasestart, prefix_index_encoding_timesincemidnight,
             prefix_index_encoding_event_nr, prefix_index_encoding_month, prefix_index_encoding_weekday,
             prefix_index_encoding_hour, prefix_index_encoding_open_cases, prefix_index_encoding_label,
             prefix_index_encoding_resource], axis=1)
    
    if dataset_name == "BPIC17_O_Accepted_exp_prefix":
        data_dummy_prefix_yes_resource = prefix_index_encoding
        #prefix_index_encoding_case_id = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("case_id")]
        prefix_index_encoding_activity_a = prefix_index_encoding.iloc[:,
                                           prefix_index_encoding.columns.str.contains("Activity_A")]
        prefix_index_encoding_activity_o = prefix_index_encoding.iloc[:,
                                           prefix_index_encoding.columns.str.contains("Activity_O")]
        prefix_index_encoding_activity_w = prefix_index_encoding.iloc[:,
                                           prefix_index_encoding.columns.str.contains("Activity_W")]
        prefix_index_encoding_application = prefix_index_encoding.iloc[:,
                                            prefix_index_encoding.columns.str.contains("ApplicationType")]
        prefix_index_encoding_loangoal = prefix_index_encoding.iloc[:,
                                         prefix_index_encoding.columns.str.contains("LoanGoal")]
        prefix_index_encoding_action = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("Action")]
        prefix_index_encoding_eventorigin = prefix_index_encoding.iloc[:,
                                            prefix_index_encoding.columns.str.contains("EventOrigin")]
        prefix_index_encoding_lifecycle_transition = prefix_index_encoding.iloc[:,
                                                     prefix_index_encoding.columns.str.contains("lifecycle.transition")]
        prefix_index_encoding_accepted_false = prefix_index_encoding.iloc[:,
                                               prefix_index_encoding.columns.str.contains("Accepted_FALSE")]
        prefix_index_encoding_accepted_true = prefix_index_encoding.iloc[:,
                                              prefix_index_encoding.columns.str.contains("Accepted_TRUE")]
        prefix_index_encoding_accepted_missing = prefix_index_encoding.iloc[:,
                                                 prefix_index_encoding.columns.str.contains("Accepted_missing")]
        prefix_index_encoding_selected_false = prefix_index_encoding.iloc[:,
                                               prefix_index_encoding.columns.str.contains("Selected_FALSE")]
        prefix_index_encoding_selected_true = prefix_index_encoding.iloc[:,
                                              prefix_index_encoding.columns.str.contains("Selected_TRUE")]
        prefix_index_encoding_selected_missing = prefix_index_encoding.iloc[:,
                                                 prefix_index_encoding.columns.str.contains("Selected_missing")]
        prefix_index_encoding_firstwithdrawalamount = prefix_index_encoding.iloc[:,
                                                      prefix_index_encoding.columns.str.contains("FirstWithdrawalAmount")]
        prefix_index_encoding_monthlycost = prefix_index_encoding.iloc[:,
                                            prefix_index_encoding.columns.str.contains("MonthlyCost")]
        prefix_index_encoding_numberofterms = prefix_index_encoding.iloc[:,
                                              prefix_index_encoding.columns.str.contains("NumberOfTerms")]
        prefix_index_encoding_offeredamount = prefix_index_encoding.iloc[:,
                                              prefix_index_encoding.columns.str.contains("OfferedAmount")]
        prefix_index_encoding_creditscore = prefix_index_encoding.iloc[:,
                                            prefix_index_encoding.columns.str.contains("CreditScore")]
        prefix_index_encoding_requestedamount = prefix_index_encoding.iloc[:,
                                                prefix_index_encoding.columns.str.contains("RequestedAmount")]
        prefix_index_encoding_timesincelastevent = prefix_index_encoding.iloc[:,
                                                   prefix_index_encoding.columns.str.contains("timesincelastevent")]
        prefix_index_encoding_timesincecasestart = prefix_index_encoding.iloc[:,
                                                   prefix_index_encoding.columns.str.contains("timesincecasestart")]
        prefix_index_encoding_timesincemidnight = prefix_index_encoding.iloc[:,
                                                  prefix_index_encoding.columns.str.contains("timesincemidnight")]
        prefix_index_encoding_event_nr = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains(
            "event_nr")]  # 'prefix' column and 'event_nr' column have same values here
        prefix_index_encoding_month = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("month")]
        prefix_index_encoding_weekday = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("weekday")]
        prefix_index_encoding_hour = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("hour")]
        prefix_index_encoding_open_cases = prefix_index_encoding.iloc[:,
                                           prefix_index_encoding.columns.str.contains("open_cases")]
        prefix_index_encoding_label = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains(
            "label_prefixlen1")]  # Should be "label_prefixlen1" all the time
        prefix_index_encoding_resource = prefix_index_encoding.iloc[:,
                                         prefix_index_encoding.columns.str.contains("org.resource")]
        data_dummy_prefix_no_resource = pd.concat(
            [prefix_index_encoding_activity_a, prefix_index_encoding_activity_o,
             prefix_index_encoding_activity_w, prefix_index_encoding_application, prefix_index_encoding_loangoal,
             prefix_index_encoding_action, prefix_index_encoding_eventorigin, prefix_index_encoding_lifecycle_transition,
             prefix_index_encoding_accepted_false, prefix_index_encoding_accepted_true,
             prefix_index_encoding_accepted_missing, prefix_index_encoding_selected_false,
             prefix_index_encoding_selected_true, prefix_index_encoding_selected_missing,
             prefix_index_encoding_firstwithdrawalamount, prefix_index_encoding_monthlycost,
             prefix_index_encoding_numberofterms, prefix_index_encoding_offeredamount, prefix_index_encoding_creditscore,
             prefix_index_encoding_requestedamount, prefix_index_encoding_timesincelastevent,
             prefix_index_encoding_timesincecasestart, prefix_index_encoding_timesincemidnight,
             prefix_index_encoding_event_nr, prefix_index_encoding_month, prefix_index_encoding_weekday,
             prefix_index_encoding_hour, prefix_index_encoding_open_cases, prefix_index_encoding_label,
             prefix_index_encoding_resource], axis=1)
    
    if dataset_name == "BPIC17_O_Refused_exp_prefix":
        data_dummy_prefix_yes_resource = prefix_index_encoding
        prefix_index_encoding_case_id = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("case_id")]
        prefix_index_encoding_activity_a = prefix_index_encoding.iloc[:,
                                           prefix_index_encoding.columns.str.contains("Activity_A")]
        prefix_index_encoding_activity_o = prefix_index_encoding.iloc[:,
                                           prefix_index_encoding.columns.str.contains("Activity_O")]
        prefix_index_encoding_activity_w = prefix_index_encoding.iloc[:,
                                           prefix_index_encoding.columns.str.contains("Activity_W")]
        prefix_index_encoding_application = prefix_index_encoding.iloc[:,
                                            prefix_index_encoding.columns.str.contains("ApplicationType")]
        prefix_index_encoding_loangoal = prefix_index_encoding.iloc[:,
                                         prefix_index_encoding.columns.str.contains("LoanGoal")]
        prefix_index_encoding_action = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("Action")]
        prefix_index_encoding_eventorigin = prefix_index_encoding.iloc[:,
                                            prefix_index_encoding.columns.str.contains("EventOrigin")]
        prefix_index_encoding_lifecycle_transition = prefix_index_encoding.iloc[:,
                                                     prefix_index_encoding.columns.str.contains("lifecycle.transition")]
        prefix_index_encoding_accepted_false = prefix_index_encoding.iloc[:,
                                               prefix_index_encoding.columns.str.contains("Accepted_FALSE")]
        prefix_index_encoding_accepted_true = prefix_index_encoding.iloc[:,
                                              prefix_index_encoding.columns.str.contains("Accepted_TRUE")]
        prefix_index_encoding_accepted_missing = prefix_index_encoding.iloc[:,
                                                 prefix_index_encoding.columns.str.contains("Accepted_missing")]
        prefix_index_encoding_selected_false = prefix_index_encoding.iloc[:,
                                               prefix_index_encoding.columns.str.contains("Selected_FALSE")]
        prefix_index_encoding_selected_true = prefix_index_encoding.iloc[:,
                                              prefix_index_encoding.columns.str.contains("Selected_TRUE")]
        prefix_index_encoding_selected_missing = prefix_index_encoding.iloc[:,
                                                 prefix_index_encoding.columns.str.contains("Selected_missing")]
        prefix_index_encoding_firstwithdrawalamount = prefix_index_encoding.iloc[:,
                                                      prefix_index_encoding.columns.str.contains("FirstWithdrawalAmount")]
        prefix_index_encoding_monthlycost = prefix_index_encoding.iloc[:,
                                            prefix_index_encoding.columns.str.contains("MonthlyCost")]
        prefix_index_encoding_numberofterms = prefix_index_encoding.iloc[:,
                                              prefix_index_encoding.columns.str.contains("NumberOfTerms")]
        prefix_index_encoding_offeredamount = prefix_index_encoding.iloc[:,
                                              prefix_index_encoding.columns.str.contains("OfferedAmount")]
        prefix_index_encoding_creditscore = prefix_index_encoding.iloc[:,
                                            prefix_index_encoding.columns.str.contains("CreditScore")]
        prefix_index_encoding_requestedamount = prefix_index_encoding.iloc[:,
                                                prefix_index_encoding.columns.str.contains("RequestedAmount")]
        prefix_index_encoding_timesincelastevent = prefix_index_encoding.iloc[:,
                                                   prefix_index_encoding.columns.str.contains("timesincelastevent")]
        prefix_index_encoding_timesincecasestart = prefix_index_encoding.iloc[:,
                                                   prefix_index_encoding.columns.str.contains("timesincecasestart")]
        prefix_index_encoding_timesincemidnight = prefix_index_encoding.iloc[:,
                                                  prefix_index_encoding.columns.str.contains("timesincemidnight")]
        prefix_index_encoding_event_nr = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains(
            "event_nr")]  # 'prefix' column and 'event_nr' column have same values here
        prefix_index_encoding_month = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("month")]
        prefix_index_encoding_weekday = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("weekday")]
        prefix_index_encoding_hour = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains("hour")]
        prefix_index_encoding_open_cases = prefix_index_encoding.iloc[:,
                                           prefix_index_encoding.columns.str.contains("open_cases")]
        prefix_index_encoding_label = prefix_index_encoding.iloc[:, prefix_index_encoding.columns.str.contains(
            "label_prefixlen1")]  # Should be "label_prefixlen1" all the time
        prefix_index_encoding_resource = prefix_index_encoding.iloc[:,
                                         prefix_index_encoding.columns.str.contains("org.resource")]
        data_dummy_prefix_no_resource = pd.concat(
            [prefix_index_encoding_activity_a, prefix_index_encoding_activity_o,
             prefix_index_encoding_activity_w, prefix_index_encoding_application, prefix_index_encoding_loangoal,
             prefix_index_encoding_action, prefix_index_encoding_eventorigin, prefix_index_encoding_lifecycle_transition,
             prefix_index_encoding_accepted_false, prefix_index_encoding_accepted_true,
             prefix_index_encoding_accepted_missing, prefix_index_encoding_selected_false,
             prefix_index_encoding_selected_true, prefix_index_encoding_selected_missing,
             prefix_index_encoding_firstwithdrawalamount, prefix_index_encoding_monthlycost,
             prefix_index_encoding_numberofterms, prefix_index_encoding_offeredamount, prefix_index_encoding_creditscore,
             prefix_index_encoding_requestedamount, prefix_index_encoding_timesincelastevent,
             prefix_index_encoding_timesincecasestart, prefix_index_encoding_timesincemidnight,
             prefix_index_encoding_event_nr, prefix_index_encoding_month, prefix_index_encoding_weekday,
             prefix_index_encoding_hour, prefix_index_encoding_open_cases, prefix_index_encoding_label,
             prefix_index_encoding_resource], axis=1)

    return data_dummy_prefix_no_resource, data_dummy_prefix_yes_resource

def variable_normalize(X_dummy_no_resource):
    if dataset_name == "BPIC11_f1_exp_prefix":
        X_dummy_no_resource['Age_prefixlen1'] = (X_dummy_no_resource['Age_prefixlen1'] - min(
            X_dummy_no_resource['Age_prefixlen1'])) / (max(X_dummy_no_resource['Age_prefixlen1']) - min(
            X_dummy_no_resource['Age_prefixlen1']))  # 0~1 normalization
        for i in range(max_prefix):
            feature_name = "Number.of.executions_prefixlen" + str(i + 1)
            if np.var(X_dummy_no_resource[feature_name]) == 0:
                break
            X_dummy_no_resource[feature_name] = (X_dummy_no_resource[feature_name] - min(
                X_dummy_no_resource[feature_name])) / (max(X_dummy_no_resource[feature_name]) - min(
                X_dummy_no_resource[feature_name]))  # 0~1 normalization
    
    if dataset_name == "BPIC11_f2_exp_prefix":
        X_dummy_no_resource['Age_prefixlen1'] = (X_dummy_no_resource['Age_prefixlen1'] - min(
            X_dummy_no_resource['Age_prefixlen1'])) / (max(X_dummy_no_resource['Age_prefixlen1']) - min(
            X_dummy_no_resource['Age_prefixlen1']))  # 0~1 normalization
        for i in range(max_prefix):
            feature_name = "Number.of.executions_prefixlen" + str(i + 1)
            if np.var(X_dummy_no_resource[feature_name]) == 0:
                break
            X_dummy_no_resource[feature_name] = (X_dummy_no_resource[feature_name] - min(
                X_dummy_no_resource[feature_name])) / (max(X_dummy_no_resource[feature_name]) - min(
                X_dummy_no_resource[feature_name]))  # 0~1 normalization
    
    if dataset_name == "BPIC11_f3_exp_prefix":
        X_dummy_no_resource['Age_prefixlen1'] = (X_dummy_no_resource['Age_prefixlen1'] - min(
            X_dummy_no_resource['Age_prefixlen1'])) / (max(X_dummy_no_resource['Age_prefixlen1']) - min(
            X_dummy_no_resource['Age_prefixlen1']))  # 0~1 normalization
        for i in range(max_prefix):
            feature_name = "Number.of.executions_prefixlen" + str(i + 1)
            if np.var(X_dummy_no_resource[feature_name]) == 0:
                break
            X_dummy_no_resource[feature_name] = (X_dummy_no_resource[feature_name] - min(
                X_dummy_no_resource[feature_name])) / (max(X_dummy_no_resource[feature_name]) - min(
                X_dummy_no_resource[feature_name]))  # 0~1 normalization
    
    if dataset_name == "BPIC11_f4_exp_prefix":
        X_dummy_no_resource['Age_prefixlen1'] = (X_dummy_no_resource['Age_prefixlen1'] - min(
            X_dummy_no_resource['Age_prefixlen1'])) / (max(X_dummy_no_resource['Age_prefixlen1']) - min(
            X_dummy_no_resource['Age_prefixlen1']))  # 0~1 normalization
        for i in range(max_prefix):
            feature_name = "Number.of.executions_prefixlen" + str(i + 1)
            if np.var(X_dummy_no_resource[feature_name]) == 0:
                break
            X_dummy_no_resource[feature_name] = (X_dummy_no_resource[feature_name] - min(
                X_dummy_no_resource[feature_name])) / (max(X_dummy_no_resource[feature_name]) - min(
                X_dummy_no_resource[feature_name]))  # 0~1 normalization
    
    if dataset_name == "BPIC15_1_f2_exp_prefix":
        X_dummy_no_resource['SUMleges_prefixlen1'] = (X_dummy_no_resource['SUMleges_prefixlen1'] - min(
            X_dummy_no_resource['SUMleges_prefixlen1'])) / (max(X_dummy_no_resource['SUMleges_prefixlen1']) - min(
            X_dummy_no_resource['SUMleges_prefixlen1']))  # 0~1 normalization
    
    if dataset_name == "BPIC15_2_f2_exp_prefix":
        X_dummy_no_resource['SUMleges_prefixlen1'] = (X_dummy_no_resource['SUMleges_prefixlen1'] - min(
            X_dummy_no_resource['SUMleges_prefixlen1'])) / (max(X_dummy_no_resource['SUMleges_prefixlen1']) - min(
            X_dummy_no_resource['SUMleges_prefixlen1']))  # 0~1 normalization
    
    if dataset_name == "BPIC15_3_f2_exp_prefix":
        X_dummy_no_resource['SUMleges_prefixlen1'] = (X_dummy_no_resource['SUMleges_prefixlen1'] - min(
            X_dummy_no_resource['SUMleges_prefixlen1'])) / (max(X_dummy_no_resource['SUMleges_prefixlen1']) - min(
            X_dummy_no_resource['SUMleges_prefixlen1']))  # 0~1 normalization
    
    if dataset_name == "BPIC15_4_f2_exp_prefix":
        X_dummy_no_resource['SUMleges_prefixlen1'] = (X_dummy_no_resource['SUMleges_prefixlen1'] - min(
            X_dummy_no_resource['SUMleges_prefixlen1'])) / (max(X_dummy_no_resource['SUMleges_prefixlen1']) - min(
            X_dummy_no_resource['SUMleges_prefixlen1']))  # 0~1 normalization
    
    if dataset_name == "BPIC15_5_f2_exp_prefix":
        X_dummy_no_resource['SUMleges_prefixlen1'] = (X_dummy_no_resource['SUMleges_prefixlen1'] - min(
            X_dummy_no_resource['SUMleges_prefixlen1'])) / (max(X_dummy_no_resource['SUMleges_prefixlen1']) - min(
            X_dummy_no_resource['SUMleges_prefixlen1']))  # 0~1 normalization
    
    # Normalization for features only available in traffic_fines_1_exp_prefix
    if dataset_name == "traffic_fines_1_exp_prefix":
        for i in range(max_prefix):
            feature_name = "amount_prefixlen" + str(i + 1)
            if np.var(X_dummy_no_resource[feature_name]) == 0:
                break
            X_dummy_no_resource[feature_name] = (X_dummy_no_resource[feature_name] - min(
                X_dummy_no_resource[feature_name])) / (max(X_dummy_no_resource[feature_name]) - min(
                X_dummy_no_resource[feature_name]))  # 0~1 normalization
        for i in range(max_prefix):
            feature_name = "expense_prefixlen" + str(i + 1)
            if np.var(X_dummy_no_resource[feature_name]) == 0:
                break
            X_dummy_no_resource[feature_name] = (X_dummy_no_resource[feature_name] - min(
                X_dummy_no_resource[feature_name])) / (max(X_dummy_no_resource[feature_name]) - min(
                X_dummy_no_resource[feature_name]))  # 0~1 normalization
        X_dummy_no_resource['points_prefixlen1'] = (X_dummy_no_resource['points_prefixlen1'] - min(
            X_dummy_no_resource['points_prefixlen1'])) / (max(X_dummy_no_resource['points_prefixlen1']) - min(
            X_dummy_no_resource['points_prefixlen1']))  # 0~1 normalization
    
    # Normalization for features only available in bpic2012_O_CANCELLED-COMPLETE_exp_prefix
    if dataset_name == "bpic2012_O_CANCELLED-COMPLETE_exp_prefix":
        X_dummy_no_resource['AMOUNT_REQ_prefixlen1'] = (X_dummy_no_resource['AMOUNT_REQ_prefixlen1'] - min(
            X_dummy_no_resource['AMOUNT_REQ_prefixlen1'])) / (max(X_dummy_no_resource['AMOUNT_REQ_prefixlen1']) - min(
            X_dummy_no_resource['AMOUNT_REQ_prefixlen1']))  # 0~1 normalization
    
    # Normalization for features only available in bpic2012_O_ACCEPTED-COMPLETE_exp_prefix
    if dataset_name == "bpic2012_O_ACCEPTED-COMPLETE_exp_prefix":
        X_dummy_no_resource['AMOUNT_REQ_prefixlen1'] = (X_dummy_no_resource['AMOUNT_REQ_prefixlen1'] - min(
            X_dummy_no_resource['AMOUNT_REQ_prefixlen1'])) / (max(X_dummy_no_resource['AMOUNT_REQ_prefixlen1']) - min(
            X_dummy_no_resource['AMOUNT_REQ_prefixlen1']))  # 0~1 normalization
    
    # Normalization for features only available in bpic2012_O_DECLINED-COMPLETE_exp_prefix
    if dataset_name == "bpic2012_O_DECLINED-COMPLETE_exp_prefix":
        X_dummy_no_resource['AMOUNT_REQ_prefixlen1'] = (X_dummy_no_resource['AMOUNT_REQ_prefixlen1'] - min(
            X_dummy_no_resource['AMOUNT_REQ_prefixlen1'])) / (max(X_dummy_no_resource['AMOUNT_REQ_prefixlen1']) - min(
            X_dummy_no_resource['AMOUNT_REQ_prefixlen1']))  # 0~1 normalization
    
    # Normalization for features only available in BPIC17_O_Cancelled_exp_prefix
    if dataset_name == "BPIC17_O_Cancelled_exp_prefix":
        X_dummy_no_resource['RequestedAmount_prefixlen1'] = (X_dummy_no_resource['RequestedAmount_prefixlen1'] - min(
            X_dummy_no_resource['RequestedAmount_prefixlen1'])) / (max(
            X_dummy_no_resource['RequestedAmount_prefixlen1']) - min(
            X_dummy_no_resource['RequestedAmount_prefixlen1']))  # 0~1 normalization
        X_dummy_no_resource['FirstWithdrawalAmount_prefixlen1'] = (X_dummy_no_resource[
                                                                       'FirstWithdrawalAmount_prefixlen1'] - min(
            X_dummy_no_resource['FirstWithdrawalAmount_prefixlen1'])) / (max(
            X_dummy_no_resource['FirstWithdrawalAmount_prefixlen1']) - min(
            X_dummy_no_resource['FirstWithdrawalAmount_prefixlen1']))  # 0~1 normalization
        X_dummy_no_resource['MonthlyCost_prefixlen1'] = (X_dummy_no_resource['MonthlyCost_prefixlen1'] - min(
            X_dummy_no_resource['MonthlyCost_prefixlen1'])) / (max(X_dummy_no_resource['MonthlyCost_prefixlen1']) - min(
            X_dummy_no_resource['MonthlyCost_prefixlen1']))  # 0~1 normalization
        X_dummy_no_resource['NumberOfTerms_prefixlen1'] = (X_dummy_no_resource['NumberOfTerms_prefixlen1'] - min(
            X_dummy_no_resource['NumberOfTerms_prefixlen1'])) / (max(X_dummy_no_resource['NumberOfTerms_prefixlen1']) - min(
            X_dummy_no_resource['NumberOfTerms_prefixlen1']))  # 0~1 normalization
        X_dummy_no_resource['OfferedAmount_prefixlen1'] = (X_dummy_no_resource['OfferedAmount_prefixlen1'] - min(
            X_dummy_no_resource['OfferedAmount_prefixlen1'])) / (max(X_dummy_no_resource['OfferedAmount_prefixlen1']) - min(
            X_dummy_no_resource['OfferedAmount_prefixlen1']))  # 0~1 normalization
        X_dummy_no_resource['CreditScore_prefixlen1'] = (X_dummy_no_resource['CreditScore_prefixlen1'] - min(
            X_dummy_no_resource['CreditScore_prefixlen1'])) / (max(X_dummy_no_resource['CreditScore_prefixlen1']) - min(
            X_dummy_no_resource['CreditScore_prefixlen1']))  # 0~1 normalization
    
    # Normalization for features only available in BPIC17_O_Cancelled_exp_prefix
    if dataset_name == "BPIC17_O_Accepted_exp_prefix":
        X_dummy_no_resource['RequestedAmount_prefixlen1'] = (X_dummy_no_resource['RequestedAmount_prefixlen1'] - min(
            X_dummy_no_resource['RequestedAmount_prefixlen1'])) / (max(
            X_dummy_no_resource['RequestedAmount_prefixlen1']) - min(
            X_dummy_no_resource['RequestedAmount_prefixlen1']))  # 0~1 normalization
        X_dummy_no_resource['FirstWithdrawalAmount_prefixlen1'] = (X_dummy_no_resource[
                                                                       'FirstWithdrawalAmount_prefixlen1'] - min(
            X_dummy_no_resource['FirstWithdrawalAmount_prefixlen1'])) / (max(
            X_dummy_no_resource['FirstWithdrawalAmount_prefixlen1']) - min(
            X_dummy_no_resource['FirstWithdrawalAmount_prefixlen1']))  # 0~1 normalization
        X_dummy_no_resource['MonthlyCost_prefixlen1'] = (X_dummy_no_resource['MonthlyCost_prefixlen1'] - min(
            X_dummy_no_resource['MonthlyCost_prefixlen1'])) / (max(X_dummy_no_resource['MonthlyCost_prefixlen1']) - min(
            X_dummy_no_resource['MonthlyCost_prefixlen1']))  # 0~1 normalization
        X_dummy_no_resource['NumberOfTerms_prefixlen1'] = (X_dummy_no_resource['NumberOfTerms_prefixlen1'] - min(
            X_dummy_no_resource['NumberOfTerms_prefixlen1'])) / (max(X_dummy_no_resource['NumberOfTerms_prefixlen1']) - min(
            X_dummy_no_resource['NumberOfTerms_prefixlen1']))  # 0~1 normalization
        X_dummy_no_resource['OfferedAmount_prefixlen1'] = (X_dummy_no_resource['OfferedAmount_prefixlen1'] - min(
            X_dummy_no_resource['OfferedAmount_prefixlen1'])) / (max(X_dummy_no_resource['OfferedAmount_prefixlen1']) - min(
            X_dummy_no_resource['OfferedAmount_prefixlen1']))  # 0~1 normalization
        X_dummy_no_resource['CreditScore_prefixlen1'] = (X_dummy_no_resource['CreditScore_prefixlen1'] - min(
            X_dummy_no_resource['CreditScore_prefixlen1'])) / (max(X_dummy_no_resource['CreditScore_prefixlen1']) - min(
            X_dummy_no_resource['CreditScore_prefixlen1']))  # 0~1 normalization
    
    # Normalization for features only available in BPIC17_O_Refused_exp_prefix
    if dataset_name == "BPIC17_O_Refused_exp_prefix":
        X_dummy_no_resource['RequestedAmount_prefixlen1'] = (X_dummy_no_resource['RequestedAmount_prefixlen1'] - min(
            X_dummy_no_resource['RequestedAmount_prefixlen1'])) / (max(
            X_dummy_no_resource['RequestedAmount_prefixlen1']) - min(
            X_dummy_no_resource['RequestedAmount_prefixlen1']))  # 0~1 normalization
        X_dummy_no_resource['FirstWithdrawalAmount_prefixlen1'] = (X_dummy_no_resource[
                                                                       'FirstWithdrawalAmount_prefixlen1'] - min(
            X_dummy_no_resource['FirstWithdrawalAmount_prefixlen1'])) / (max(
            X_dummy_no_resource['FirstWithdrawalAmount_prefixlen1']) - min(
            X_dummy_no_resource['FirstWithdrawalAmount_prefixlen1']))  # 0~1 normalization
        X_dummy_no_resource['MonthlyCost_prefixlen1'] = (X_dummy_no_resource['MonthlyCost_prefixlen1'] - min(
            X_dummy_no_resource['MonthlyCost_prefixlen1'])) / (max(X_dummy_no_resource['MonthlyCost_prefixlen1']) - min(
            X_dummy_no_resource['MonthlyCost_prefixlen1']))  # 0~1 normalization
        X_dummy_no_resource['NumberOfTerms_prefixlen1'] = (X_dummy_no_resource['NumberOfTerms_prefixlen1'] - min(
            X_dummy_no_resource['NumberOfTerms_prefixlen1'])) / (max(X_dummy_no_resource['NumberOfTerms_prefixlen1']) - min(
            X_dummy_no_resource['NumberOfTerms_prefixlen1']))  # 0~1 normalization
        X_dummy_no_resource['OfferedAmount_prefixlen1'] = (X_dummy_no_resource['OfferedAmount_prefixlen1'] - min(
            X_dummy_no_resource['OfferedAmount_prefixlen1'])) / (max(X_dummy_no_resource['OfferedAmount_prefixlen1']) - min(
            X_dummy_no_resource['OfferedAmount_prefixlen1']))  # 0~1 normalization
        X_dummy_no_resource['CreditScore_prefixlen1'] = (X_dummy_no_resource['CreditScore_prefixlen1'] - min(
            X_dummy_no_resource['CreditScore_prefixlen1'])) / (max(X_dummy_no_resource['CreditScore_prefixlen1']) - min(
            X_dummy_no_resource['CreditScore_prefixlen1']))  # 0~1 normalization

    return X_dummy_no_resource














