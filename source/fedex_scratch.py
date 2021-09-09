import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_columns', 50)

consignment = pd.read_csv('data/fedex/Consignment_data.csv',
                          parse_dates=['package_nbr_create date local',
                                       'package_nbr create date', 'package_nbr local time',
                                       'package_nbr create local date',
                                       'package_nbr pickup date',
                                       'package_nbr pickup time', 'package_nbr pickup local time',
                                       'package_nbr pickup local date',
                                       'package_nbr departure actual date',
                                       'Date / Time to be delivered', 'Date to be delivered',
                                       'Local Date / Time to be delivered',
                                       'Local Date to be delivered',
                                       'package_nbr_create date'])  # 25770
country = pd.read_csv('data/fedex/Countries.csv')
pod = pd.read_csv('data/fedex/POD.csv')


consignment.columns = [c.replace(' ', '_').lower() for c in consignment.columns]
country.columns = [c.replace(' ', '_').lower() for c in country.columns]
pod.columns = [c.replace(' ', '_').lower() for c in pod.columns]

consignment = consignment[(consignment['country_of_origin'] == 'DE') &
                          (consignment['country_of_destination'] == 'DE')]

consignment_vc = consignment['package_nbr'].value_counts()
consignment_vc[consignment_vc > 1]
consignment[consignment['package_nbr'] == 4609991269]  # dup, 14348
consignment.drop_duplicates(subset=['package_nbr'], inplace=True)  # 14347

check = pd.merge(consignment, country, on='package_nbr', how='left',
                 suffixes=['_consignment', '_country'])
check_vc = check['package_nbr'].value_counts()
check_vc[check_vc > 1]
check[check['package_nbr'] == 4609991269]
check[(check['country_of_origin_country'] != 'DE') |
      (check['country_of_destination_country'] != 'DE')]  # 5175 / 14350
# HERE DONT FILTER

consignment.info()
pod.info()

pod['event_desciption'].value_counts()

data = pd.merge(consignment, pod, on='package_nbr', how='left')
data = data[['package_nbr', 'item_number', 'local_date_to_be_delivered', 'event_date',
             'event_desciption']]
data['package_nbr'].nunique()  # 14347

deliver_set = ['Delivered', 'Delivered to Neighbor', 'Delivered unattended with permission',
               'In mailbox no signature']
data['local_date_to_be_delivered'] = pd.to_datetime(data['local_date_to_be_delivered'])
data['event_date'] = pd.to_datetime(data['event_date'])
data['on_time'] = (data['event_date'].dt.date <= data['local_date_to_be_delivered']) & \
                  (data['event_desciption'].isin(deliver_set))
data['on_time'].value_counts()  # T/F --> 10223 / 3259

data_to_be_delivered = data[data['local_date_to_be_delivered'] == datetime(2019, 6, 15)]

data_to_be_delivered['event_desciption'].isnull().sum() * (1 - data['on_time'].sum()/ len(data)) + \
len(data_to_be_delivered[data_to_be_delivered['on_time'] == False])  # 57


