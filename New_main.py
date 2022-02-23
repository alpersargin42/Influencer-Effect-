import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn import *
from sklearn.ensemble import *
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import *
from sklearn.preprocessing import *
from sklearn.neighbors import *
from sklearn.metrics import *
from sklearn.tree import *
from sklearn.datasets import *
from sklearn.cluster import *
from sklearn.utils import *
from sklearn.svm import *
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier

import seaborn as sns
from matplotlib.colors import ListedColormap
from six import StringIO
import pylab as pl

def session_control1():
    dosya_listesi = os.listdir()
    if "Cosine.csv"in dosya_listesi:
        return True

    else:
        return False

def session_control2():
    dosya_listesi=os.listdir()
    if "Choice.csv" in dosya_listesi:
        return True
    else:
        return False

def session_control3():
    dosya_listesi = os.listdir()
    if "A_Pearson.csv" in dosya_listesi:
        return True
    else:
        return False

def session_control4():
    dosya_listesi = os.listdir()
    if "B_Pearson.csv" in dosya_listesi:
        return True
    else:
        return False
def session_control5():
    dosya_listesi = os.listdir()
    if "TOTAL_Pearson.csv" in dosya_listesi:
        return True
    else:
        return False
def session_control6():
    dosya_listesi = os.listdir()
    if "submit.csv" in dosya_listesi:
        return True
    else:
        return False
def session_control7():
    dosya_listesi = os.listdir()
    if "sub.csv" in dosya_listesi:
        return True
    else:
        return False
def session_control8():
    dosya_listesi = os.listdir()
    if "sirala.csv" in dosya_listesi:
        return True
    else:
        return False

def aDegerler():
    adegerlerDizisi=[]
    for i in range(5500):
        for j in range(11):
            adegerlerDizisi = aData[i][j]
            # print(adegerlerDizisi)

def bDegerler():
    bdegerlerDizisi=[]
    for i in range(5500):
        for j in range(11):
            bdegerlerDizisi = bData[i][j]
            # print(bdegerlerDizisi)

data=np.array(pd.read_csv("PeerIndex.csv",delimiter=","))
aData=data[0:,0:11]
bData=data[0:,12:23]

aDegerler()
bDegerler()
adegerlerDizisi2=aData.flatten()
adegerlerDizisi3=np.array_split(adegerlerDizisi2,11)
# print(adegerlerDizisi3,end="")
# print(aData.shape)

bdegerlerDizisi2=bData.flatten()
bdegerlerDizisi3=np.array_split(bdegerlerDizisi2,11)
# print(bdegerlerDizisi3,end="\n")
# print(bData.shape)

choicedizi=np.array([])

if session_control2()==False:
    dosya=open("Choice.csv","w")
    for i in range(5500):
        choicedizi = np.array(aData[i][0])
        print(choicedizi, file=dosya)
else:
    dosya=open("Choice.csv","r")

dosya.close()
# scaler = StandardScaler()
# scaler.fit(aData)
# X_scaled = scaler.transform(aData)
# print(X_scaled)
# similarities = cosine_similarity(X_scaled,X_scaled)
# print(similarities)

print("Cosine Simularity")
similarities = pd.DataFrame(cosine_similarity(aData, bData))
if session_control1()==False:
    similarities.to_csv("Cosine.csv")
print(similarities)

print("Pearson Simularity")
#dataa = np.corrcoef(adegerlerDizisi2, bdegerlerDizisi2)

dataa=pd.DataFrame(np.corrcoef(aData))
if session_control3()==False:
    dataa.to_csv("A_Pearson.csv")
print(dataa)

datab=pd.DataFrame(np.corrcoef(bData))
if session_control4()==False:
    datab.to_csv("B_Pearson.csv")
print(datab)

data_total=pd.DataFrame(np.corrcoef(data))
print(data_total)

feature_cols=['Choice','A_follower_count',
               'A_following_count','A_listed_count',
               'A_mentions_received','A_retweets_received',
               'A_mentions_sent','A_retweets_sent',
               'A_posts','A_network_feature_1',
               'A_network_feature_2','A_network_feature_3',
               'B_follower_count','B_following_count',
               'B_listed_count','B_mentions_received',
               'B_retweets_received','B_mentions_sent',
               'B_retweets_sent','B_posts',
               'B_network_feature_1','B_network_feature_2',
               'B_network_feature_3']
dataset=pd.DataFrame(pd.read_csv("PeerIndex.csv",delimiter=",",header=None,names=feature_cols))

traindf = pd.read_csv("train.csv")
testdf = pd.read_csv("test.csv")
train_outcome = pd.crosstab(index=traindf["Choice"],
                              columns="count")
features=['A_follower_count','A_listed_count','A_mentions_received','A_retweets_received','A_posts','A_network_feature_1','A_network_feature_2','A_network_feature_3','B_follower_count','B_following_count','B_listed_count','B_mentions_received','B_retweets_received','B_posts','B_network_feature_1','B_network_feature_2','B_network_feature_3']

np.set_printoptions(precision=4)
pd.set_option("display.precision",4)
predictors = list(traindf.columns[1:])
target = [traindf.columns[0]]
A_predictors = list(traindf.columns[1:12])
B_predictors = list(traindf.columns[12:23])
X_train = traindf[predictors]

y_train = traindf[target]
choice_A_data = traindf.loc[traindf['Choice']==0]
choice_B_data = traindf.loc[traindf['Choice']==1]
x_space = np.linspace(start= traindf.A_follower_count.min(), stop=traindf.A_follower_count.max(),num=len(traindf['A_follower_count']))
x_space = np.linspace(start= choice_A_data.A_follower_count.min(), stop=choice_A_data.A_follower_count.max(),num=len(choice_A_data['A_follower_count']))
fig = plt.figure(figsize=(12,6))
plt.scatter(x_space,choice_A_data['A_follower_count'])
plt.show()
fig = plt.figure(figsize=(12,6))
plt.plot(x_space, choice_A_data[['A_follower_count','A_mentions_received']],'-g')
plt.show()
rfc = RandomForestClassifier()
rfc.fit(X= traindf[predictors], y =traindf[target] )
predict_values = rfc.predict(testdf)
predict_df = pd.DataFrame({"Id": list(range(1,len(predict_values)+1)),
                           "Choice": predict_values})

if session_control6()==False:
      predict_df.to_csv("submit.csv",index = False)
      print('Saved..')

def pre_pro(df):
    df = df.astype('float32')
    col = df.columns
    for i in range(len(col)):
        m = df.loc[df[col[i]] != -np.inf, col[i]].min()
        df[col[i]].replace(-np.inf, m, inplace=True)
        M = df.loc[df[col[i]] != np.inf, col[i]].max()
        df[col[i]].replace(np.inf, M, inplace=True)

    df.fillna(0, inplace=True)
    return df


def feat_eng(df):
    df.replace(0, 0.001)

    df['follower_diff'] = (df['A_follower_count'] > df['B_follower_count'])
    df['following_diff'] = (df['A_following_count'] > df['B_following_count'])
    df['listed_diff'] = (df['A_listed_count'] > df['B_listed_count'])
    df['ment_rec_diff'] = (df['A_mentions_received'] > df['B_mentions_received'])
    df['rt_rec_diff'] = (df['A_retweets_received'] > df['B_retweets_received'])
    df['ment_sent_diff'] = (df['A_mentions_sent'] > df['B_mentions_sent'])
    df['rt_sent_diff'] = (df['A_retweets_sent'] > df['B_retweets_sent'])
    df['posts_diff'] = (df['A_posts'] > df['B_posts'])

    df['A_pop_ratio'] = df['A_mentions_sent'] / df['A_listed_count']
    df['A_foll_ratio'] = df['A_follower_count'] / df['A_following_count']
    df['A_ment_ratio'] = df['A_mentions_sent'] / df['A_mentions_received']
    df['A_rt_ratio'] = df['A_retweets_sent'] / df['A_retweets_received']

    df['B_pop_ratio'] = df['B_mentions_sent'] / df['B_listed_count']
    df['B_foll_ratio'] = df['B_follower_count'] / df['B_following_count']
    df['B_ment_ratio'] = df['B_mentions_sent'] / df['B_mentions_received']
    df['B_rt_ratio'] = df['B_retweets_sent'] / df['B_retweets_received']

    df['A/B_foll_ratio'] = (df['A_foll_ratio'] > df['B_foll_ratio'])
    df['A/B_ment_ratio'] = (df['A_ment_ratio'] > df['B_ment_ratio'])
    df['A/B_rt_ratio'] = (df['A_rt_ratio'] > df['B_rt_ratio'])

    df['nf1_diff'] = (df['A_network_feature_1'] > df['B_network_feature_1'])
    df['nf2_diff'] = (df['A_network_feature_2'] > df['B_network_feature_2'])
    df['nf3_diff'] = (df['A_network_feature_3'] > df['B_network_feature_3'])

    df['nf3_ratio'] = df['A_network_feature_3'] / df['B_network_feature_3']
    df['nf2_ratio'] = df['A_network_feature_2'] / df['B_network_feature_2']
    df['nf1_ratio'] = df['A_network_feature_1'] / df['B_network_feature_1']
    df=df.sort_values(by="A/B_foll_ratio")
    print(pre_pro(df))
    sirala=pd.DataFrame(pre_pro(df))
    if session_control8()==False:
        sirala.to_csv("sirala.csv")
    return (pre_pro(df))
fe_train = feat_eng(traindf.copy())
fe_test = feat_eng(testdf.copy())
train_df = fe_train
test_df = fe_test
y_train = np.array(train_df['Choice'])
target = 'Choice'
predictors = train_df.columns.values.tolist()[1:]

param_lgb = {
        'feature_fraction': 0.4647875434283183,
        'lambda_l1': 0.14487098904632512,
        'lambda_l2': 0.9546002933329684,
        'learning_rate': 0.050592093295320606,
        'max_depth': int(round(7.696194993998026)),
        'min_data_in_leaf': int(round(9.879507661608065)),
        'min_gain_to_split': 0.7998292013880356,
        'min_sum_hessian_in_leaf': 0.24962103361366683,
        'num_leaves': int(round(2.854239951949671)),
        'max_bin': 63,
        'bagging_fraction': 1.0,
        'bagging_freq': 5,
        'save_binary': True,
        'seed': 1965,
        'feature_fraction_seed': 1965,
        'bagging_seed': 1965,
        'drop_seed': 1965,
        'data_random_seed': 1965,
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'verbose': 1,
        'metric': 'auc',
        'is_unbalance': True,
        'boost_from_average': False}
nfold = 20
importances = rfc.feature_importances_
sorted_indices = np.argsort(importances)[::-1]
plt.title('Feature Importance')
plt.bar(range(X_train.shape[1]), importances[sorted_indices], align='center')
plt.xticks(range(X_train.shape[1]), X_train.columns[sorted_indices], rotation=90)
plt.tight_layout()
plt.show()
skf = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=2019)

oof = np.zeros(len(fe_train))
predictions = np.zeros((len(fe_test), nfold))
lgb_bay = []

for i in range(len(predictions)):
    lgb_bay.append(predictions[i][-1])

submission = pd.read_csv('sample_predictions.csv')
submission['Choice'] = lgb_bay

if session_control7()==False:
    submission.to_csv('sub.csv', index = False, header = True)
    print('Saved..')