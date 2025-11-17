#!/usr/bin/env python
# coding: utf-8

# ## ML for Classification
# 
# source dataset: https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset

# ### Preparation data

# In[112]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


# In[113]:


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression


# In[114]:


df = pd.read_csv('data/customer_churn_dataset-testing-master.csv')


# In[115]:


df


# In[116]:


df.describe(include='all').T


# In[117]:


df.columns = df.columns.str.lower().str.replace(' ', '_')

categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)

for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(' ', '_')


# In[118]:


df


# In[119]:


df.isnull().sum()


# In[120]:


df.nunique()


# In[121]:


df.dtypes


# In[122]:


df.info()


# ### Setting up the validation framework
# 
# Perform the train/validation/test split with Scikit-Learn

# In[123]:


from sklearn.model_selection import train_test_split


# In[124]:


df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)


# In[125]:


len(df_train), len(df_val), len(df_test)


# In[126]:


len(df_train) + len(df_val)+ len(df_test)


# In[127]:


y_train = df_train.churn.values
y_val = df_val.churn.values
y_test = df_test.churn.values

del df_train['churn']
del df_val['churn']
del df_test['churn']


# ### EDA
# 
# - Check missing values
# - Look at the target variable (churn)
# - Look at numerical and categorical variable

# In[128]:


df_full_train.head(n=3)


# In[129]:


df_train.head(n=3)


# In[130]:


y_train[:3]


# In[131]:


df_full_train = df_full_train.reset_index(drop=True)


# In[132]:


df_full_train.head()


# In[133]:


df_full_train.isnull().sum()


# In[134]:


df_full_train.churn.value_counts(normalize=True)


# In[135]:


df_full_train.churn.mean()


# In[136]:


df_full_train.info()


# In[137]:


numerical = ['age','tenure','usage_frequency','support_calls','payment_delay','total_spend','last_interaction']
numerical


# In[138]:


categorical = ['gender','subscription_type','contract_length']
categorical


# In[139]:


df_full_train[categorical].nunique()


# #### Feature importance: Churn rate and risk ratio
# 
# Feature importance analysis (part of EDA) - identifying which features affect our target variable
# 
# - Churn rate
# - Risk ratio
# - Mutual information - later

# ##### Churn rate

# In[140]:


df_full_train.head()


# In[141]:


df_full_train.churn.value_counts(normalize=True)


# In[142]:


df_full_train.churn.mean()


# In[143]:


churn_female = df_full_train[df_full_train.gender == 'female'].churn.mean()
churn_female


# In[144]:


churn_male = df_full_train[df_full_train.gender == 'male'].churn.mean()
churn_male


# In[145]:


global_churn = df_full_train.churn.mean()
global_churn


# In[146]:


global_churn - churn_female


# In[147]:


for c in categorical:
    print(c)
    df_group = df_full_train.groupby(c).churn.agg(['mean', 'count'])
    df_group['diff'] = df_group['mean'] - global_churn
    df_group['risk'] = df_group['mean'] / global_churn
    display(df_group)
    print()
    print()


# ####  Feature importance: Mutual information
# 
# Mutual information - concept from information theory, it tells us how much we can learn about one variable if we know the value of another

# In[148]:


from sklearn.metrics import mutual_info_score


# In[149]:


mutual_info_score(df_full_train.churn, df_full_train.gender)


# In[150]:


mutual_info_score(df_full_train.churn, df_full_train.subscription_type)


# In[151]:


mutual_info_score(df_full_train.churn, df_full_train.contract_length)


# In[152]:


def mutual_info_churn_score(series):
    return mutual_info_score(series, df_full_train.churn)


# In[153]:


mi = df_full_train[categorical].apply(mutual_info_churn_score)
mi.sort_values(ascending=False)


# #### Feature importance: Correlation

# In[154]:


numerical = ['age','tenure','usage_frequency','support_calls','payment_delay','total_spend','last_interaction']
numerical


# In[155]:


df_full_train.tenure.max()


# In[156]:


# Tính hệ số tương quan Pearson giữa mỗi cột numerical và biến mục tiêu churn, chú ý lấy giá trị tuyệt đối vì có .abs

df_full_train[numerical].corrwith(df_full_train.churn).abs()


# In[157]:


df_full_train[df_full_train.tenure <= 2].churn.mean()


# In[158]:


# Tỷ lệ churn trung bình của nhóm khách hàng có thời gian sử dụng từ 3 đến 12 tháng

df_full_train[(df_full_train.tenure > 2) & (df_full_train.tenure <= 12)].churn.mean()


# In[159]:


# Tỷ lệ churn trung bình của nhóm khách hàng có thời gian sử dụng dịch vụ trên 12 tháng.

df_full_train[df_full_train.tenure > 12].churn.mean()


# In[160]:


df_full_train.payment_delay.max()


# ####  One-hot encoding
# Use Scikit-Learn to encode categorical features

# In[161]:


# Loại bỏ các thuộc tính không quan trọng: subscription_type và last_interaction

categorical = ['gender', 'contract_length']
numerical = ['age', 'tenure', 'usage_frequency', 'support_calls', 'payment_delay', 'total_spend']


# In[163]:


from sklearn.feature_extraction import DictVectorizer


# In[164]:


dv = DictVectorizer(sparse=False)

train_dict = df_train[categorical + numerical].to_dict(orient='records')
X_train = dv.fit_transform(train_dict)

val_dict = df_val[categorical + numerical].to_dict(orient='records')
X_val = dv.transform(val_dict)


# In[165]:


X_train.shape


# In[167]:


X_val.shape


# In[173]:


set(dv.feature_names_)


# ####  Modeelling with Logistic regression

# In[174]:


from sklearn.linear_model import LogisticRegression


# In[175]:


model = LogisticRegression(solver='lbfgs')

model.fit(X_train, y_train)


# In[176]:


model.intercept_[0]


# In[177]:


model.coef_[0].round(3)


# In[178]:


y_pred = model.predict_proba(X_val)[:, 1]


# In[179]:


y_pred


# In[180]:


y_pred.shape


# In[181]:


churn_decision = (y_pred >= 0.5)


# In[182]:


churn_decision.shape


# In[183]:


(y_val == churn_decision).mean()


# In[184]:


df_pred = pd.DataFrame()
df_pred['probability'] = y_pred
df_pred['prediction'] = churn_decision.astype(int)
df_pred['actual'] = y_val


# In[185]:


df_pred.head(n=5)


# In[186]:


df_pred['correct'] = df_pred.prediction == df_pred.actual


# In[187]:


df_pred.head(n=15)


# In[188]:


df_pred.correct.mean()


# In[189]:


churn_decision.astype(int)


# #### Using model

# In[192]:


dicts_full_train = df_full_train[categorical + numerical].to_dict(orient='records')


# In[214]:


dicts_full_train[:2]


# In[202]:


len(dicts_full_train)


# In[203]:


dv = DictVectorizer(sparse=False)
X_full_train = dv.fit_transform(dicts_full_train)


# In[204]:


y_full_train = df_full_train.churn.values


# In[205]:


model = LogisticRegression(solver='lbfgs')
model.fit(X_full_train, y_full_train)


# In[206]:


dicts_test = df_test[categorical + numerical].to_dict(orient='records')


# In[207]:


X_test = dv.transform(dicts_test)


# In[208]:


y_pred = model.predict_proba(X_test)[:, 1]


# In[209]:


y_pred


# In[210]:


y_pred.shape


# In[211]:


churn_decision = (y_pred >= 0.5)


# In[212]:


(churn_decision == y_test).mean()


# In[223]:


y_test


# In[224]:


customer = dicts_test[1]
customer


# In[225]:


X_small = dv.transform([customer])


# In[226]:


X_small


# In[227]:


model.predict_proba(X_small)[0, 1]


# In[228]:


y_test[1]


# ### Evaluation Metrics for Classification

# In[232]:


from sklearn.metrics import accuracy_score


# In[233]:


accuracy_score(y_val, y_pred >= 0.5)


# In[234]:


thresholds = np.linspace(0, 1, 21) 

scores = []

for t in thresholds:
    score = accuracy_score(y_val, y_pred >= t)
    print('%.2f %.3f' % (t, score))
    scores.append(score)


# In[235]:


plt.plot(thresholds, scores)


# In[236]:


from collections import Counter


# In[237]:


y_pred


# In[240]:


len(y_pred)


# In[241]:


Counter(y_pred >= 1.0)


# In[242]:


y_val


# In[243]:


len(y_val)


# In[244]:


1 - y_val.mean()


# #### Confusion table

# In[245]:


actual_positive = (y_val == 1)
actual_negative = (y_val == 0)


# In[246]:


t = 0.5
predict_positive = (y_pred >= t)
predict_negative = (y_pred < t)


# In[247]:


tp = (predict_positive & actual_positive).sum()
tn = (predict_negative & actual_negative).sum()

fp = (predict_positive & actual_negative).sum()
fn = (predict_negative & actual_positive).sum()


# In[248]:


confusion_matrix = np.array([
    [tn, fp],
    [fn, tp]
])
confusion_matrix


# In[249]:


(confusion_matrix / confusion_matrix.sum()).round(3)


# #### Precision and Recal

# In[250]:


p = tp / (tp + fp)
p


# In[251]:


r = tp / (tp + fn)
r


# #### ROC Curves

# In[252]:


tpr = tp / (tp + fn)
tpr


# In[253]:


fpr = fp / (fp + tn)
fpr


# In[254]:


scores = []

thresholds = np.linspace(0, 1, 101)

for t in thresholds:
    actual_positive = (y_val == 1)
    actual_negative = (y_val == 0)
    
    predict_positive = (y_pred >= t)
    predict_negative = (y_pred < t)

    tp = (predict_positive & actual_positive).sum()
    tn = (predict_negative & actual_negative).sum()

    fp = (predict_positive & actual_negative).sum()
    fn = (predict_negative & actual_positive).sum()
    
    scores.append((t, tp, fp, fn, tn))


# In[255]:


scores


# In[256]:


columns = ['threshold', 'tp', 'fp', 'fn', 'tn']
df_scores = pd.DataFrame(scores, columns=columns)

df_scores['tpr'] = df_scores.tp / (df_scores.tp + df_scores.fn)
df_scores['fpr'] = df_scores.fp / (df_scores.fp + df_scores.tn)


# In[257]:


plt.plot(df_scores.threshold, df_scores['tpr'], label='TPR')
plt.plot(df_scores.threshold, df_scores['fpr'], label='FPR')
plt.legend()


# #### Random model

# In[258]:


np.random.seed(1)
y_rand = np.random.uniform(0, 1, size=len(y_val))


# In[259]:


y_rand


# In[260]:


((y_rand >= 0.5) == y_val).mean()


# In[261]:


def tpr_fpr_dataframe(y_val, y_pred):
    scores = []

    thresholds = np.linspace(0, 1, 101)

    for t in thresholds:
        actual_positive = (y_val == 1)
        actual_negative = (y_val == 0)

        predict_positive = (y_pred >= t)
        predict_negative = (y_pred < t)

        tp = (predict_positive & actual_positive).sum()
        tn = (predict_negative & actual_negative).sum()

        fp = (predict_positive & actual_negative).sum()
        fn = (predict_negative & actual_positive).sum()

        scores.append((t, tp, fp, fn, tn))

    columns = ['threshold', 'tp', 'fp', 'fn', 'tn']
    df_scores = pd.DataFrame(scores, columns=columns)

    df_scores['tpr'] = df_scores.tp / (df_scores.tp + df_scores.fn)
    df_scores['fpr'] = df_scores.fp / (df_scores.fp + df_scores.tn)
    
    return df_scores


# In[262]:


df_rand = tpr_fpr_dataframe(y_val, y_rand)


# In[263]:


df_rand


# In[264]:


plt.plot(df_rand.threshold, df_rand['tpr'], label='TPR')
plt.plot(df_rand.threshold, df_rand['fpr'], label='FPR')
plt.legend()


# In[265]:


num_neg = (y_val == 0).sum()
num_pos = (y_val == 1).sum()
num_neg, num_pos


# In[266]:


y_ideal = np.repeat([0, 1], [num_neg, num_pos])
y_ideal


# In[267]:


y_ideal_pred = np.linspace(0, 1, len(y_val))


# In[268]:


y_ideal_pred


# In[269]:


1 - y_val.mean()


# In[270]:


accuracy_score(y_ideal, y_ideal_pred >= 0.726)


# In[271]:


df_ideal = tpr_fpr_dataframe(y_ideal, y_ideal_pred)
df_ideal[::10]


# In[272]:


plt.plot(df_ideal.threshold, df_ideal['tpr'], label='TPR')
plt.plot(df_ideal.threshold, df_ideal['fpr'], label='FPR')
plt.legend()


# ##### Putting everything together

# In[273]:


plt.plot(df_scores.threshold, df_scores['tpr'], label='TPR', color='black')
plt.plot(df_scores.threshold, df_scores['fpr'], label='FPR', color='blue')

plt.plot(df_ideal.threshold, df_ideal['tpr'], label='TPR ideal')
plt.plot(df_ideal.threshold, df_ideal['fpr'], label='FPR ideal')

plt.plot(df_rand.threshold, df_rand['tpr'], label='TPR random', color='grey')
plt.plot(df_rand.threshold, df_rand['fpr'], label='FPR random', color='grey')

plt.legend()


# In[276]:


plt.figure(figsize=(7, 7))

plt.plot(df_scores.fpr, df_scores.tpr, label='Model')
plt.plot([0, 1], [0, 1], label='Random', linestyle='--')

plt.xlabel('FPR')
plt.ylabel('TPR')

plt.legend()


# In[277]:


from sklearn.metrics import roc_curve


# In[278]:


fpr, tpr, thresholds = roc_curve(y_val, y_pred)


# In[280]:


plt.figure(figsize=(7, 7))

plt.plot(fpr, tpr, label='Model')
plt.plot([0, 1], [0, 1], label='Random', linestyle='--')

plt.xlabel('FPR')
plt.ylabel('TPR')

plt.legend()


# ##### ROC AUC

# In[281]:


from sklearn.metrics import auc


# In[282]:


auc(fpr, tpr)


# In[283]:


auc(df_scores.fpr, df_scores.tpr)


# In[284]:


auc(df_ideal.fpr, df_ideal.tpr)


# In[285]:


fpr, tpr, thresholds = roc_curve(y_val, y_pred)
auc(fpr, tpr)


# In[286]:


from sklearn.metrics import roc_auc_score


# In[287]:


roc_auc_score(y_val, y_pred)


# In[288]:


neg = y_pred[y_val == 0]
pos = y_pred[y_val == 1]


# In[289]:


import random


# In[290]:


n = 100000
success = 0 

for i in range(n):
    pos_ind = random.randint(0, len(pos) - 1)
    neg_ind = random.randint(0, len(neg) - 1)

    if pos[pos_ind] > neg[neg_ind]:
        success = success + 1

success / n


# In[291]:


n = 50000

np.random.seed(1)
pos_ind = np.random.randint(0, len(pos), size=n)
neg_ind = np.random.randint(0, len(neg), size=n)

(pos[pos_ind] > neg[neg_ind]).mean()


# ### Cross-Validation

# In[292]:


def train(df_train, y_train, C=1.0):
    dicts = df_train[categorical + numerical].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(X_train, y_train)
    
    return dv, model


# In[293]:


dv, model = train(df_train, y_train, C=0.001)


# In[294]:


def predict(df, dv, model):
    dicts = df[categorical + numerical].to_dict(orient='records')

    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred


# In[313]:


C = 1.0
n_splits = 5


# In[314]:


y_pred = predict(df_val, dv, model)


# In[315]:


from sklearn.model_selection import KFold

from tqdm.auto import tqdm


# In[316]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)


# In[301]:


n_splits = 5

for C in tqdm([0.001, 0.01, 0.1, 0.5, 1, 5, 10]):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

    scores = []

    for train_idx, val_idx in kfold.split(df_full_train):
        df_train = df_full_train.iloc[train_idx]
        df_val = df_full_train.iloc[val_idx]

        y_train = df_train.churn.values
        y_val = df_val.churn.values

        dv, model = train(df_train, y_train, C=C)
        y_pred = predict(df_val, dv, model)

        auc = roc_auc_score(y_val, y_pred)
        scores.append(auc)

    print('C=%s %.3f +- %.3f' % (C, np.mean(scores), np.std(scores)))


# In[302]:


scores


# In[306]:


dv, model = train(df_full_train, df_full_train.churn.values, C=1.0)
y_pred = predict(df_test, dv, model)


auc = roc_auc_score(y_test, y_pred)
auc


# #### Save the model

# In[307]:


import pickle


# In[317]:


output_file = f'model_C={C}.bin'
output_file


# In[318]:


f_out = open('output_file', 'wb')
pickle.dump((dv, model), f_out)
f_out.close()


# In[319]:


with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)




# #### Load the model

# In[320]:


import pickle


# In[323]:


model_file = 'model_C=1.0.bin'


# In[324]:


with open(model_file, 'rb') as f_in:
    (dv, model) = pickle.load(f_in)


# In[325]:


dv, model


# In[326]:


customer = {
    'gender': 'male',
    'contract_length': 'annual',
    'age': 50,
    'tenure': 24,
    'usage_frequency': 30,
    'support_calls': 10,
    'payment_delay': 7,
    'total_spend': 530
}


# In[328]:


X = dv.transform([customer])


# In[329]:


model.predict_proba(X)[0, 1]


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




