
# coding: utf-8

# In[1]:


import os,sys
os.chdir('/home/caofan/prediction/from_dnase/')


# In[2]:


sys.path.append('/home/caofan/repos/chinn/')


# In[3]:


import h5py
import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from sklearn.externals import joblib
import matplotlib.pyplot as plt


# In[4]:


def load_factor_outputs(fn):
    f = h5py.File(fn,'r')
    left_out = f['left_out'][:]
    right_out = f['right_out'][:]
    dists = f['dists'][:]
    labels = f['labels'][:]
    if 'pairs' in f:
        pairs = f['pairs'][:]
        dists = [[np.log10(abs(p[5]-p[2] + p[4]-p[1])/5000*0.5) / np.log10(2000001 / 5000)] for p in pairs]
    else:
        pairs = None
    data = np.concatenate((left_out, right_out, dists), axis=1)
    return data, labels, pairs


# In[5]:


import xgboost as xgb


# In[6]:


def train_estimator(train_data, train_label, val_data, val_label, 
                    n_estimators=1000, threads=20, max_depth=6, verbose_eval=True):
    dtrain = xgb.DMatrix(train_data, label=train_label)
    dval = xgb.DMatrix(val_data, label=val_label)
    evallist = [(dtrain, 'train'), (dval, 'eval')]
    evals_result = {}
    params = {'max_depth': max_depth, 'objective': 'binary:logistic', 'max_delta_step':1,
              'eta': 0.1, 'nthread': threads, 'eval_metric': ['aucpr', 'map']}
    bst = xgb.train(params, dtrain, n_estimators, evallist, early_stopping_rounds=40,
                    verbose_eval=verbose_eval, evals_result=evals_result)

    return bst


# # POL2

# In[7]:


train_data, train_labels, train_pairs = load_factor_outputs('pol2/gm12878_polr2a.merged_3000_extended_500.train_factor_outputs.hdf5')


# In[8]:


val_data,val_labels,val_pairs = load_factor_outputs('pol2/gm12878_polr2a.merged_3000_extended_500.val_factor_outputs.hdf5')


# In[9]:


plt.hist(train_data[:,-1])


# In[10]:


classifier = train_estimator(train_data, train_labels, val_data, val_labels, 
                             n_estimators=1000, threads=40, max_depth=6, verbose_eval=True)


# In[10]:


classifier_d = train_estimator(train_data[:,-1:], train_labels, val_data[:,-1:], val_labels, 
                             n_estimators=1000, threads=20, max_depth=3, verbose_eval=True)


# In[11]:


joblib.dump(classifier, "gm12878_polr2a.merged3000_e500.gbt.pkl")
joblib.dump(classifier_d, "gm12878_polr2a.merged3000_e500.dist_only.gbt.pkl")


# In[33]:


classifier = joblib.load('gm12878_polr2a.merged3000_e500.gbt.pkl')


# In[34]:


test_data,test_labels,test_pairs = load_factor_outputs('pol2/gm12878_polr2a.merged_3000_extended_500.test_factor_outputs.hdf5')


# In[35]:


test_pred = classifier.predict(xgb.DMatrix(test_data), ntree_limit=classifier.best_ntree_limit)


# In[36]:


average_precision_score(test_labels, test_pred)


# In[51]:


from sklearn.metrics import f1_score
scores = []
recalls = []
precisions = []
for i in range(100):
    t = test_pred > 0.01*i
    scores.append(f1_score(test_labels, t))
    recalls.append(recall_score(test_labels, t))
    precisions.append(precision_score(test_labels, t))
print(np.argmax(scores), max(scores))


# In[55]:


plt.plot(scores, label='f1')
plt.plot(recalls, label='recall')
plt.plot(precisions, label='precision')
plt.legend()


# # CTCF

# In[15]:


train_data, train_labels, train_pairs = load_factor_outputs('ctcf/gm12878_ctcf.merged_3000_extended_1000.train_factor_outputs.hdf5')
val_data,val_labels,val_pairs = load_factor_outputs('ctcf/gm12878_ctcf.merged_3000_extended_1000.val_factor_outputs.hdf5')


# In[16]:


classifier = train_estimator(train_data, train_labels, val_data, val_labels, 
                             n_estimators=1000, threads=20, max_depth=6, verbose_eval=True)


# In[17]:


classifier_d = train_estimator(train_data[:,-1:], train_labels, val_data[:,-1:], val_labels, 
                             n_estimators=1000, threads=20, max_depth=3, verbose_eval=True)


# In[18]:


joblib.dump(classifier, "gm12878_ctcf.merged3000_e1000.gbt.pkl")
joblib.dump(classifier_d, "gm12878_ctcf.merged3000_e1000.dist_only.gbt.pkl")


# In[56]:


classifier = joblib.load('gm12878_ctcf.merged3000_e1000.gbt.pkl')


# In[57]:


test_data,test_labels,test_pairs = load_factor_outputs('ctcf/gm12878_ctcf.merged_3000_extended_1000.test_factor_outputs.hdf5')


# In[58]:


test_pred = classifier.predict(xgb.DMatrix(test_data), ntree_limit=classifier.best_ntree_limit)


# In[59]:


average_precision_score(test_labels, test_pred)


# In[60]:


sum(test_labels)/len(test_labels)


# In[61]:


from sklearn.metrics import f1_score
scores = []
recalls = []
precisions = []
for i in range(100):
    t = test_pred > 0.01*i
    scores.append(f1_score(test_labels, t))
    recalls.append(recall_score(test_labels, t))
    precisions.append(precision_score(test_labels, t))
print(np.argmax(scores), max(scores))


# In[68]:


print(np.argmax(scores), max(scores), scores[26], precisions[26], recalls[26])


# In[63]:


plt.plot(scores, label='f1')
plt.plot(recalls, label='recall')
plt.plot(precisions, label='precision')
plt.legend()


# In[71]:


max(test_pred)

