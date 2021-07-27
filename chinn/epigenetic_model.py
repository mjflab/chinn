import numpy as np
from sklearn.metrics import f1_score, precision_score, average_precision_score, recall_score, accuracy_score, roc_curve
from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.utils import shuffle
import pylab as pl
from chinn.common import col1Width, colors
import xgboost as xgb


def train_estimator(train_data, train_label, val_data, val_label, 
                    n_estimators=1000, threads=20, max_depth=6, verbose_eval=True):
    dtrain = xgb.DMatrix(train_data, label=train_label)
    dval = xgb.DMatrix(val_data, label=val_label)
    evallist = [(dtrain, 'train'), (dval, 'eval')]
    evals_result = {}
    params = {'max_depth': max_depth, 'objective': 'binary:logistic',
              'eta': 0.1, 'nthread': threads, 'eval_metric': ['aucpr', 'map', 'logloss']}
    bst = xgb.train(params, dtrain, n_estimators, evallist, early_stopping_rounds=40,
                    verbose_eval=verbose_eval, evals_result=evals_result)

    return bst


def get_val_results(estimator, data, labels, pairs, metrics=True, plot=False, name=None):

    dtest = xgb.DMatrix(data, labels)
    val_probs = estimator.predict(dtest, ntree_limit=estimator.best_ntree_limit)
    val_preds = np.zeros_like(val_probs)
    val_preds[val_probs >= 0.5] = 1
    if metrics:
        shuffled_labels = shuffle(labels)
        print('f1:', f1_score(labels, val_preds))
        print('precision:', precision_score(labels, val_preds))
        print('recall:', recall_score(labels, val_preds))
        print('aupr:', average_precision_score(labels, val_probs))
        print('roc:', roc_auc_score(labels, val_probs))
        print("\n")
        print('f1:', f1_score(labels, shuffled_labels))
        print('precision:', precision_score(labels, shuffled_labels))
        print('recall:', recall_score(labels, shuffled_labels))
        print('aupr:', average_precision_score(labels, shuffle(val_probs)))
        print('roc:', roc_auc_score(labels, shuffle(val_probs)))
        one_prec = precision_score(labels, np.ones(len(labels)))
        
        if plot:
            precision, recall, _ = precision_recall_curve(labels, val_probs)
            #print([(a,b) for a,b in zip(precision, recall)])
            fpr, tpr, _ = roc_curve(labels, val_probs)
            fig = pl.figure(figsize=(col1Width, col1Width))
            ax = fig.add_subplot(111)
            ax.plot(fpr, tpr, color=colors(0))
            ax.plot(recall, precision, color=colors(2))
            ax.axhline(one_prec, color='r')
            #pl.xlim(-0.05, 1.05)
    if name is not None:
        print(name)
        with open("{}_probs.txt".format(name), 'w') as out:
            for p,l,prob in zip(pairs, labels, val_probs):
                out.write("\t".join(map(str, list(p) + [l,prob])) + "\n")
    print('\n===============\n')
    return val_probs


def test_other(estimator, name, dset, type_str, mode='all', out_name=None):
    '''

    :param estimator:
    :param name:
    :param dset:
    :param type_str:
    :param mode: 3 modes, all: all features, epi: all but distance, dis: distance only
    :return:
    '''
    from pair_features import load_data_from_hdf5
    data, labels, pairs = load_data_from_hdf5(name, dset, type_str)
    if mode == 'epi':
        data = data[:, 1:]
    elif mode == 'dis':
        data = data[:, 0:1]
    get_val_results(estimator, data, labels, pairs, name=out_name)
