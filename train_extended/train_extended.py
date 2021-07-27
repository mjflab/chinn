import os,sys
import argparse
import h5py
import glob
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve
import xgboost as xgb
from sklearn.externals import joblib
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
from chinn.epigenetic_model import train_estimator, get_val_results, test_other


def load_factor_outputs(fn):
    f = h5py.File(fn,'r')
    left_out = f['left_out'][:]
    right_out = f['right_out'][:]
    dists = f['dists'][:]
    labels = f['labels'][:]
    if 'pairs' in f:
        pairs = f['pairs'][:]
    else:
        pairs = None
    data = np.concatenate((left_out, right_out, dists), axis=1)
    return data, labels, pairs


if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Train classifiers using extended datasets.")
    parser.add_argument('data_dir', help='The directory of the data location with train, valid, and test.')
    parser.add_argument('dataset_name', help='The name (prefix) of the dataset before _[train|valid|test]_factor_outputs.hdf5.' + 
                                             ' For example: gm12878_ctcf_train_factor_outputs.hdf5 -> gm12878_ctcf')
    parser.add_argument('model_dir', help='The directory to store the models. 3 models will be generated: '
                        + '1. using all features; 2. using all features but distance (_nodist); '
                        + '3. using distance only (_dist_only).')

    args = parser.parse_args()
    name = args.dataset_name

    train_data, train_labels, _ = load_factor_outputs(os.path.join(args.data_dir, f'{name}_train_factor_outputs.hdf5'))
    val_data, val_labels, _ = load_factor_outputs(os.path.join(args.data_dir, f'{name}_valid_factor_outputs.hdf5'))

    test_data, test_labels, test_pairs = load_factor_outputs(os.path.join(args.data_dir, f'{name}_test_factor_outputs.hdf5'))

    for depth in [3, 6, 10]:
        estimator = train_estimator(train_data, train_labels, val_data, val_labels,
                                    max_depth=depth, threads=20, verbose_eval=True)
        estimator_0 = train_estimator(train_data[:, :-1], train_labels, val_data[:, :-1], val_labels,
                                      max_depth=depth, threads=20, verbose_eval=True)
        estimator_d = train_estimator(train_data[:, [-1]], train_labels, val_data[:, [-1]], val_labels,
                                      max_depth=depth, threads=20, verbose_eval=True)
        joblib.dump(estimator, os.path.join(args.model_dir, f"{name}_depth{depth}.gbt.pkl"))
        joblib.dump(estimator_0, os.path.join(args.model_dir, f"{name}_depth{depth}_nodist.gbt.pkl"))
        joblib.dump(estimator_d, os.path.join(args.model_dir, f"{name}_depth{depth}_dist_only.gbt.pkl"))
