import os,sys
import numpy as np
from chinn.seq_to_hdf5 import load_bed
from chinn.pair_features import load_pairs_as_dict, plot_dist_distr, generate_data, save_data_to_hdf5, load_data_from_hdf5
from chinn.epigenetic_model import train_estimator
from sklearn.externals import joblib
import argparse

if __name__=='__main__':

    parser = argparse.ArgumentParser('Train Epigenetic models')
    parser.add_argument('name', help='name of the dataset')
    parser.add_argument('datadir', help='directory of the pairs')
    parser.add_argument('--generate_from_pairs', default=False, action='store_true',
                        help='Whether to generate data to hdf5 files.')

    args = parser.parse_args()
    name = args.name

    num_bins = 50
    dist_range = (np.log10(5000), np.log10(2000000))
    binary=False
    type_str = 'common'
    if binary:
        type_str += '_binary'

    pos_pairs, pos_dists = load_pairs_as_dict(
        [os.path.join(args.datadir, f"{name}.clustered_interactions.both_dnase.bedpe")])

    selected_neg_pairs, selected_neg_dists = load_pairs_as_dict(
        [os.path.join(args.datadir, f'{name}.neg_pairs_5x.from_singleton_inter_tf_random.bedpe')])

    ann_files = []
    cell = name.split('_')[0]
    with open(os.path.join(args.datadir, f"{cell}_files_common.txt")) as f:
        for r in f:
            r = r.strip()
            if r[0] != "#":
                ann_files.append(r.strip())

    if args.generate_from_pairs:
        annotations = []
        for i in ann_files:
            annotations.append(load_bed(i))
        ((train_data, train_label, train_pairs), 
        (val_data, val_label, val_pairs), 
        (test_data, test_label, test_pairs)) = generate_data(pos_pairs, selected_neg_pairs, annotations, binary)
        save_data_to_hdf5(train_data, train_label, train_pairs, name, 'train', type_str)
        save_data_to_hdf5(val_data, val_label, val_pairs, name, 'valid', type_str)
        save_data_to_hdf5(test_data, test_label, test_pairs, name, 'test', type_str)
    else:
        train_data, train_label, train_pairs = load_data_from_hdf5(name, 'train', type_str)
        val_data, val_label, val_pairs = load_data_from_hdf5(name, 'valid', type_str)
        test_data, test_label, test_pairs = load_data_from_hdf5(name, 'test', type_str)

    for depth in [3,6,10]:
        print('==================depth :', depth,"=========================")
        estimator = train_estimator(train_data, train_label, val_data, val_label, max_depth=depth)
        estimator_0 = train_estimator(train_data[:,1:], train_label, val_data[:,1:], val_label, max_depth=depth)
        estimator_1 = train_estimator(train_data[:,1:1+len(ann_files)], train_label, val_data[:,1:1+len(ann_files)], val_label, max_depth=depth)
        estimator_2 = train_estimator(train_data[:,1+len(ann_files):], train_label, val_data[:,1+len(ann_files):], val_label, max_depth=depth)
        estimator_d = train_estimator(train_data[:,[0]], train_label, val_data[:,[0]], val_label, max_depth=depth)
        
        model_prefix = f'{name}_{type_str}_{depth}'
        joblib.dump(estimator, f"{model_prefix}.gbt.pkl")
        joblib.dump(estimator_0, f"{model_prefix}_nodist.gbt.pkl")
        joblib.dump(estimator_1, f"{model_prefix}_left_only.gbt.pkl")
        joblib.dump(estimator_2, f"{model_prefix}_right_only.gbt.pkl")
        joblib.dump(estimator_d, f"{model_prefix}_dist_only.gbt.pkl")
        print('\n\n')
