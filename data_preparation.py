'''
This file contains functions to produce hdf5 files for training/validation/test.
The validation chroms are 5, 14
The test chromosomes are 4, 7, 8, 11
'''
import os,sys
import argparse
from chinn import variables
from chinn.data_preparation_helper import get_and_save_data, load_pairs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate hdf5 files for prediction.'
                                     + " The validation chromosomes are 5, 14."
                                     + " The test chromosomes are 4, 7, 8, 11. Rest will be training.")
    parser.add_argument('-m', '--min_size', type=int, required=True, help='minimum size of anchors to use')
    parser.add_argument('-e', '--ext_size', type=int, required=False, help='extension size of anchors to use')
    parser.add_argument('-n', '--name', type=str, required=True, help='The prefix of the files.')
    parser.add_argument('-g', '--genome', required=True, help='The fasta file of reference genome.')
    parser.add_argument('-o', '--out_dir', type=str, required=True, help='The output directory.')
    parser.add_argument('--pos_files', nargs='*', default=[], help='The positive files')
    parser.add_argument('--neg_files', nargs='*', default=[], help='The negative files')
    parser.add_argument('-t', '--all_test', action='store_true', default=False, help='Use all data for test.')
    parser.add_argument('--out_test_only', action='store_true', default=False,
                        help='Produce only the test data but not validation and training data.'
                             + ' Will be ignored if -t/--all_test is set.')
    parser.add_argument('--no_test', action='store_true', default=False,
                        help='Produce no test data but validation and training data.'
                             + ' Will be ignored if -t/--all_test or --out_test_only is set.')
    args = parser.parse_args()
    min_size = args.min_size
    
    variables.init(args.genome)
    name = args.name

    if len(args.pos_files) <= 0 and len(args.neg_files) <= 0:
        print('Nothing to do')
        sys.exit(0)
    dataset_names = ['train', 'valid', 'test']
    train_pairs, train_labels, val_pairs, val_labels, test_pairs, test_labels = load_pairs(args.pos_files,
                                                                                           args.neg_files,
                                                                                           variables.hg19)
    data_pairs = [train_pairs, val_pairs, test_pairs]
    data_labels = [train_labels, val_labels, test_labels]
    out_idxes = []
    if args.all_test:
        pairs = train_pairs + val_pairs + test_pairs
        labels = train_labels + val_labels + test_labels
        fn = os.path.join(args.out_dir, '{}_test.hdf5'.format(name))
        print(fn)
        get_and_save_data(pairs, labels, fn, min_size, ext_size=args.ext_size)
    elif args.out_test_only:
        out_idxes.append(2)
    elif args.no_test:
        out_idxes += [0, 1]
    else:
        out_idxes = [0, 1, 2]

    for idx in out_idxes:
        pairs = data_pairs[idx]
        labels = data_labels[idx]
        dset = dataset_names[idx]
        fn = os.path.join(args.out_dir,
                          "{}_singleton_tf_with_random_neg_seq_data_length_filtered_{}.hdf5".format(name, dset))
        print(fn)
        get_and_save_data(pairs, labels, fn, min_size, ext_size=args.ext_size)
