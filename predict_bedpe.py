import torch
import argparse
import h5py
import numpy as np
import xgboost as xgb
from sklearn.externals import joblib
from multiprocessing.pool import Pool
import sys
from chinn.common import check_chrom, chrom_to_int
from chinn.data_preparation_helper import get_one_side_data_parallel, load_pairs
from chinn.train import compute_one_side
from chinn.models import PartialDeepSeaModel
from chinn import variables


def get_args():
    parser = argparse.ArgumentParser('Perform prediction')
    parser.add_argument('-m', '--model_file', type=str, required=True, help='The model prefix')
    parser.add_argument('-c', '--classifier_file', type=str, required=True, help='The classifier')
    parser.add_argument('--pos_files', nargs='*', default=[], help='The positive files')
    parser.add_argument('--neg_files', nargs='*', default=[], help='The negative files')
    parser.add_argument('--output_pre', required=True, help='The output file prefix')
    parser.add_argument('-s', '--sigmoid', action='store_true', default=False,
                        help='use sigmoid after weightsum. Default: False')
    parser.add_argument('-d', '--use_distance', action='store_true', default=False,
                        help='use distance. Default: False')
    parser.add_argument('--same', action='store_true', default=False,
                        help='Use the same subsequence for all features. Default: False')
    parser.add_argument('--store_factor_outputs', action='store_true', default=False,
                        help='Whether to store the factor outputs. Default: False')
    parser.add_argument('--min_size', type=int, required=True, help='minimum size of anchors to use')
    parser.add_argument('-e', '--ext_size', type=int, required=False, help='extension size of anchors to use')
    parser.add_argument('-g', '--genome', default='/data/hg19all.fa', help='The fasta file of reference genome.')
    parser.add_argument('-b', '--batch_size', type=int, default=500, help='The batch size to use. Default: 500')
    parser.add_argument('--inter_chrom', action='store_true', default=False,
                        help='Whether the pairs are inter-chromosome')
    parser.add_argument('--breakpoints', nargs='*', type=str, default=[], help='Breakpoints locations.')
    parser.add_argument('--crispr', type=str, help='Breakpoints locations.')
    parser.add_argument('--no_classifier', action='store_true', default=False,
                        help="Do not invoke classifer, only factor outputs will be computed.")

    args = parser.parse_args()
    return args


def load_crispred(filename):
    if filename is None:
        return None
    regions = []
    with open(filename) as f:
        for r in f:
            tokens = r.strip().split()
            if not check_chrom(tokens[0]):
                continue
            tokens[0] = chrom_to_int(tokens[0])
            for i in [1,2]:
                tokens[i] = int(tokens[i])
            tokens = tuple(tokens)
            regions.append(tokens)
    regions.sort(key=lambda k: (k[0], k[1], k[2]))
    return regions


if __name__=='__main__':
    args = get_args()

    crispred = load_crispred(args.crispr)

    model = PartialDeepSeaModel(4, use_weightsum=True, leaky=True, use_sigmoid=args.sigmoid)
    model.load_state_dict(torch.load(args.model_file))
    model.cuda()
    model.eval()
    classifier = None if args.no_classifier else joblib.load(args.classifier_file)
    variables.init(args.genome)
    breakpoints = {}

    if args.inter_chrom:
        if len(args.breakpoints) < 1:
            print('No breakpoints specified')
            sys.exit(0)
        for b in args.breakpoints:
            chrom, point = b.split(':')
            chrom = 23 if chrom == 'chrX' else int(chrom.replace('chr', ''))
            point = int(point)
            breakpoints[chrom] = point

    train_pairs, train_labels, val_pairs, val_labels, test_pairs, test_labels = load_pairs(args.pos_files,
                                                                                           args.neg_files,
                                                                                           variables.hg19,
                                                                                           allow_inter=args.inter_chrom,
                                                                                           breakpoints=breakpoints)
    pairs = train_pairs + val_pairs + test_pairs
    labels = train_labels + val_labels + test_labels


    dists = []
    for p in pairs:
        if p[0] == p[3]:
            temp_d = [np.log10(abs(p[5] / 5000 - p[2] / 5000 + p[4] / 5000 - p[1] / 5000) * 0.5) / np.log10(2000001 / 5000),]
        else:
            temp_dl = 0.5*(p[1]+p[2]) - breakpoints[p[0]]
            temp_dr = 0.5*(p[4]+p[5]) - breakpoints[p[3]]
            temp_d = [np.log10((abs(temp_dl) + abs(temp_dr))/5000) / np.log10(2000001 / 5000),]
        dists.append(temp_d)

    probs = np.zeros(len(pairs))
    print('loaded data')
    

    if args.store_factor_outputs:
        data_store = h5py.File(args.output_pre + '_factor_outputs.hdf5', 'w')
        left_data_store = data_store.create_dataset('left_out', (len(labels), model.num_filters[-1] * 2),
                                                    dtype='float32',
                                                    chunks=True, compression='gzip')
        right_data_store = data_store.create_dataset('right_out', (len(labels), model.num_filters[-1] * 2),
                                                     dtype='float32',
                                                     chunks=True, compression='gzip')
        dist_data_store = data_store.create_dataset('dists', (len(labels),1), dtype='float32',
                                                    chunks=True, compression='gzip')

        pair_dtype = ','.join('uint8,u8,u8,uint8,u8,u8,u8'.split(',') + ['uint8'] * (len(pairs[0]) - 7))
        pair_data_store = data_store.create_dataset('pairs', data=np.array(pairs, dtype=pair_dtype), chunks=True,
                                                    compression='gzip')
        labels_data_store = data_store.create_dataset('labels', data=labels, dtype='uint8')

    i = 0
    last_print = 0
    batch_size = args.batch_size
    evaluation = True

    pool = Pool(processes=6)
    print('starting', len(labels))
    with torch.no_grad():
        for i in range(0, len(labels), batch_size):
            end = i+batch_size
            curr_left_data, curr_left_edges = get_one_side_data_parallel(pairs[i:end], pool, left=True, verbose=True,
                                                                         min_size=args.min_size, ext_size=args.ext_size,
                                                                         crispred=crispred)
            #print('left done')
            curr_right_data, curr_right_edges = get_one_side_data_parallel(pairs[i:end], pool, left=False, verbose=True,
                                                                           min_size=args.min_size, ext_size=args.ext_size,
                                                                           crispred=crispred)
            curr_labels = labels[i:i+batch_size]

            left_out = compute_one_side(curr_left_data, curr_left_edges, model, evaluation, same=args.same)
            right_out = compute_one_side(curr_right_data, curr_right_edges, model, evaluation, same=args.same)

            left_out = left_out.data.cpu().numpy()
            right_out = right_out.data.cpu().numpy()
            if not args.no_classifier:
                if args.use_distance:
                    curr_dists = dists[i:end]
                    #print(len(curr_left_edges), len(curr_right_edges), left_out.shape, right_out.shape, curr_dists.shape)
                    input_for_classifier = np.concatenate([left_out, right_out, curr_dists], axis=1)
                else:
                    input_for_classifier = np.concatenate([left_out, right_out], axis=1)
                probs[i:end] = classifier.predict(xgb.DMatrix(input_for_classifier),
                                                  ntree_limit=classifier.best_ntree_limit)
            if args.store_factor_outputs:
                left_data_store[i:end] = left_out
                right_data_store[i:end] = right_out
            if end - last_print > 5000:
                last_print = end
                print('generating input : %d / %d. With %d >= 0.5 so far' % (end, len(labels), sum(probs[:end]>=0.5)))
    if args.store_factor_outputs:
        data_store.close()

    if not args.no_classifier:
        with open(args.output_pre + '_probs.txt', 'w') as out:
            for pair, prob in zip(pairs, probs):
                pair_str = [str(p) for p in pair] + [str(prob)]
                for i in [0,3]:
                    if pair_str[i] == '23':
                        pair_str[i] = 'chrX'
                    else:
                        pair_str[i] = 'chr' + pair_str[i]
                out.write('\t'.join(pair_str) + '\n')
    pool.close()
    pool.join()



