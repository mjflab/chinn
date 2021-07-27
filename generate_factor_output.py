import os
import argparse
import h5py
import numpy as np
import torch
from chinn.models import PartialDeepSeaModel
from chinn import train


def get_args():
    parser = argparse.ArgumentParser('Perform test')
    parser.add_argument('model', help='The model')
    parser.add_argument('data_file', help='The data file')
    parser.add_argument('out_pre', help='The output file prefix')
    parser.add_argument('out_dir', help='The output directory')
    parser.add_argument('-s', '--sigmoid', action='store_true', default=False, help='use sigmoid after weightsum')
    parser.add_argument('-d', '--use_distance', action='store_true', default=False, help='use distance')
    parser.add_argument('--same', action='store_true', default=False, help='Use the same subsequence for all features')
    return parser.parse_args()


if __name__=='__main__':
    args = get_args()
    model = PartialDeepSeaModel(4, use_weightsum=True, leaky=True, use_sigmoid=args.sigmoid)
    model.load_state_dict(torch.load(args.model))
    model.cuda()
    model.eval()
    
    (train_data, train_left_data, train_right_data,
     train_left_edges, train_right_edges,
     train_labels, train_dists) = train.load_hdf5_data(args.data_file)

    data_store = h5py.File(os.path.join(args.out_dir, args.out_pre + '_factor_outputs.hdf5'), 'w')
    left_data_store = data_store.create_dataset('left_out', (len(train_labels), model.num_filters[-1] * 2),
                                                dtype='float32',
                                                chunks=True, compression='gzip')
    right_data_store = data_store.create_dataset('right_out', (len(train_labels), model.num_filters[-1] * 2),
                                                 dtype='float32',
                                                 chunks=True, compression='gzip')
    dist_data_store = data_store.create_dataset('dists', data=train_dists, dtype='float32',
                                                chunks=True, compression='gzip')
    pairs = train_data['pairs'][:]
    pair_dtype = ','.join('uint8,u8,u8,uint8,u8,u8,u8'.split(',') + ['uint8'] * (len(pairs[0]) - 7))
    pair_data_store = data_store.create_dataset('pairs', data=np.array(pairs, dtype=pair_dtype), chunks=True,
                       compression='gzip')

    labels_data_store = data_store.create_dataset('labels', data=train_labels, dtype='uint8')
    i = 0
    last_print = 0
    with torch.no_grad():
        while i < len(train_left_edges) - 1:
            end, left_out, right_out, _, _ = train.compute_factor_output(train_left_data, train_left_edges,
                                                                train_right_data,
                                                                train_right_edges,
                                                                train_dists, train_labels, i,
                                                                True, factor_model=model, max_size=2000, same=args.same)
            left_data_store[i:end] = left_out.data.cpu().numpy()
            right_data_store[i:end] = right_out.data.cpu().numpy()
            if end - last_print > 5000:
                last_print = end
                print('generating input : %d / %d' % (end, len(train_labels)))
            i = end
    data_store.close()

