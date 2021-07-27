'''
This file contains functions to produce hdf5 files for training/validation/test.
The validation chroms are 5, 14
The test chromosomes are 4, 7, 8, 11
'''
from sklearn.utils import shuffle
import numpy as np
import h5py
from functools import partial
from chinn.common import check_chrom, chrom_to_int
from chinn.pair_features import encode_seq


def load_peaks(fn):
    peaks = {}
    with open(fn) as f:
        for r in f:
            tokens = r.strip().split()
            for i in [1,2]:
                tokens[i] = int(tokens[i])
            if tokens[0] not in peaks:
                peaks[tokens[0]] = []
            peaks[tokens[0]].append(tokens)
    for c in peaks:
        peaks[c].sort(key=lambda k:(k[1], k[2]))
    return peaks


def check_peaks(peaks, chrom, start, end):
    if chrom not in peaks:
        return False
    for p in peaks[chrom]:
        if min(end, p[2]) - max(start, p[1]) > 0:
            return True
    return False


def check_all_peaks(peaks_list, chrom, start, end):
    return [1 if check_peaks(peaks, chrom,start,end) else 0 for peaks in peaks_list]


def _load_data(fn, hg19, label,
               train_pairs, train_labels, val_pairs, val_labels,
               test_pairs, test_labels, peaks_list, allow_inter=False, breakpoints={}):
    int_cols = [1, 2, 4, 5]
    chrom_cols = [0, 3]
    val_chroms = [5, 14]
    test_chroms = [4, 11, 7, 8]
    with open(fn) as f:
        for r in f:
            tokens = r.strip().split()
            if not check_chrom(tokens[0]):
                continue

            for i in chrom_cols:
                tokens[i] = chrom_to_int(tokens[i])

            for i in int_cols:
                tokens[i] = int(tokens[i])

            if tokens[0] >= len(hg19) or tokens[3] >= len(hg19):
                #print(tokens[0], tokens[3])
                continue

            if not allow_inter and tokens[0] != tokens[3]:
                print('skipping different chrom ', tokens)
                continue
            elif allow_inter and tokens[0] != tokens[3]:
                #check distances
                if not (tokens[0] in breakpoints and tokens[3] in breakpoints):
                    continue
                temp_dl = 0.5 * (tokens[1] + tokens[2]) - breakpoints[tokens[0]]
                temp_dr = 0.5 * (tokens[4] + tokens[5]) - breakpoints[tokens[3]]
                #proper translocation, different sides of the translocation breakpoints
                if temp_dl * temp_dr > 0 or not (5000<=abs(temp_dl) + abs(temp_dr)<=2000000):
                    print('distance issues for different chromosome')
                    continue

                #change chromosome order
                if tokens[0] > tokens[3]:
                    temp = tokens[3:6]
                    tokens[3:6] = tokens[:3]
                    tokens[:3] = temp

            if tokens[1] > tokens[4]:
                temp1,temp2,temp3 = tokens[3:6]
                tokens[3], tokens[4], tokens[5] = tokens[0], tokens[1], tokens[2]
                tokens[0], tokens[1], tokens[2] = temp1, temp2, temp3

            if (tokens[1] < 0 or tokens[4] < 0 or
                tokens[2] >= len(hg19[tokens[0]]) or
                tokens[5] > len(hg19[tokens[3]]) or
                (tokens[0] == tokens[3] and tokens[4] < tokens[2])):
                print('skipping', tokens)
                continue

            if (tokens[0] != tokens[3]) or (tokens[0] == tokens[3]
                      and 5000. <= 0.5 * (tokens[4] - tokens[1] + tokens[5] - tokens[2]) <= 2000000):
                if len(tokens) < 7:
                    tokens.append(label)
                else:
                    tokens[6] = int(float(tokens[6]))
                if peaks_list is not None:
                    temp_peaks = check_all_peaks(peaks_list, *tokens[:3]) + check_all_peaks(peaks_list, *tokens[3:6])
                    tokens += temp_peaks

                tokens = tuple(tokens)
                if tokens[0] in val_chroms:
                    val_pairs.append(tokens)
                    val_labels.append(label)
                elif tokens[0] in test_chroms:
                    test_pairs.append(tokens)
                    test_labels.append(label)
                else:
                    train_pairs.append(tokens)
                    train_labels.append(label)


def load_pairs(pos_files, neg_files, hg19, peaks_list=None, allow_inter=False, breakpoints={}):
    train_pairs = []
    train_labels = []
    val_pairs = []
    val_labels = []
    test_pairs = []
    test_labels = []

    for fn in pos_files:
        _load_data(fn, hg19, 1,
                   train_pairs, train_labels,
                   val_pairs, val_labels,
                   test_pairs, test_labels, peaks_list, allow_inter, breakpoints)

    for fn in neg_files:
        _load_data(fn, hg19, 0,
                   train_pairs, train_labels,
                   val_pairs, val_labels,
                   test_pairs, test_labels, peaks_list, allow_inter, breakpoints)

    train_pairs, train_labels = shuffle(train_pairs, train_labels)
    val_pairs, val_labels = shuffle(val_pairs, val_labels)
    test_pairs, test_labels = shuffle(test_pairs, test_labels)
    return train_pairs, train_labels, val_pairs, val_labels, test_pairs, test_labels


def __get_mat(p, left, min_size, ext_size, crispred=None):
    if left:
        chrom, start, end = (0, 1, 2)
    else:
        chrom, start, end = (3, 4, 5)
    curr_chrom = p[chrom]
    
    if ext_size is not None:
        min_size = p[end]-p[start] + 2*ext_size
    temp = encode_seq(curr_chrom, p[start], p[end], min_size=min_size, crispred=crispred)
    if temp is None:
        raise ValueError('Nong value for matrix')
    return temp
    

def get_one_side_data_parallel(pairs, pool, left=True, out=None, verbose=False,
                               min_size=1000, ext_size=None, crispred=None):
    edges = [0]
    data = pool.map(partial(__get_mat, left=left, min_size=min_size, ext_size=ext_size, crispred=crispred), pairs)
    for d in data:
        edges.append(d.shape[0] + edges[-1])

    return np.concatenate(data, axis=0), edges


def get_one_side_data(pairs, left=True, out=None, verbose=False, min_size=1000, ext_size=None, crispred=None):
    if out is not None:
        data_name = "left_data" if left else "right_data"
        data_store = out.create_dataset(data_name, (50000, 4, 1000), dtype='uint8', maxshape=(None, 4, 1000),
                                        chunks=True, compression='gzip')
    if left:
        chrom, start, end = (0, 1, 2)
    else:
        chrom, start, end = (3, 4, 5)
    edges = [0]
    data = []
    last_cut = 0

    for p in pairs:
        curr_chrom = p[chrom]
        if type(curr_chrom) == int:
            if curr_chrom == 23:
                curr_chrom = 'chrX'
            else:
                curr_chrom = 'chr%d' % curr_chrom
        if ext_size is not None:
            min_size = p[end]-p[start] + 2*ext_size
        temp = encode_seq(p[chrom], p[start], p[end], min_size=min_size, crispred=crispred)
        if temp is None:
            raise ValueError('Nong value for matrix')
        new_cut = edges[-1] + temp.shape[0]
        data.append(temp)
        edges.append(new_cut)
        if out is not None and new_cut - last_cut > 50000:
            data_store.resize((edges[-1], 4, 1000))
            data_store[last_cut:edges[-1]] = np.concatenate(data, axis=0)
            data = []
            last_cut = edges[-1]
            if verbose:
                print(last_cut, len(edges))
    if out is not None:
        data_store.resize((edges[-1], 4, 1000))
        data_store[last_cut:edges[-1]] = np.concatenate(data, axis=0)
        edge_name = 'left_edges' if left else 'right_edges'
        out.create_dataset(edge_name, data=edges, dtype='long', chunks=True, compression='gzip')
    else:
        return np.concatenate(data, axis=0), edges


def get_and_save_data(pairs, labels, filename, min_size, ext_size=None, crispred=None):
    print('using ext_size: ', ext_size)
    with h5py.File(filename, 'w') as out:
        pair_dtype = ','.join('uint8,u8,u8,uint8,u8,u8,u8'.split(',') + ['uint8'] * (len(pairs[0]) - 7))
        out.create_dataset('labels', data=np.array(labels, dtype='uint8'), chunks=True, compression='gzip')
        out.create_dataset('pairs', data=np.array(pairs, dtype=pair_dtype), chunks=True,
                           compression='gzip')
        get_one_side_data(pairs, left=True, out=out, verbose=True,
                          min_size=min_size, ext_size=ext_size, crispred=crispred)
        get_one_side_data(pairs, left=False, out=out, verbose=True,
                          min_size=min_size, ext_size=ext_size, crispred=crispred)
