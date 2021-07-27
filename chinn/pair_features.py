# coding: utf-8

import numpy as np
import pylab as pl
import h5py
from sklearn.utils import shuffle
import bisect
import os
from chinn import variables


CHANNELS_FIRST = "channels_first"
CHANNELS_LAST = "channels_last"


def load_pairs_as_dict(files, min_length=1000, min_dist=5000, max_dist=2000000, max_length=None):
    scores = {}
    t_dists = {}
    for inter_file in files:
        with open(inter_file, 'r') as f:
            for r in f:
                tokens = r.strip().split()
                if len(tokens) < 7:
                    tokens.append(0)
                for i in [1, 2, 4, 5, 6]:
                    try:
                        tokens[i] = int(tokens[i])
                    except:
                        tokens[i] = int(float(tokens[i]))
                if tokens[0] == tokens[3] and tokens[1] > tokens[4]:
                    temp = tokens[1]
                    tokens[1] = tokens[4]
                    tokens[4] = temp
                    temp = tokens[2]
                    tokens[2] = tokens[5]
                    tokens[5] = temp
                if max_length is not None and (tokens[2]-tokens[1] > max_length or tokens[5] - tokens[4] > max_length):
                    continue
                if min_length > 0:
                    for i,j in zip([1,4],[2,5]):
                        if tokens[j] - tokens[i] < min_length:
                            diff = min_length - (tokens[j]-tokens[i])
                            half_diff = diff // 2
                            tokens[i] -= half_diff
                            tokens[j] += diff - half_diff
                curr_dist = 0.5 * (tokens[4] + tokens[5] - tokens[1] - tokens[2])
                if min_dist <= curr_dist <= max_dist:
                    scores[tuple(tokens[:6])] = tokens[6]
                    if tokens[0] not in t_dists:
                        t_dists[tokens[0]] = []
                    t_dists[tokens[0]].append(curr_dist)

    return scores, t_dists

def get_seq_matrix(seq, seq_len: int, data_format: str, one_d: bool, rc=False):
    channels = 4
    mat = np.zeros((seq_len, channels), dtype="float32")

    for i, a in enumerate(seq):
        idx = i
        if idx >= seq_len:
            break
        a = a.lower()

        if a == 'a':
            mat[idx, 0] = 1
        elif a == 'g':
            mat[idx, 1] = 1
        elif a == 'c':
            mat[idx, 2] = 1
        elif a == 't':
            mat[idx, 3] = 1
        else:
            mat[idx, 0:4] = 0

    if rc:
        mat = mat[::-1, ::-1]

    if not one_d:
        mat = mat.reshape((1, seq_len, channels))
    if data_format == CHANNELS_FIRST:
        axes_order = [len(mat.shape)-1,] + [i for i in range(len(mat.shape)-1)]
        mat = mat.transpose(axes_order)

    return mat


def get_annotation_matrix(chrom, start, end, annotations, seq_len, data_format, rc=False):
    out = np.zeros((len(annotations), seq_len), dtype="float32")
    end = min(start + seq_len, end)
    for i, anno in enumerate(annotations):
        if chrom in anno[0]:
            recs = anno[0][chrom]
            rec_ends = anno[1][chrom]
            si = bisect.bisect(rec_ends, start)
            while si < len(recs) and recs[si][1] < end:
                r = recs[si]
                if end > r[1] and start < r[2]:
                    r_start = max(r[1], start) - start
                    r_end = min(r[2], end) - start
                    out[i][r_start:r_end] = r[6]
                si += 1
    if rc:
        out = out[:, ::-1]
    if data_format != CHANNELS_FIRST:
        out = out.transpose()
    return out


def _get_sequence(chrom, start, end, min_size=1000, crispred=None):
    # assumes the CRISPRed regions were not overlapping
    # assumes the CRISPRed regions were sorted
    if crispred is not None:
        #print('crispred is not None')
        seq = ''
        curr_start = start
        for cc, cs, ce in crispred:
            # overlapping
            #print('check', chrom, start, end, cc, cs, ce)
            if chrom == cc and min(end, ce) > max(cs, curr_start):
                #print('over', curr_start, end, cs, ce)
                if curr_start > cs:
                    seq += variables.hg19[chrom][curr_start:cs]
                curr_start = ce
        if curr_start < end:
            seq += variables.hg19[chrom][curr_start:end]
        #print(start, end, end-start, len(seq))

    else:
        seq = variables.hg19[chrom][start:end]

    if len(seq) < min_size:
        diff = min_size - (end - start)
        ext_left = diff // 2
        if start - ext_left < 0:
            ext_left = start
        elif diff - ext_left + end > len(variables.hg19[chrom]):
            ext_left = diff - (len(variables.hg19[chrom]) - end)
        curr_start = start - ext_left
        curr_end = end + diff - ext_left

        if curr_start < start:
            seq = variables.hg19[chrom][curr_start:start] + seq
        if curr_end > end:
            seq = seq + variables.hg19[chrom][end:curr_end]
    if start < 0 or end > len(variables.hg19[chrom]):
        return None
    return seq


def encode_seq(chrom, start, end, min_size=1000, crispred=None):
    seq = _get_sequence(chrom, start, end, min_size, crispred)
    if seq is None:
        return None
    mat = get_seq_matrix(seq, len(seq), 'channels_first', one_d=True, rc=False)
    parts = []
    for i in range(0, len(seq), 500):
        if i + 1000 >= len(seq):
            break
        parts.append(mat[:, i:i + 1000])
    parts.append(mat[:, -1000:])
    parts = np.array(parts, dtype='float32')
    return parts


def print_feature_importances(estimator):
    f = open("/data/protein/gm12878_files.txt")
    ann_files = [r.strip().split('/')[-1] for r in f]
    f.close()
    feature_importances = [(idx, n, i) for idx, (n, i) in
                           enumerate(zip(['distance', 'correlation'] + ann_files * 2, estimator.feature_importances_))]
    feature_importances.sort(key=lambda k: -k[2])
    for idx, n, i in feature_importances:
        print(idx, '\t', n, "\t", i)


def plot_dist_distr(pos_dists_dict, neg_dists_dict, normed=True, savefig=None,
                    num_bins=50, dist_range=(np.log10(5000), np.log10(2000000))):
    pos_dists = []
    for c in pos_dists_dict:
        pos_dists += pos_dists_dict[c]
    neg_dists = []
    for c in neg_dists_dict:
        neg_dists += neg_dists_dict[c]
    n_counts, n_edges = np.histogram(np.log10(neg_dists), normed=normed, bins=num_bins, range=dist_range)
    p_counts, p_edges = np.histogram(np.log10(pos_dists), bins=num_bins, normed=normed, range=dist_range)
    n_centers = 0.5*(n_edges[:-1] + n_edges[1:])
    p_centers = 0.5*(p_edges[:-1] + p_edges[1:])
    fig = pl.figure()
    pl.plot(n_centers, n_counts, label="negative\n(%d)"%len(neg_dists))
    pl.plot(p_centers, p_counts, label="positive\n(%d)"%len(pos_dists))
    pl.legend(loc='upper left')
    pl.xlabel("Distance between centers of anchors (log10)", fontsize=14)
    if normed:
        pl.ylabel("Density", fontsize=14)
    else:
        pl.ylabel("Frequency", fontsize=14)
    if savefig is not None:
        fig.savefig(savefig + ".pdf", dpi=600)
        fig.savefig(savefig + ".jpg", dpi=300)
    #print("\n".join(["\t".join(map(str, i)) for i in zip(n_centers, n_counts, p_centers, p_counts)]))


def generate_features(a, annotations):
    temp_dist = np.log10(0.5*(a[4] + a[5] - a[1] - a[2])) / 7.0
    temp_mat1 = get_annotation_matrix(a[0],a[1],a[2], annotations, a[2]-a[1], 'channels_first')
    temp_mat2 = get_annotation_matrix(a[3],a[4],a[5], annotations, a[5]-a[4], 'channels_first')
    temp_mean1 = list(np.mean(temp_mat1, axis=1))
    temp_mean2 = list(np.mean(temp_mat2, axis=1))

    return [temp_dist] + temp_mean1 + temp_mean2


def get_matrix_binary(chrom, start, end, annotations, seq_len, data_format):
    curr_features = [0 for _ in range(len(annotations))]
    end = min(start + seq_len, end)
    for i, anno in enumerate(annotations):
        if chrom in anno[0]:
            recs = anno[0][chrom]
            rec_ends = anno[1][chrom]
            si = bisect.bisect(rec_ends, start)
            while si < len(recs) and recs[si][1] < end:
                curr_overlap = min(recs[si][2], end) - max(recs[si][1], start)
                if curr_overlap >= 100 or curr_overlap / (end - start) >= 0.5 or curr_overlap / (recs[si][2] - recs[si][1]) >= 0.5:
                    curr_features[i] += 1
                si += 1
    return curr_features


def generate_features_binary(a, annotations):
    chrom_map = {}
    for i, c in enumerate(list(range(1, 23)) + ['X']):
        chrom_map[i + 1] = 'chr' + str(c)
    a = list(a)
    if type(a[0]) == int:
        a[0] = chrom_map[a[0]]
    if type(a[3]) == int:
        a[3] = chrom_map[a[3]]

    temp_mean1 = get_matrix_binary(a[0], a[1], a[2], annotations, a[2] - a[1], 'channels_first')
    temp_mean2 = get_matrix_binary(a[3], a[4], a[5], annotations, a[5] - a[4], 'channels_first')
    temp_dist = np.log10(0.5 * (a[4] + a[5] - a[1] - a[2])) / 7.0
    return [temp_dist, ] + temp_mean1 + temp_mean2


def generate_data(pos_pairs, neg_pairs, annotations, binary=False, min_size=1000):
    if binary:
        gen_fn = generate_features_binary
    else:
        gen_fn = generate_features

    chrom_sizes = {}
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hg19.len'), 'r') as f:
        for r in f:
            tokens = r.strip().split()
            chrom_sizes[tokens[0]] = int(tokens[1])
    print(len(pos_pairs), len(neg_pairs))
    train_data = []
    val_data = []
    test_data = []
    train_label = []
    val_label = []
    test_label = []
    train_pairs = []
    val_pairs = []
    test_pairs = []
    val_chroms = ['chr5', 'chr14']
    test_chroms = ['chr' + str(i) for i in [4, 7, 8, 11]]
    for a in pos_pairs:
        if a[2] - a[1] < min_size:
            temp = (a[1] + a[2]) // 2
            a[1] = temp - min_size // 2
            a[2] = temp + (min_size - min_size // 2)
        if a[5] - a[4] < min_size:
            temp = (a[4] + a[5]) // 2
            a[4] = temp - min_size // 2
            a[5] = temp + (min_size - min_size // 2)
        if (a[1] < 0 or a[4] < 0 or
            a[2] >= chrom_sizes[a[0]] or a[5] >= chrom_sizes[a[3]]):
            print('skipping', a)
            continue
        curr_features = gen_fn(a, annotations)
        if a[0] in val_chroms:
            val_data.append(curr_features)
            val_label.append(1)
            val_pairs.append(a)
        elif a[0] in test_chroms:
            test_data.append(curr_features)
            test_label.append(1)
            test_pairs.append(a)
        else:
            train_data.append(curr_features)
            train_label.append(1)
            train_pairs.append(a)
    print("finished positive")

    for a in neg_pairs:
        curr_features = gen_fn(a, annotations)
        if a[0] in val_chroms:
            val_data.append(curr_features)
            val_label.append(0)
            val_pairs.append(a)
        elif a[0] in test_chroms:
            test_data.append(curr_features)
            test_label.append(0)
            test_pairs.append(a)
        else:
            train_data.append(curr_features)
            train_label.append(0)
            train_pairs.append(a)
    train_data, train_label, train_pairs = shuffle(train_data, train_label, train_pairs)
    val_data, val_label, val_pairs = shuffle(val_data, val_label, val_pairs)
    test_data, test_label, test_pairs = shuffle(test_data, test_label, test_pairs)
    train_data = np.array(train_data)
    val_data = np.array(val_data)
    test_data = np.array(test_data)
    return (train_data, train_label, train_pairs), (val_data, val_label, val_pairs), (test_data, test_label, test_pairs)


def save_data_to_hdf5(data, labels, pairs, name, dset, type_str):
    chrom_map = {}
    for i, x in enumerate(list(range(1, 23)) + ['X']):
        chrom_map['chr' + str(x)] = i + 1
    new_pairs = []
    for p in pairs:
        p = list(p)
        p[0] = chrom_map[p[0]]
        p[3] = chrom_map[p[3]]
        new_pairs.append(p)
    new_pairs = np.array(new_pairs)
    with h5py.File('%s_%s_%s.hdf5' % (name, type_str, dset), 'w') as a:
        a.create_dataset('data', data=data, chunks=True, compression='gzip')
        a.create_dataset('pairs', data=new_pairs, compression='gzip')
        a.create_dataset('labels', data=np.array(labels, dtype='int8'), compression='gzip')


def load_data_from_hdf5(name, dset, type_str):
    a = h5py.File('%s_%s_%s.hdf5' % (name, type_str, dset), 'r')
    data = a['data'][:]
    labels = a['labels'][:]
    pairs = a['pairs'][:]
    a.close()

    return data, labels, pairs
