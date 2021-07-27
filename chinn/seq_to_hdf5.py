import h5py
import bisect
import numpy as np
import time

CHANNELS_FIRST = "channels_first"
CHANNELS_LAST = "channels_last"


def get_point(values, pct):
    """
    Pass in array values and return the point at the specified top percent
    :param values: array: float
    :param pct: float, top percent
    :return: float
    """
    assert 0 < pct < 1, "percentage should be lower than 1"
    values = sorted(values)
    return values[-int(len(values)*pct)]


def load_bed(fn):
    import gzip
    recs = {}
    rec_ends = {}

    values = []
    if fn.endswith('.gz'):
        f = gzip.open(fn, 'rt')
    else:
        f = open(fn)
    for r in f:
        tokens = r.strip().split('\t')
        if tokens[0] not in recs:
            recs[tokens[0]] = []
        intCols = [1,2]
        for i in intCols:
            tokens[i] = int(tokens[i])
        tokens[6] = float(tokens[6])
        values.append(tokens[6])
        recs[tokens[0]].append(tokens)

    if len(values) > 10000:
        pct = 0.001
    else:
        pct = 0.01
    max_value = get_point(values, pct)
    print("{} max clipping value at top {} is: {}".format(fn, pct, max_value))

    for c in recs:
        recs[c].sort(key=lambda k: (k[1], k[2]))
        rec_ends[c] = [k[2] for k in recs[c]]

        for i in range(len(recs[c])):
            recs[c][i][6] = min(recs[c][i][6], max_value) * 1.0 / max_value

    return recs, rec_ends


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



def _process_dnase(curr_r, annotations, seq_len, data_format, one_d, use_rc):
    assert annotations is not None and len(annotations) > 0
    a = curr_r.id.replace(':', ' ').replace('-', ' ')
    chrom, start, end = a.split()
    start = int(start)
    end = int(end)
    curr_matrix = get_annotation_matrix(chrom, start, end, annotations, seq_len, data_format, False)

    curr_rc_matrix = None
    if use_rc:
        curr_rc_matrix = get_annotation_matrix(chrom, start, end, annotations, seq_len, data_format, True)

    return curr_matrix, curr_rc_matrix


def _process_seq(curr_r, seq_len, data_format, one_d, use_rc):
    curr_matrix = get_seq_matrix(curr_r, seq_len, data_format, one_d)
    curr_rc_matrix = None
    if use_rc:
        curr_rc_matrix = get_seq_matrix(curr_r, seq_len, data_format, one_d, rc=True)
    return curr_matrix, curr_rc_matrix


def determine_shapes(data_format, seq_len, channels, one_d, num_instances, fold, annotations):
    start_shape = min(50000, num_instances * fold)
    if not one_d:
        if data_format == CHANNELS_LAST:
            data_shape = (start_shape, 1, seq_len, channels)
            max_shape = (num_instances * fold, 1, seq_len, channels)
        else:
            data_shape = (start_shape, channels, 1, seq_len)
            max_shape = (num_instances * fold, channels, 1, seq_len)
    else:
        if data_format == CHANNELS_LAST:
            data_shape = (start_shape, seq_len, channels)
            max_shape = (num_instances * fold, seq_len, channels)
        else:
            data_shape = (start_shape, channels, seq_len)
            max_shape = (num_instances * fold, channels, seq_len)

    if annotations is None or len(annotations) < 1:
        anno_shape = None
        max_anno_shape = None
    else:
        if data_format == CHANNELS_LAST:
            anno_shape = list(data_shape)[:-1] + [len(annotations)]
            max_anno_shape = list(max_shape)[:-1] + [len(annotations)]
        else:
            anno_shape = list(data_shape)
            anno_shape[1] = len(annotations)
            anno_shape = tuple(anno_shape)

            max_anno_shape = list(max_shape)
            max_anno_shape[1] = len(annotations)
            max_anno_shape = tuple(max_anno_shape)

    return data_shape, max_shape, anno_shape, max_anno_shape


def process_data(fa, seq_len, out_pre, labels, data_format,
                 min_label=0, fa2=None, annotations=None, one_d=False, use_rc=False):

    channels = 4
    dnames = ['data']
    dnase_names = ['dnase']
    if fa2:
        dnames = ['data_left', 'data_right']
        dnase_names = ['dnase_left', 'dnase_right']

    fold = 1
    if use_rc:
        fold = 2
        if fa2 is not None:
            fold *= fold

    data_shape, max_shape, anno_shape, max_anno_shape = determine_shapes(data_format, seq_len, channels,
                                                                         one_d, len(labels), fold, annotations)
    instance_shape = list(data_shape[1:])
    from Bio import SeqIO
    file_handles = [SeqIO.parse(fa, 'fasta')]
    if fa2 is not None:
        file_handles.append(SeqIO.parse(fa2, 'fasta'))
    out = h5py.File(out_pre + "_" + data_format + ("_1d" if one_d else "") + ("" if use_rc else "_norc") + ".h5", "w")

    dsets = []
    dnase_dsets = []
    for dname in dnames:
        dsets.append(out.create_dataset(dname, shape=data_shape,
                                        maxshape=max_shape,
                                        chunks=True,
                                        dtype="float32",
                                        compression="gzip"))
    if annotations is not None:
        for dname in dnase_names:
            dnase_dsets.append(out.create_dataset(dname, shape=anno_shape,
                                                  maxshape=max_anno_shape,
                                                  chunks=True,
                                                  dtype='float32',
                                                  compression='gzip'))

    data = [[] for _ in range(len(file_handles))]
    dnase_data = [[] for _ in range(len(file_handles))]
    distances = []

    last_count = 0
    kept = [[] for _ in data]
    start_time = time.time()
    for count, rs in enumerate(zip(*file_handles)):
        if sum(labels[count]) < min_label:
            continue

        for i, r in enumerate(rs):
            curr_matrices = _process_seq(r, seq_len, data_format, one_d, use_rc)
            for j in range(fold):
                data[i].append(curr_matrices[(j//(2**(len(rs)-i-1))) % 2])
                kept[i].append(count)

        if fa2 is not None:
            coord_centers = []
            for r in rs:
                temp_coords = r.name.replace(":", "\t").replace("-", "\t").split()
                temp_center = (int(temp_coords[2]) + int(temp_coords[1])) // 2
                coord_centers.append((temp_coords[0], temp_center))

            if coord_centers[0][0] == coord_centers[1][0]:
                curr_distance = abs(coord_centers[0][1] - coord_centers[1][1])
            else:
                curr_distance = 20000000
            for i in range(fold):
                distances.append(curr_distance)

        if annotations is not None and len(annotations) > 0:
            for i, r in enumerate(rs):
                curr_matrices = _process_dnase(r, annotations, seq_len, data_format, one_d, use_rc)
                for j in range(fold):
                    dnase_data[i].append(curr_matrices[(j // (2 ** (len(rs) - i - 1))) % 2])

        #if count >= dset.shape[0]:
        #    dset.resize((dset.shape[0] + 5000, 1, 1000, channels))

        if len(kept[0]) % 500000 == 0:
            for i in range(len(dsets)):
                dsets[i].resize([len(kept[i]),] + instance_shape)
                dsets[i][last_count:len(kept[i])] = np.array(data[i])
            if annotations is not None:
                for i in range(len(dnase_dsets)):
                    dnase_dsets[i].resize([len(kept[i]), ] + list(anno_shape[1:]))
                    dnase_dsets[i][last_count:len(kept[i])] = np.array(dnase_data[i])
            if fa2 is not None:
                pass

            last_count = len(kept[0])
            data = [[] for _ in range(len(file_handles))]
            dnase_data = [[] for _ in range(len(file_handles))]
            print(time.time() - start_time, count, len(kept[0]))

    for i in range(len(dsets)):
        dsets[i].resize([len(kept[i]), ] + instance_shape)
        dsets[i][last_count:len(kept[i])] = np.array(data[i])
        if annotations is not None:
            dnase_dsets[i].resize([len(kept[i]), ] + list(anno_shape[1:]))
            dnase_dsets[i][last_count:len(kept[i])] = np.array(dnase_data[i])

    last_count = len(kept[0])

    for f in file_handles:
        f.close()
    print(len(data[0]), data[0][0].shape, data[0][-1].shape, len(kept[0]))
    print(count, labels.shape[0])
    #data = np.array(data, dtype="float32")
    assert labels.shape[0] == count + 1
    assert labels[kept[0]].shape[0] == len(kept[0])
    lset = out.create_dataset("labels", data=labels[kept[0]], dtype="uint8", compression="gzip")

    if fa2 is not None:
        out.create_dataset("distances", data=np.array(distances), dtype="uint", compression="gzip")

    out.close()


def load_labels(labelFile, clip=1, paired=False):
    labels = []
    startIdx = 3
    if paired:
        startIdx = 6
    with open(labelFile) as f:
        for r in f:
            labels.append(list(map(int, r.strip().split()[startIdx:])))
    labels = np.array(labels, dtype="uint8")
    if clip > 0:
        labels[labels > clip] = clip
    return labels
