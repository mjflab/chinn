import os
import numpy as np
import yaml
import argparse
from preprocess.pair_generation import load_data

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Generate random pairs of DNase regions")
    parser.add_argument("name", help="The prefix of the dataset. For example k562_polr2a.")
    parser.add_argument("input_file", help="The DNase/TF file in BED format")
    parser.add_argument("type", choices=['dnase', 'tf'],
                        help="Can be dnase or tf, indicating whether this is for DNase or TF peaks pairs")
    parser.add_argument("datadir", help="The directory where the input and output are in.")
    args = parser.parse_args()

    anchor_file = "{}/{}_merged_anchors.both_dnase.bed".format(args.datadir, args.name)
    inter_file = "{}/{}.clustered_interactions.both_dnase.bedpe".format(args.datadir, args.name)
    anchors, scores, t_dists = load_data(anchor_file, inter_file)
    anchor_sizes = []
    for a in anchors:
        anchor_sizes.append(a[2]-a[1])

    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                           '{}_anchor_offsets.yaml'.format(args.type)), 'r') as f:
        offsets = yaml.safe_load(f)
    mean_offsets = offsets['mean_offsets']
    std_offsets = offsets['std_offsets']

    log_sizes = np.log10(anchor_sizes)
    anchor_size_mean = np.mean(log_sizes)-mean_offsets[args.name]
    anchor_size_std = np.std(log_sizes)-std_offsets[args.name]
    random_anchor_size = np.random.normal(anchor_size_mean, anchor_size_std, len(log_sizes))

    neg_anchors = []
    with open(args.input_file, 'r') as f:
        for r in f:
            temp = r.strip().split()
            temp[1] = int(temp[1])
            temp[2] = int(temp[2])
            center = 0.5*(temp[1] + temp[2])
            temp_size = max(temp[2]-temp[1], 10**np.random.normal(anchor_size_mean, anchor_size_std))
            temp[1] = int(max(0, center - temp_size/2))
            temp[2] = int(center + temp_size/2)
            neg_anchors.append(list(temp))
    neg_anchors.sort(key=lambda k:(k[0], k[1], k[2]))
    merged_neg_anchors = []
    last_anchor = neg_anchors[0]
    for i in range(1, len(neg_anchors)):
        curr_anchor = neg_anchors[i]
        if curr_anchor[0] == last_anchor[0] and curr_anchor[1] <= last_anchor[2]+500:
            last_anchor[2] = max(last_anchor[2], curr_anchor[2])
        else:
            merged_neg_anchors.append(last_anchor)
            last_anchor = curr_anchor

    merged_neg_anchors.append(last_anchor)

    v1, e1 = np.histogram(np.log10([a[2]-a[1] for a in merged_neg_anchors]), bins=100, density=True)
    v2, e2 = np.histogram(log_sizes, bins=100, density=True)
    c1 = 0.5*(e1[:-1] + e1[1:])
    c2 = 0.5*(e2[:-1] + e2[1:])

    print(len(merged_neg_anchors))
    if args.type == "dnase":
        outfile = "{}/{}.shuffled_neg_anchor.neg_pairs.bedpe".format(args.datadir, args.name)
    elif args.type == "tf":
        outfile = "{}/{}.random_tf_peak_pairs.bedpe".format(args.datadir, args.name)
    with open(outfile, 'w') as out:
        neg_pairs = 0
        for i in range(len(merged_neg_anchors)):
            a1 = merged_neg_anchors[i]
            for j in range(i+1, len(merged_neg_anchors)):
                a2 = merged_neg_anchors[j]
                if a1[0] != a2[0]:
                    break
                if 5000 < 0.5*(a2[1] - a1[1] + a2[2] - a1[2]) < 2000000:
                    out.write("\t".join(map(str, list(a1[:3]) + list(a2[:3]))) + "\n")
                    neg_pairs += 1
            if i % 2000 == 0:
                print(neg_pairs)