import os
import numpy as np
import argparse
from chinn.pair_features import load_pairs_as_dict
from preprocess.pair_generation import sample_from_neg_pairs


if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Sampling 5x negative samples.")
    parser.add_argument("name", help="The prefix of the dataset. For example k562_polr2a.")
    parser.add_argument("datadir", help="The directory where the input and output are in.")
    args = parser.parse_args()

    num_bins = 50
    dist_range = (np.log10(5000), np.log10(2000000))

    pos_pairs, pos_dists = load_pairs_as_dict(
        [os.path.join(args.datadir, "{}.clustered_interactions.both_dnase.bedpe".format(args.name))])

    neg_pairs, neg_dists = load_pairs_as_dict(
        [os.path.join(args.datadir, '{}.no_intra_all.negative_pairs.bedpe'.format(args.name)),
         os.path.join(args.datadir, '{}.random_tf_peak_pairs.filtered.bedpe'.format(args.name))],
        min_length=1000)
    other_neg_pairs, other_neg_dists = load_pairs_as_dict(
        [os.path.join(args.datadir, '{}.shuffled_neg_anchor.neg_pairs.filtered.tf_filtered.bedpe'.format(args.name))],
        min_length=1000)

    selected_neg_pairs = sample_from_neg_pairs(pos_dists, neg_pairs, 5, other_neg_pairs, num_bins, dist_range)
    selected_neg_dists = [0.5*(p[5]+p[4]-p[2]-p[1]) for p in selected_neg_pairs]
    with open(os.path.join(args.datadir,
                           "{}.neg_pairs_5x.from_singleton_inter_tf_random.bedpe".format(args.name)),'w') as out:
        for p in selected_neg_pairs:
            out.write("\t".join(map(str, p)) + "\n")