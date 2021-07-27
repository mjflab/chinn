import numpy as np
import argparse
from preprocess.pair_generation import load_data, get_clusters, get_neg_pairs, get_cluster_sizes, print_total_pairs
from preprocess.pair_generation import save_neg_pairs

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Generate random pairs of anchors")
    parser.add_argument("name", help="The prefix of the dataset. For example k562_polr2a.")
    parser.add_argument("datadir", help="The directory where the input and output are in.")
    args = parser.parse_args()

    anchor_file = "{}/{}_merged_anchors.both_dnase.bed".format(args.datadir, args.name)
    inter_file = "{}/{}.clustered_interactions.both_dnase.bedpe".format(args.datadir, args.name)
    folds = []
    only_intra = [False, True]

    for intra in only_intra:
        anchors, scores, t_dists = load_data(anchor_file, inter_file)
        clusters = get_clusters(anchors)
        bin_stats = np.histogram(np.log10(t_dists), bins=10, range=(np.log10(5000), np.log10(2000000)))
        curr_counts, curr_edges = bin_stats
        cluster_sizes = get_cluster_sizes(clusters)
        print_total_pairs(cluster_sizes)
        all_pairs = get_neg_pairs(scores, clusters, bin_stats, allow_intra=intra, only_intra=intra, fold=None)
        selected_pairs = []
        for k in all_pairs:
            selected_pairs += list(k)
        print(len(selected_pairs))
        intra_str = "only_intra" if intra else "no_intra"
        name = "{}.{}_all".format(args.name, intra_str)
        save_neg_pairs("{}/{}.negative_pairs.bedpe".format(args.datadir, name), selected_pairs)
