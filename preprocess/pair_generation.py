import numpy as np
import bisect


def select_neg_pairs(fold, chrom_pairs, curr_counts):
    selected_pairs = []
    shortage = [0 for _ in curr_counts]
    for k in range(len(chrom_pairs)):
        chrom_pairs[k] = list(chrom_pairs[k])
        if len(chrom_pairs[k]) <= 0:
            continue
        if curr_counts[k] * fold >= len(chrom_pairs[k]):
            shortage[k] += curr_counts[k] * fold - len(chrom_pairs[k])
            selected_pairs += chrom_pairs[k]
        elif curr_counts[k] > 0:
            temp_num_to_select = min(curr_counts[k] * fold + shortage[k], len(chrom_pairs[k]))
            shortage[k] -= temp_num_to_select - curr_counts[k] * fold
            selected_idxes = np.random.choice(range(len(chrom_pairs[k])),
                                              temp_num_to_select,
                                              replace=False)
            selected_pairs += [chrom_pairs[k][x] for x in selected_idxes]
    print("shortage:", "\t".join(map(str, shortage)))
    return selected_pairs


def load_data(anchor_file, inter_file):
    anchors = {}
    scores = {}
    with open(anchor_file, 'r') as f:
        for r in f:
            tokens = r.strip().split()
            for i in [1, 2]:
                tokens[i] = int(tokens[i])
            anchors[tuple(tokens)] = set()
    t_dists = []
    with open(inter_file, 'r') as f:
        for r in f:
            tokens = r.strip().split()
            for i in [1, 2, 4, 5, 6]:
                try:
                    tokens[i] = int(tokens[i])
                except:
                    tokens[i] = int(float(tokens[i]))
            scores[tuple(tokens[:6])] = tokens[6]
            anchor1 = tuple(tokens[:3])
            anchor2 = tuple(tokens[3:6])
            anchors[anchor1].add(anchor2)
            anchors[anchor2].add(anchor1)
            t_dists.append(0.5 * (tokens[4] + tokens[5] - tokens[1] - tokens[2]))
    return anchors, scores, t_dists


def get_clusters(anchors):
    clusters = []

    def get_anchors(a, curr_cluster):
        if a not in anchors:
            return
        curr_cluster.add(a)
        linked_anchors = anchors.pop(a)
        for k in linked_anchors:
            get_anchors(k, curr_cluster)

    while True:
        curr_cluster = set()
        curr_key = list(anchors.keys())[0]
        get_anchors(curr_key, curr_cluster)
        clusters.append(curr_cluster)
        if len(anchors) <= 0:
            break
    for i in range(len(clusters)):
        clusters[i] = list(clusters[i])
        clusters[i].sort()
    clusters.sort(key=lambda k: (k[0][0], k[0][1], k[-1][2]))
    return clusters



def get_bin_idx(bin_edges, d, logged=True):
    if logged:
        idx = bisect.bisect_left(bin_edges, np.log10(d))
    else:
        idx = bisect.bisect_left(bin_edges, d)
    if 0 < idx < len(bin_edges):
        return idx
    else:
        return None


def get_cluster_sizes(clusters):
    cluster_sizes = {}
    for i in range(len(clusters)):
        chrom = clusters[i][0][0]
        if chrom not in cluster_sizes:
            cluster_sizes[chrom] = []
        cluster_sizes[chrom].append(len(clusters[i]))
    return cluster_sizes


def print_total_pairs(cluster_sizes):
    total_pairs = 0
    for c in cluster_sizes:
        c_s = cluster_sizes[c]
        for i in range(len(c_s) - 1):
            for j in range(i, len(c_s)):
                total_pairs += c_s[i] * c_s[j]
    print(total_pairs)


def get_neg_pairs(scores, clusters, bin_stats, allow_intra=True, only_intra=False, fold=None,
                  min_dist=5000, max_dist=2000000):
    selected_pairs = []

    curr_chrom = clusters[0][0][0]
    curr_counts, curr_edges = bin_stats
    all_pairs = [set() for _ in curr_counts]
    shortage = [0 for _ in curr_counts]

    def _select_chrom_pairs():
        selected_pairs = []
        for k in range(len(all_pairs)):
            all_pairs[k] = list(all_pairs[k])
            if len(all_pairs[k]) <= 0:
                continue
            if curr_counts[k] * fold >= len(all_pairs[k]):
                shortage[k] += curr_counts[k] * fold - len(all_pairs[k])
                selected_pairs += all_pairs[k]
            elif curr_counts[k] > 0:
                temp_num_to_select = min(curr_counts[k] * fold + shortage[k], len(all_pairs[k]))
                shortage[k] -= temp_num_to_select - curr_counts[k] * fold
                selected_idxes = np.random.choice(range(len(all_pairs[k])),
                                                  temp_num_to_select,
                                                  replace=False)
                selected_pairs += [all_pairs[k][x] for x in selected_idxes]

    for i in range(len(clusters) - 1):
        if clusters[i][0][0] != curr_chrom:
            print("finishing", curr_chrom)
            curr_chrom = clusters[i][0][0]

        next_cluster_end = len(clusters)
        if allow_intra:
            next_cluster_start = i
            if only_intra:
                next_cluster_end = i+1
        else:
            next_cluster_start = i+1
        for j in range(next_cluster_start, next_cluster_end):
            if clusters[i][0][0] != clusters[j][0][0]:
                break

            for temp_i, p1 in enumerate(clusters[i]):
                temp_start_idx = 0
                if i == j:
                    temp_start_idx = temp_i + 1
                for p2 in clusters[j][temp_start_idx:]:

                    first = p1
                    second = p2

                    if p1[1] > p2[1]:
                        first = p2
                        second = p1
                    if first == second or tuple(list(first) + list(second)) in scores:
                        continue
                    curr_dist = 0.5 * (second[2] - first[2] + second[1] - first[1])

                    if min_dist <= curr_dist <= max_dist and second[1] - first[2] >= 1000:
                        curr_idx = get_bin_idx(curr_edges, curr_dist)
                        if curr_idx is not None:
                            all_pairs[curr_idx - 1].add(tuple(first + second))
    if fold is None:
        selected_pairs = all_pairs
    else:
        _select_chrom_pairs()
    print("final shortage: ", shortage)
    return selected_pairs


def save_neg_pairs_by_chrom(filename, selected_pairs):
    out = open(filename, 'w')
    for chrom in selected_pairs:
        for p in selected_pairs[chrom]:
            out.write("\t".join(map(str, p)) + "\n")
    out.close()


def save_neg_pairs(filename, selected_pairs):
    out = open(filename, 'w')
    for p in selected_pairs:
        out.write("\t".join(map(str, p)) + "\n")
    out.close()


def plot_dist_distri(t_dists, selected_pairs):
    import pylab as pl
    print(len(selected_pairs))
    n_counts, n_edges = np.histogram(
        np.log10([0.5 * (-p[1] + p[4] - p[2] + p[5]) for p in selected_pairs]),
        normed=True, bins=20)
    p_counts, p_edges = np.histogram(np.log10([p for p in t_dists]), bins=20, normed=True)
    n_centers = 0.5 * (n_edges[:-1] + n_edges[1:])
    p_centers = 0.5 * (p_edges[:-1] + p_edges[1:])
    _ = pl.plot(n_centers, n_counts)
    _ = pl.plot(p_centers, p_counts)
    pl.show()
    print("\n".join(["\t".join(map(str, i)) for i in zip(n_centers, n_counts, p_centers, p_counts)]))


def sample_from_neg_pairs(pos_dists_dict, neg_pairs, fold, other_neg_pairs, num_bins, dist_range):
    selected = []
    shortage = {}

    for chrom in pos_dists_dict:
        print(chrom)
        pos_dists = pos_dists_dict[chrom]
        counts, edges = np.histogram(np.log10(pos_dists), normed=False, bins=num_bins, range=dist_range)
        neg_classes = [[] for _ in counts]
        other_neg_classes = [[] for _ in counts]
        shortage[chrom] = [0 for _ in counts]
        for p in neg_pairs:
            if p[0] != chrom:
                continue
            p_dist = 0.5 * (p[5] + p[4] - p[2] - p[1])
            idx = get_bin_idx(edges, p_dist)
            if idx is not None:
                neg_classes[idx - 1].append(p)

        for p in other_neg_pairs:
            if p[0] != chrom:
                continue
            p_dist = 0.5 * (p[5] + p[4] - p[2] - p[1])
            idx = get_bin_idx(edges, p_dist)
            if idx is not None:
                other_neg_classes[idx - 1].append(p)

        for i, n in enumerate(counts):
            shortage[chrom][i] = min(len(neg_classes[i]) + len(other_neg_classes[i]) - int(n * fold), 0)
            # print(shortage[chrom])

    shortage = list(shortage.items())
    shortage.sort(key=lambda k: np.sum(k[1]))
    print(shortage)
    running_shortage = np.zeros(len(shortage[0][1]))
    for chrom, _ in shortage:
        print(chrom)
        pos_dists = pos_dists_dict[chrom]

        counts, edges = np.histogram(np.log10(pos_dists), normed=False, bins=num_bins, range=dist_range)
        neg_classes = [[] for _ in counts]
        other_neg_classes = [[] for _ in counts]

        for p in neg_pairs:
            if p[0] != chrom:
                continue
            p_dist = 0.5 * (p[5] + p[4] - p[2] - p[1])
            idx = get_bin_idx(edges, p_dist)
            if idx is not None:
                neg_classes[idx - 1].append(p)

        for p in other_neg_pairs:
            if p[0] != chrom:
                continue
            p_dist = 0.5 * (p[5] + p[4] - p[2] - p[1])
            idx = get_bin_idx(edges, p_dist)
            if idx is not None:
                other_neg_classes[idx - 1].append(p)

        for i, n in enumerate(counts):
            has_shortage = False
            surplus = len(neg_classes[i]) + len(other_neg_classes[i]) - int(n * fold)
            to_select = int(n * fold) + int(min(surplus, running_shortage[i]))
            if running_shortage[i] > 0:
                has_shortage = True
            if to_select > len(neg_classes[i]):

                selected += neg_classes[i]

                idx = np.random.choice(np.arange(len(other_neg_classes[i])),
                                       min(len(other_neg_classes[i]), to_select - len(neg_classes[i])),
                                       replace=False)
                selected += [other_neg_classes[i][x] for x in idx]

            elif to_select > 0:
                idx = np.random.choice(np.arange(len(neg_classes[i])), to_select, replace=False)
                selected += [neg_classes[i][x] for x in idx]
            running_shortage[i] -= to_select - int(n * fold)
            if has_shortage and running_shortage[i] == 0:
                print("Done for %d" % i)
        print(running_shortage)

        # print(len(selected), shortage)
        # print(surplus[chrom])
    return selected