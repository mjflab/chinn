import argparse
from itertools import product
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser("Make anchors of chromatin interactions into uniform sizes.")

    parser.add_argument("input", help="The input bedpe file. Should have at least 7 columns."
                                      + " The 7th column should be the score if it exists.")
    parser.add_argument("-s", "--target_size", type=int, required=True, help="The target size")
    parser.add_argument("-c", "--clip_oversize",
                        action="store_true",
                        default=False,
                        help="For samples with anchors larger than target size, take the middle part of the anchor."
                        + "Otherwise, the anchors will be splitted into multiple samples.")
    parser.add_argument("--overlap",
                        type=int,
                        default=500,
                        help="How much overlap between consecutive windows "
                        + "when splitting large anchors. Default: 500. Not used when --clip_oversize is true.")
    parser.add_argument("--max_sample", type=int, default=-1, help="The maximum number of samples per large anchors. "
                                                                   + "<= 0 denotes no limit. Default: -1")
    parser.add_argument("-o", "--output", required=True, help="The output file")

    args = parser.parse_args()

    return args

def processAnchor(chrom, start, end, size, clip_oversize, overlap):
    if end - start == size:
        return [[chrom, start, end],]
    elif end - start < size:
        diff = size - end + start
        ext = diff // 2
        return [[chrom, start - ext, end + diff - ext],]
    else:
        if clip_oversize:
            mid = (start + end) // 2
            ext = size // 2
            return [[chrom, mid-ext, mid+size-ext],]
        else:
            norm_anchors = []
            curr_start = start
            while curr_start <= end - size/2:
                norm_anchors.append([chrom, curr_start, curr_start + size])
                curr_start += size - overlap
            return norm_anchors


def process(args):
    pairs = []
    with open(args.input) as f:
        for r in f:
            tokens = r.strip().split('\t')
            if len(tokens) > 6:
                score = int(tokens[6])
            for i in [1,2,4,5]:
                tokens[i] = int(tokens[i])
            if tokens[0] == tokens[3] and tokens[1] > tokens[4]:
                temp = tokens[1]
                tokens[1] = tokens[4]
                tokens[4] = temp
                temp = tokens[2]
                tokens[2] = tokens[5]
                tokens[5] = temp
            left_anchors = processAnchor(tokens[0], tokens[1], tokens[2], args.target_size, args.clip_oversize, args.overlap)
            right_anchors = processAnchor(tokens[3], tokens[4], tokens[5], args.target_size, args.clip_oversize, args.overlap)
            pair_idx = list(product(range(len(left_anchors)), range(len(right_anchors))))
            #to_sample = len(pair_idx)
            #if to_sample > args.max_sample > 0:
            np.random.shuffle(pair_idx)
            to_sample = min(args.max_sample, max(int(np.log2(len(left_anchors))), int(np.log2(len(right_anchors)))) + 1)

            for p in pair_idx[:to_sample]:
                temp_pair = left_anchors[p[0]] + right_anchors[p[1]]
                if len(tokens) > 6:
                    temp_pair.append(score)
                pairs.append(temp_pair)

    return pairs


def write_pairs(pairs, outFile):
    with open(outFile, 'w') as out:
        for p in pairs:
            out.write("\t".join(map(str, p)) + "\n")



if __name__=="__main__":
    args = parse_args()
    pairs = process(args)
    write_pairs(pairs, args.output)