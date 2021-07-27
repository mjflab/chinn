#!/usr/bin/env python
import sys
import argparse


def parse_args():
    parser = argparse.ArgumentParser("Group bed by columns.")
    parser.add_argument("-i", '--input', help="The input file.")
    parser.add_argument('-g', '--grp', type=int, nargs='+', default=[1, 2, 3],
                        help="The columns used for grouping. Starting from 1.")
    parser.add_argument('-c', '--opCols', type=int, nargs='+', default=[4],
                        help="The columns used for operation. Starting from 1.")
    parser.add_argument('-o', '--op', type=str, default='sum', choices=['sum', 'collapse'],
                        help="The operation. Can be [sum, collapse]. Default: sum")
    return parser.parse_args()


def write_group(group, op, last_key, op_cols):
    if len(group) > 0:
        if op == "collapse":
            grouped_strs = [",".join([g[i - 1] for g in group]) for i in op_cols]
        else:
            grouped_strs = list(map(str, [sum([float(g[i - 1]) for g in group]) for i in op_cols]))
        sys.stdout.write('\t'.join(list(last_key) + grouped_strs) + "\n")


def main():
    args = parse_args()
    if args.input is None:
        file_in = sys.stdin
    else:
        file_in = open(args.input)
    curr_group = []
    last_key = None
    for r in file_in:
        tokens = r.strip().split('\t')
        curr_key = []
        for i in args.grp:
            curr_key.append(tokens[i-1])
        curr_key = tuple(curr_key)
        if last_key is None:
            last_key = curr_key
        if curr_key == last_key:
            curr_group.append(tokens)
        else:
            write_group(curr_group, args.op, last_key, args.opCols)
            last_key = curr_key
            curr_group = [tokens]
    write_group(curr_group, args.op, last_key, args.opCols)


if __name__=="__main__":
    main()