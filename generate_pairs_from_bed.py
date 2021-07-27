import argparse


if __name__=='__main__':
    parser = argparse.ArgumentParser('Generate all possible peak pairs on the same chromosome '
                                     + 'and have a distance within [5Kb, 2Mb] from a BED file')
    parser.add_argument('input',
                        help='The input file in BED format, or at least the first three columns are in BED format')
    parser.add_argument('out_file', help='THe output file in BEDPE format')
    args = parser.parse_args()

    peaks = []
    with open(args.input) as f:
        for r in f:
            tokens = r.strip().split()
            peaks.append([tokens[0], int(tokens[1]), int(tokens[2])])

    pairs = []
    for i in range(len(peaks)-1):
        for j in range(i+1, len(peaks)):
            if peaks[i][0] != peaks[j][0]:
                break
            else:
                if 5000 < 0.5*(abs(peaks[i][1]-peaks[j][1] + peaks[i][2] - peaks[j][2])) < 2000000:
                    pairs.append(peaks[i] + peaks[j])
                elif 0.5*(abs(peaks[i][1]-peaks[j][1] + peaks[i][2] - peaks[j][2]))  >= 2000000:
                    break
        if i % 5000 == 0:
            print(i, '/', len(peaks))
    with open(args.out_file, 'w') as out:
        for p in pairs:
            out.write('\t'.join(map(str, p)) + '\n')

