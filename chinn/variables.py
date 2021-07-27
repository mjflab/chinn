from Bio import SeqIO

def init(gfile):
    global hg19
    hg19 = ['' for i in range(24)]
    for o in SeqIO.parse(open(gfile), 'fasta'):
        if o.name.replace('chr', '') not in list(map(str, range(1,23))) + ['X']:
            continue
        if o.name == 'chrX':
            temp_key = 23
        else:
            temp_key = int(o.name.replace('chr', ''))
        hg19[temp_key] = str(o.seq)
