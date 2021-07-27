inch = 25.4
col1Width = 87 / inch

__colors = [
        (214, 39, 40), (255, 152, 150),    #redish
        (31, 119, 180), (174, 199, 232),   #blueish
        (44, 160, 44), (152, 223, 138),   #greenish
        (148, 103, 189), (197, 176, 213), #purple
        (255, 127, 14), (255, 187, 120),   #orangish
        (140, 86, 75), (196, 156, 148),    #brown
        (227, 119, 194), (247, 182, 210),  #pink
        (127, 127, 127), (199, 199, 199),    #grey
        (188, 189, 34), (219, 219, 141),   #dark green
        (23, 190, 207), (158, 218, 229)  #bright blue
    ]


def colors(i):
    return (__colors[i][0]/255., __colors[i][1]/255., __colors[i][2]/255.)


def check_chrom(chrom):
    import re
    return re.match('^chr(\d+|X)$', chrom)


def chrom_to_int(chrom):
    if chrom == 'chrX':
        chrom = 23
    else:
        chrom = int(chrom.replace('chr', ''))
    return chrom