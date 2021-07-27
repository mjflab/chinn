#!/bin/bash
# The input data should have been filtered against blacklisted regions

usage()
{
  echo "$(basename "$0") [-h] INTERS DNASE TFPEAKS NAME DATADIR"
  echo "-- Progam to preprocess the interactions and generate negative samples."

  echo "where:"
  echo "-h           show this help text"
  echo "INTERS       Interaction file in BEDPE format"
  echo "DNASE        Dnase/open chromatin regions in BED format"
  echo "TFPEAKS      The transcription factor peaks for the ChIA-PET protein in BED format"
  echo "NAME         The prefix/name for the sample/experiment"
  echo "DATADIR      Location of the output directory"
}


if [ "$1" != "" ]; then
    case $1 in
        -h | --help )           usage
                                exit
                                ;;
    esac
fi

if [ $# -lt 5 ]; then
  usage
  exit
fi

INTERS=$1 
DNASE=$2
TFPEAKS=$3
NAME=$4
DATADIR=$5

DIR=$(dirname "$0")

echo "Removing interactions whose two anchors are overlapping or on different chromosomes"
cat $INTERS | awk '$1==$4 && ($3<$5 || $6<$2)' > ${DATADIR}/${NAME}.std.bedpe
bash ${DIR}/process_pos.sh ${DATADIR}/${NAME}.std.bedpe $DNASE 500 $NAME ${DATADIR}

echo "Generating random anchor pairs"
python ${DIR}/generate_random_anchor_pairs.py $NAME $DATADIR
echo "Generating random TF peak pairs"
python ${DIR}/generate_random_pairs_bed.py $NAME $TFPEAKS tf ${DATADIR}
echo "Generating random DNase pairs"
python ${DIR}/generate_random_pairs_bed.py $NAME $DNASE dnase ${DATADIR}

echo "Filtering TF peak pairs"
pairToPair -a ${DATADIR}/${NAME}.random_tf_peak_pairs.bedpe -b $INTERS -type notboth  \
    | pairToPair -a stdin -b  ${DATADIR}/${NAME}.no_intra_all.negative_pairs.bedpe   -type notboth \
    | pairToPair -a stdin -b  ${DATADIR}/${NAME}.only_intra_all.negative_pairs.bedpe  -type notboth \
    | uniq > ${DATADIR}/${NAME}.random_tf_peak_pairs.filtered.bedpe


echo "Filtering DNase pairs"
pairToPair -a ${DATADIR}/${NAME}.shuffled_neg_anchor.neg_pairs.bedpe -b $INTERS -type notboth \
    | pairToPair -a stdin -b  ${DATADIR}/${NAME}.no_intra_all.negative_pairs.bedpe -type notboth \
    | pairToPair -a stdin -b  ${DATADIR}/${NAME}.only_intra_all.negative_pairs.bedpe -type notboth \
    | uniq \
    | pairToPair -a stdin -b ${DATADIR}/${NAME}.random_tf_peak_pairs.filtered.bedpe -type notboth \
    | uniq > ${DATADIR}/${NAME}.shuffled_neg_anchor.neg_pairs.filtered.tf_filtered.bedpe

echo "Sampling 5x negative samples"
python ${DIR}/generate_5fold_neg.py $NAME ${DATADIR}

echo "Generate extended dataset of negative samples"
cat ${DATADIR}/${DATADIR}/${NAME}.{only,no}_intra_all.negative_pairs.bedpe \
    ${DATADIR}/${NAME}.random_tf_peak_pairs.filtered.bedpe \
    | pairToPair -a stdin -b ${DATADIR}/${NAME}.neg_pairs_5x.from_singleton_inter_tf_random.bedpe -type notboth \
    | sort -u > ${DATADIR}/${NAME}.extended_negs_with_intra.bedpe
