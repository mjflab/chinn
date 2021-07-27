i=3000
j=1000
factor="ctcf";

cat ../gm12878_dnase.non_black.bed | cut -f 1-3 | sort -k1,1 -k2,2n -k3,3n | mergeBed -d ${i} > gm12878_dnase.merged_${i}.bed

python ~/repos/chinn/generate_pairs_from_peaks.py gm12878_dnase.merged_${i}.bed gm12878_dnase.merged_${i}.bedpe

pairToPair -a gm12878_dnase.merged_${i}.bedpe -b ../../organized_v2/raw/interactions/TangZ_etal.Cell2015.ChIA-PET_GM12878_CTCF.published_PET_clusters.no_black.txt -is -type both | cut -f 1-6 | uniq > gm12878_dnase.merged_${i}.${factor}_pos.bedpe

pairToPair -a gm12878_dnase.merged_${i}.bedpe -b ../../organized_v2/raw/interactions/TangZ_etal.Cell2015.ChIA-PET_GM12878_CTCF.published_PET_clusters.no_black.txt -is -type notboth | cut -f 1-6 | uniq > gm12878_dnase.merged_${i}.${factor}_neg.bedpe

for t in pos neg; do cat gm12878_dnase.merged_${i}.${factor}_${t}.bedpe | awk '$1=="chr5" || $1=="chr14"' > gm12878_dnase.merged_${i}.${factor}_${t}.val.bedpe; cat gm12878_dnase.merged_${i}.${factor}_${t}.bedpe | awk '$1=="chr4" || $1=="chr7" || $1=="chr8" || $1=="chr11"' > gm12878_dnase.merged_${i}.${factor}_${t}.test.bedpe; cat gm12878_dnase.merged_${i}.${factor}_${t}.bedpe | awk '!($1=="chr4" || $1=="chr7" || $1=="chr8" || $1=="chr11" || $1=="chr5" || $1=="chr14")' > gm12878_dnase.merged_${i}.${factor}_${t}.train.bedpe; done

for s in val test train; do python ~/repos/chinn/predict_bedpe.py -c ~/prediction/organized_v2/final_models/gm12878_${factor}_extended.gbt.pkl -m ~/prediction/organized_v2/final_models/gm12878_${factor}_nodistance.model.pt --pos_files gm12878_dnase.merged_${i}.${factor}_pos.${s}.bedpe --neg_files gm12878_dnase.merged_${i}.${factor}_neg.${s}.bedpe --output_pre gm12878_${factor}.merged_${i}_extended_${j}.${s} -d -e ${j} -g ~/genome/hg19all.fa -b 200 --min_size 1000 --store_factor_outputs; done
