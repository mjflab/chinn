# Chromatin Interaction Neural Network (ChINN): 
## A machine learning-based method for predicting chromatin interactions from DNA sequences 

### This document provides a brief intro of running models in ChINN for training and testing. Before launching any job, make sure you have properly downloaded the ChINN code and you have prepared the dataset following DATASET.md with the correct format.

Chromatin Interaction Neural Network (ChINN) only uses DNA sequences of the interacting open chromatin regions. ChINN is able to predict CTCF-, RNA polymerase II- and HiC- associated chromatin interactions between open chromatin regions. 

ChINN was able to identify convergent CTCF motifs, AP-1 transcription family member motifs such as FOS, and other transcription factors such as MYC as being important in predicting chromatin interactions.

ChINN also shows good across-sample performances and captures various sequence features that are predictive of chromatin interactions. 

### Get the source code

```shell
git clone https://github.com/mjflab/chinn

#enter the directory. Use the repository's root directory as working directory.
cd chinn
```
The Python version tested is 3.6.5. The environment is specified in `environment.yaml`, 
which can be used by Anaconda to create
a new environment by `conda env create -f environment.yml`.

### Dataset:
The models were trained on GM12878 CTCF, GM12878 RNA Pol II, HelaS3 CTCF, K562 RNA Pol II, 
and MCF-7 RNA Pol II datasets separately. The data used to generate the datasets and
build the modles are placed in the `data/` folder. The `data/` folder has the following structure.

```text
data
├── consensusBlacklist.bed # consensus blacklist regions from ENCODE
├── gm12878_ctcf
│   ├── TangZ_etal.Cell2015.ChIA-PET_GM12878_CTCF.published_PET_clusters.no_black.txt
│   ├── wgEncodeAwgDnaseUwdukeGm12878UniPk.narrowPeak
│   └── wgEncodeAwgTfbsBroadGm12878CtcfUniPk.narrowPeak
├── gm12878_polr2a
│   ├── TangZ_etal.Cell2015.ChIA-PET_GM12878_POLR2A.published_PET_clusters.no_black.txt
│   ├── wgEncodeAwgDnaseUwdukeGm12878UniPk.narrowPeak
│   └── wgEncodeAwgTfbsHaibGm12878Pol2Pcr2xUniPk.narrowPeak
├── helas3_ctcf
│   ├── TangZ_etal.Cell2015.ChIA-PET_HelaS3_CTCF.published_PET_clusters.no_black.txt
│   ├── wgEncodeAwgDnaseUwdukeHelas3UniPk.narrowPeak
│   └── wgEncodeAwgTfbsBroadHelas3CtcfUniPk.narrowPeak
├── hg19.len # chromosome lengths
├── k562_polr2a
│   ├── k562_polr2a.interactions.all.non_black.bedpe
│   ├── wgEncodeAwgDnaseUwdukeK562UniPk.narrowPeak
│   └── wgEncodeAwgTfbsSydhK562Pol2UniPk.narrowPeak
└── mcf7_polr2a
    ├── mcf7_polr2a.interactions.all.non_black.bedpe
    ├── wgEncodeAwgDnaseUwdukeMcf7UniPk.narrowPeak
    └── wgEncodeAwgTfbsUtaMcf7Pol2UniPk.narrowPeak
```

We will walk through an example with GM12878 CTCF dataset.

__Note__: to run the scripts, use the root directory of this repository as the working directory.

### Data generation and preprocessing
The data generation and preprocessing scripts are placed under the `preprocess` directory.
The main entry script is `pipe.sh`. This script will process the interactions, cluster the interactions,
generate negative samples, generate distance-matched negative dataset, and the extended
negative datasets. The details of the inputs to the script is shown below:
```
pipe.sh [-h] INTERS DNASE TFPEAKS NAME DATADIR
-- Progam to preprocess the interactions and generate negative samples.
where:
-h           show this help text
INTERS       Interaction file in BEDPE format
DNASE        Dnase/open chromatin regions in BED format
TFPEAKS      The transcription factor peaks for the ChIA-PET protein in BED format
NAME         The prefix/name for the sample/experiment
DATADIR      Location of the output directory
```

__Using GM12878 CTCF dataset as an example.__
```shell
# prepare output directory
mkdir out_dir
bash preprocess/pipe.sh data/gm12878_ctcf/TangZ_etal.Cell2015.ChIA-PET_GM12878_CTCF.published_PET_clusters.no_black.txt \
                        data/gm12878_ctcf/wgEncodeAwgDnaseUwdukeGm12878UniPk.narrowPeak \
                        data/gm12878_ctcf/wgEncodeAwgTfbsBroadGm12878CtcfUniPk.narrowPeak \
                        gm12878_ctcf \
                        out_dir
```

Running the above commands will generate the following files in the `out_dir`:
```shell
gm12878_ctcf.std.bedpe # After removing chromatin interactions whose anchors are overlapping.
gm12878_ctcf_merged_anchors.bed # Merged anchors of the chromatin interactions.
gm12878_ctcf.clustered_interactions.bedpe # Clustered interactions based on merged anchors.
gm12878_ctcf.clustered_interactions.both_dnase.bedpe # Clustered interactions whose both anchors overlap with DNase peaks.
gm12878_ctcf_merged_anchors.both_dnase.bed # The resulting anchors of the chromatin interactions in the above step.
gm12878_ctcf.no_intra_all.negative_pairs.bedpe # Putative negative anchor pairs that are not indirectly connected.
gm12878_ctcf.only_intra_all.negative_pairs.bedpe # Putative negative anchor pairs that are indirectly connected.
gm12878_ctcf.random_tf_peak_pairs.bedpe # Random pairs of TF peaks.
gm12878_ctcf.random_tf_peak_pairs.filtered.bedpe # Random pairs of TF peaks that are not in anchor pairs.
gm12878_ctcf.shuffled_neg_anchor.neg_pairs.bedpe # Random pairs of DNase regions.
gm12878_ctcf.shuffled_neg_anchor.neg_pairs.filtered.tf_filtered.bedpe # Random pairs of DNase regions that are not in anchor pairs or TF peak pairs.
gm12878_ctcf.neg_pairs_5x.from_singleton_inter_tf_random.bedpe # Final sampled 5x negative samples.
gm12878_ctcf.extended_negs_with_intra.bedpe # The additional negative samples for extended dataset.
```

### Distance-matched models

#### Prepare data
To prepare data for training the distance-mached models, first need to prepare the 
reference genome sequences. The reference genome sequences can be downloaded and placed in the `data/` folder. 
In this example, hg19 assembly is used. It needs to be indexed by samtools. 

```shell
# assuming the reference genome assembly is hg19.fa
samtools index hg19.fa
```

The input data can then be prepared using the `data_preparation.py` script.
Use `PYTHONPATH=. python data_preparation.py -h` will generate the help message.
```shell
usage: data_preparation.py [-h] -m MIN_SIZE [-e EXT_SIZE] -n NAME -g GENOME -o
                           OUT_DIR [--pos_files [POS_FILES [POS_FILES ...]]]
                           [--neg_files [NEG_FILES [NEG_FILES ...]]] [-t]
                           [--out_test_only] [--no_test]

generate hdf5 files for prediction. The validation chromosomes are 5, 14. The
test chromosomes are 4, 7, 8, 11. Rest will be training.

optional arguments:
  -h, --help            show this help message and exit
  -m MIN_SIZE, --min_size MIN_SIZE
                        minimum size of anchors to use
  -e EXT_SIZE, --ext_size EXT_SIZE
                        extension size of anchors to use
  -n NAME, --name NAME  The prefix of the files.
  -g GENOME, --genome GENOME
                        The fasta file of reference genome.
  -o OUT_DIR, --out_dir OUT_DIR
                        The output directory.
  --pos_files [POS_FILES [POS_FILES ...]]
                        The positive files
  --neg_files [NEG_FILES [NEG_FILES ...]]
                        The negative files
  -t, --all_test        Use all data for test.
  --out_test_only       Produce only the test data but not validation and
                        training data. Will be ignored if -t/--all_test is
                        set.
  --no_test             Produce no test data but validation and training data.
                        Will be ignored if -t/--all_test or --out_test_only is
                        set.
```

For GM12878 CTCF dataset
```shell
PYTHONPATH=. python data_preparation.py -m 1000 -e 500 \
                      --pos_files out_dir/gm12878_ctcf.clustered_interactions.both_dnase.bedpe \
                      --neg_files out_dir/gm12878_ctcf.neg_pairs_5x.from_singleton_inter_tf_random.bedpe \
                      -g data/hg19.fa \
                      -n gm12878_ctcf_distance_matched -o out_dir
```
Adding `PYTHONPATH=.` will include the current working directory in the search path and
allow proper importing of relevant libraries. This will produce 3 files for training, validation, and testing, respectively.
```shell
gm12878_ctcf_distance_matched_singleton_tf_with_random_neg_seq_data_length_filtered_train.hdf5
gm12878_ctcf_distance_matched_singleton_tf_with_random_neg_seq_data_length_filtered_valid.hdf5
gm12878_ctcf_distance_matched_singleton_tf_with_random_neg_seq_data_length_filtered_test.hdf5
```
### 

#### Train the distance-matched models
The scripts used to train the distance-mached models are placed in
`train_distance_matched`. The training script `train_distance_matched.py`
has the following arguments.
```shell
usage: train_distance_matched.py [-h] [-e EPOCHS] [-s] [-d]
                                 data_name model_name model_dir

Train distance matched models

positional arguments:
  data_name             The name of the data
  model_name            The prefix of the output model.
  model_dir             Directory for storing the models.

optional arguments:
  -h, --help            show this help message and exit
  -e EPOCHS, --epochs EPOCHS
                        Number of epochs for training. Default: 40
  -s, --sigmoid         Use Sigmoid at end of feature extraction. Tanh will be
                        used by default. Default: False.
  -d, --distance        Include distance as a feature for classifier. Default:
                        False.
```

For the GM12878 CTCF dataset, we can train the distance-matched models by
```shell
PYTHONPATH=. python train_distance_matched/train_distance_matched.py \
                       out_dir/gm12878_ctcf_distance_matched_singleton_tf_with_random_neg_seq_data_length_filtered \
                       gm12878_ctcf_model \
                       out_dir
```
Note that we have not included distance as a feature for classifer.

This will generate these files in the output directory
```shell
gm12878_ctcf_model.model.pt # The CNN model 
gm12878_ctcf_model.classifier.pt  # The classifier part.
gm12878_ctcf_model_re.model.pt # The CNN model after retraining
gm12878_ctcf_model_re.classifier.pt  # The classifer part after retraining.
```

### Extended models
#### Prepare data for training classifiers using extended datasets.
First generate one-hot encoded data.
```shell
PYTHONPATH=. python data_preparation.py -m 1000 -e 500 \
                --pos_files out_dir/gm12878_ctcf.clustered_interactions.both_dnase.bedpe \
                --neg_files out_dir/gm12878_ctcf.neg_pairs_5x.from_singleton_inter_tf_random.bedpe \
                            out_dir/gm12878_ctcf.extended_negs_with_intra.bedpe \
                -g path/to/hg19.fa \
                -n gm12878_ctcf_extended -o out_dir
```
This will produce these files for train, validation, and testing in the output directory:
```shell
gm12878_ctcf_extended_singleton_tf_with_random_neg_seq_data_length_filtered_train.hdf5
gm12878_ctcf_extended_singleton_tf_with_random_neg_seq_data_length_filtered_valid.hdf5
gm12878_ctcf_extended_singleton_tf_with_random_neg_seq_data_length_filtered_test.hdf5 
```

Since at this stage, we are not training the CNN part of the model, 
the outputs of the CNN part can be computed to speed up training of
the classifiers. This can be done by calling the `generate_factor_output.py` script.
This script has the following usage information:
```text
usage: Perform test [-h] [-s] [-d] [--same] model data_file out_pre

positional arguments:
  model               The model
  data_file           The data file
  out_pre             The output file prefix

optional arguments:
  -h, --help          show this help message and exit
  -s, --sigmoid       use sigmoid after weightsum
  -d, --use_distance  use distance
  --same              Use the same subsequence for all features
```

For the GM12878 CTCF dataset, we will do the following
```shell
for i in train valid test; 
do 
  PYTHONPATH=. python generate_factor_output.py \
                  out_dir/gm12878_ctcf_model_re.model.pt \
                  out_dir/gm12878_ctcf_distance_matched_singleton_tf_with_random_neg_seq_data_length_filtered_${i}.hdf5 \
                  gm1278_ctcf_${i}; 
done
```
This will generate the following files in the output directory:
```text
gm12878_ctcf_extended_with_intra_test_factor_outputs.hdf5   
gm12878_ctcf_extended_with_intra_valid_factor_outputs.hdf5
gm12878_ctcf_extended_with_intra_train_factor_outputs.hdf5
```

#### Train an XGBoost classifier using extended datasets
Next step is to train the classifier using the extended datasets. The script
to train the extended classifier is placed under `train_extended/`. The `train_extended.py` has 
the following usage information:
```text
usage: train_extended.py [-h] data_dir dataset_name model_dir

Train classifiers using extended datasets.

positional arguments:
  data_dir      The directory of the data location with train, valid, and
                test.
  dataset_name  The name (prefix) of the dataset before
                _[train|valid|test]_factor_outputs.hdf5. For example:
                gm12878_ctcf_train_factor_outputs.hdf5 -> gm12878_ctcf
  model_dir     The directory to store the models. 3 models will be generated:
                1. using all features; 2. using all features but distance
                (_nodist); 3. using distance only (_dist_only).

optional arguments:
  -h, --help    show this help message and exit
```

For the GM12878 CTCF example, the command is as follows:
```shell
PYTHONPATH=. python train_extended/train_extended.py out_dir gm12878_ctcf_extended_with_intra out_dir/
```
This command will produce several files in the model directory:
```text
# tree depth of 3
gm12878_ctcf_extended_with_intra_depth3_nodist.gbt.pkl      # without distance
gm12878_ctcf_extended_with_intra_depth3_dist_only.gbt.pkl   # only distance
gm12878_ctcf_extended_with_intra_depth3.gbt.pkl             # all features 
# tree depth of 6  
gm12878_ctcf_extended_with_intra_depth6_nodist.gbt.pkl
gm12878_ctcf_extended_with_intra_depth6_dist_only.gbt.pkl
gm12878_ctcf_extended_with_intra_depth6.gbt.pkl
# tree depth of 10
gm12878_ctcf_extended_with_intra_depth10_nodist.gbt.pkl     
gm12878_ctcf_extended_with_intra_depth10_dist_only.gbt.pkl  
gm12878_ctcf_extended_with_intra_depth10.gbt.pkl            
```

To get the results on the test dataset, the `predict.py` script can be used.
The script has the following usage message:
```text
usage: Perform prediction using data generated by data_preparation.py
       [-h] -m MODEL_FILE -c CLASSIFIER_FILE --data_file DATA_FILE
       --output_pre OUTPUT_PRE [-s] [-d] [--same] [--store_factor_outputs]
       [--legacy]

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL_FILE, --model_file MODEL_FILE
                        The model prefix
  -c CLASSIFIER_FILE, --classifier_file CLASSIFIER_FILE
                        The classifier
  --data_file DATA_FILE
                        The data file
  --output_pre OUTPUT_PRE
                        The output file prefix
  -s, --sigmoid         use sigmoid after weightsum. Default: False
  -d, --use_distance    use distance. Default: False
  --same                Use the same subsequence for all features. Default:
                        False
  --store_factor_outputs
                        Whether to store the factor outputs. Default: False
  --legacy              Whether to use legacy (softmax) for classifier.
```

On the Gm12878 CTCF extended test dataset, we can run
```shell
PYTHONPATH=. python predict.py -m out_dir/gm12878_ctcf_model_re.model.pt \
                -c out_dir/gm12878_ctcf_extended_with_intra_depth6.gbt.pkl \
                --data_file out_dir/gm12878_ctcf_extended_singleton_tf_with_random_neg_seq_data_length_filtered_test.hdf5 \
                --output_pre out_dir/gm12878_ctcf_extended_test -d
```
which will generate `gm12878_ctcf_extended_test_probs.txt` in the output directory.

### From DNase model
#### Data preparation
Create merged anchors from DNase regions and generate anchor pairs. 
For GM12878 CTCF
```shell
intersectBed -a data/gm12878_ctcf/wgEncodeAwgDnaseUwdukeGm12878UniPk.narrowPeak -b data/consensusBlacklist.bed -v \
   | cut -f 1-3 | cut -f 1-3 | sort -k1,1 -k2,2n -k 3,3n | mergeBed -d 3000 > out_dir/gm12878_dnase_merged_3000.bed
python generate_pairs_from_bed.py out_dir/gm12878_dnase_merged_3000.bed out_dir/gm12878_dnase_merged_3000.bedpe
```

Generate positive pairs and negative pairs based on overlapping with chromatin interactions.
For the GM12878 CTCF dataset:
```
pairToPair -a out_dir/gm12878_dnase_merged_3000.bedpe -b data/gm12878_ctcf/TangZ_etal.Cell2015.ChIA-PET_GM12878_CTCF.published_PET_clusters.no_black.txt -type both \
        | cut -f 1-6 | uniq > out_dir/gm12878_ctcf_dnase_in_ctcf_merged3000.pos.bedpe; 
pairToPair -a out_dir/gm12878_dnase_merged_3000.bedpe -b data/gm12878_ctcf/TangZ_etal.Cell2015.ChIA-PET_GM12878_CTCF.published_PET_clusters.no_black.txt -type notboth \
        | cut -f 1-6 | uniq > out_dir/gm12878_ctcf_dnase_in_ctcf_merged3000.neg.bedpe; 
```
Next step would be to run the `data_preparation.py` and `generate_factor_output.py` as in the extended dataset step.

#### Training
Training the from-dnase model is similar to training model on extended datasets.

### For prediction 
Given a BED file containing peaks of open chromatin regions, all possible pairs 
on the same chromosome can be generated following the 'Data preparation' step in the 'From DNase model'
section.

After generating the BEDPE file, the probabilities of interactions for these anchor pairs can be generated
using `predict_bedpe.py`. This script has the following usage information:
```text
usage: Perform prediction [-h] -m MODEL_FILE -c CLASSIFIER_FILE
                          [--pos_files [POS_FILES [POS_FILES ...]]]
                          [--neg_files [NEG_FILES [NEG_FILES ...]]]
                          --output_pre OUTPUT_PRE [-s] [-d] [--same]
                          [--store_factor_outputs] --min_size MIN_SIZE
                          [-e EXT_SIZE] [-g GENOME] [-b BATCH_SIZE]
                          [--inter_chrom]
                          [--breakpoints [BREAKPOINTS [BREAKPOINTS ...]]]
                          [--crispr CRISPR] [--no_classifier]

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL_FILE, --model_file MODEL_FILE
                        The model prefix
  -c CLASSIFIER_FILE, --classifier_file CLASSIFIER_FILE
                        The classifier
  --pos_files [POS_FILES [POS_FILES ...]]
                        The positive files
  --neg_files [NEG_FILES [NEG_FILES ...]]
                        The negative files
  --output_pre OUTPUT_PRE
                        The output file prefix
  -s, --sigmoid         use sigmoid after weightsum. Default: False
  -d, --use_distance    use distance. Default: False
  --same                Use the same subsequence for all features. Default:
                        False
  --store_factor_outputs
                        Whether to store the factor outputs. Default: False
  --min_size MIN_SIZE   minimum size of anchors to use
  -e EXT_SIZE, --ext_size EXT_SIZE
                        extension size of anchors to use
  -g GENOME, --genome GENOME
                        The fasta file of reference genome.
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        The batch size to use. Default: 500
  --inter_chrom         Whether the pairs are inter-chromosome
  --breakpoints [BREAKPOINTS [BREAKPOINTS ...]]
                        Breakpoints locations.
  --crispr CRISPR       Breakpoints locations.
  --no_classifier       Do not invoke classifer, only factor outputs will be
                        computed.
```

For GM12878 CTCF, an example to generate all the predictions:
```shell
PYTHONPATH=. python predict_bedpe.py -m out_dir/gm12878_ctcf_model_re.model.pt \
                -c out_dir/gm12878_ctcf_extended_with_intra_depth6.gbt.pkl \
                --pos_files out_dir/gm12878_ctcf_dnase_in_ctcf_merged3000.pos.bedpe \
                --neg_files out_dir/gm12878_ctcf_dnase_in_ctcf_merged3000.neg.bedpe \
                -g path/to/hg19.fa \
                --min_size 1000 -e 500 -d \
                --output_pre out_dir/gm12878_ctcf_from_dnase_all
```
