import os,sys
import argparse
from chinn.models import PartialDeepSeaModel, NNClassifier
from chinn import train
import torch


def get_args():
    parser = argparse.ArgumentParser(description="Train distance matched models")
    parser.add_argument('data_name', help='The prefix of the data (without _[train|valid|test].hdf5')
    parser.add_argument('model_name', help="The prefix of the model.")
    parser.add_argument('model_dir', help='Directory for storing the models.')
    parser.add_argument('-s', '--sigmoid', action='store_true', default=False,
                        help='Use Sigmoid at end of feature extraction. Tanh will be used by default. Default: False.')
    parser.add_argument('-d', '--distance', action='store_true', default=False,
                        help='Include distance as a feature for classifier. Default: False.')

    return parser.parse_args()


if __name__=='__main__':
    args = get_args()
    legacy = True

    deepsea_model = PartialDeepSeaModel(4, use_weightsum=True, leaky=True, use_sigmoid=args.sigmoid)
    n_filters = deepsea_model.num_filters[-1]*4
    if args.distance:
        n_filters += 1
    classifier = NNClassifier(n_filters, legacy=legacy)

    train.test(deepsea_model, classifier, args.model_name, args.data_name, False, data_set='test',
               save_probs=True, use_distance=False, model_dir=args.model_dir, legacy=legacy, plot=False)
