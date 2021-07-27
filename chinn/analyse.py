import torch
import train
from models import PartialDeepSeaModel
import numpy as np
import time


def compute_one_side(data, edges, model: PartialDeepSeaModel, feature_size, f_mat, rc_mat, evaluation=False, level=3):

    def get_counts(x_in, count_mat):
        '''curr_outputs = model.get_conv_activations(x, level).data#.cpu().numpy()

        results = torch.zeros((model.num_filters[2], 4, feature_size)).cuda()
        for i in range(len(edges)-1):
            n_max, n_max_idx = torch.max(curr_outputs[edges[i]:edges[i+1],:], dim=0, keepdim=False)
            #n_max = np.max(curr_outputs[edges[i]:edges[i+1],:], axis=0)
            l_max, l_max_idx = torch.max(n_max, dim=1, keepdim=False)
            #l_max = np.max(n_max, axis=1)
            n_idx = [n_max_idx[j][l_max_idx[j]] for j in range(model.num_filters[2])]
            for j in range(model.num_filters[2]):
                if l_max[j] > 0:
                    lower = l_max_idx[j]*4**(level-1)
                    results[j] += x.data[n_idx[j], :, lower:lower+feature_size]'''

        curr_outputs = model.get_conv_activations(x_in, level).data.cpu().numpy()

        results = np.zeros((model.num_filters[level-1], 4, feature_size))
        temp_data = x_in.data.cpu().numpy()
        for i in range(len(edges)-1):
            n_max_idx = np.argmax(curr_outputs[edges[i]:edges[i+1],:], axis=0)
            n_max = np.max(curr_outputs[edges[i]:edges[i+1],:], axis=0)
            l_max_idx = np.argmax(n_max, axis=1)
            l_max = np.max(n_max, axis=1)
            n_idx = [n_max_idx[j][l_max_idx[j]] for j in range(model.num_filters[level-1])]
            for j in range(model.num_filters[level-1]):
                if l_max[j] > 0:
                    lower = l_max_idx[j]*(4**(level-1))
                    results[j] += temp_data[n_idx[j] + edges[i], :, lower:lower+feature_size]
        count_mat += results

    x = torch.autograd.Variable(torch.from_numpy(data).float(), volatile=evaluation).cuda()
    x_rc = torch.autograd.Variable(torch.from_numpy(np.array(data[:, ::-1, ::-1])).float(), volatile=evaluation).cuda()
    edges = np.array(edges) - edges[0]
    get_counts(x, f_mat)
    get_counts(x_rc, rc_mat)


def get_feature_logos(model: PartialDeepSeaModel, level: int, model_name: str, data_pre: str, stop_after=None):
    (test_data, test_left_data, test_right_data,
     test_left_edges, test_right_edges,
     test_labels, test_dists) = train.load_hdf5_data("%s_test.hdf5" % data_pre)
    model.load_state_dict(torch.load("%s.model.pt" % model_name))
    start = 0
    feature_size = 0
    for i in range(level - 1):
        feature_size += 7
        feature_size = (feature_size + 1) * 4 - 1
    feature_size += 8
    start_time = time.time()
    last_print = 0
    left_f = np.zeros((model.num_filters[level-1], 4, feature_size))
    left_rc = np.zeros_like(left_f)
    right_f = np.zeros_like(left_f)
    right_rc = np.zeros_like(left_f)
    while start < len(test_left_edges) - 1:
        (end, curr_left_data, curr_left_edges, curr_right_data,
         curr_right_edges, curr_dists, curr_labels) = train.get_data_batch(test_left_data,
                                                                           test_left_edges,
                                                                           test_right_data,
                                                                           test_right_edges,
                                                                           test_dists,
                                                                           test_labels,
                                                                           start,
                                                                           limit_to_one=False)

        compute_one_side(curr_left_data, curr_left_edges, model, feature_size, left_f, left_rc,
                         evaluation=True, level=level)
        compute_one_side(curr_right_data, curr_right_edges, model, feature_size, right_f, right_rc,
                         evaluation=True, level=level)

        start = end
        if end - last_print > 2000:
            print(time.time() - start_time, end)
            last_print = end
        if stop_after is not None and end > stop_after:
            break

    return left_f, left_rc, right_f, right_rc
