import h5py
import numpy as np
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import time
from six.moves import cPickle
import logging
import sys
from collections import Counter
import torch
from chinn.models import *

def get_data_batch(left_data, left_edges, right_data, right_edges, dists, labels, start,
                   max_size=300, limit_to_one=False):
    end_idx = start + 1
    i = start + 1
    if not limit_to_one:
        while i < len(left_edges) - 1:
            if max(left_edges[i] - left_edges[start], right_edges[i] - right_edges[start]) > max_size:
                break
            i += 1
    end_idx = max(end_idx, i - 1)
    curr_left = left_data[left_edges[start]:left_edges[end_idx]]
    curr_left_edges = left_edges[start:(end_idx + 1)]

    curr_right = right_data[right_edges[start]:right_edges[end_idx]]
    curr_right_edges = right_edges[start:(end_idx + 1)]

    curr_labels = labels[start:end_idx]
    curr_dists = dists[start:end_idx]
    return end_idx, curr_left, curr_left_edges, curr_right, curr_right_edges, curr_dists, curr_labels


def __get_max(curr_out, start, end):
    curr_max = torch.max(curr_out[start:end, :], dim=0)
    curr_idxes = curr_max[1].data.cpu().numpy()

    counts = Counter(curr_idxes)
    items = sorted(counts.items())
    largest_idx = 0
    max_count = 0
    if len(items) > 4:
        for i in range(len(items) - 4):
            curr_count = sum([x[1] for x in items[i:i+5]])
            if curr_count > max_count:
                max_count = curr_count
                largest_idx = i
    selected_idx = start + largest_idx
    return selected_idx


def compute_one_side(data, edges, model, evaluation=False, same=False):
    x = torch.autograd.Variable(torch.from_numpy(data).float()).cuda()
    x_rc = torch.autograd.Variable(torch.from_numpy(np.array(data[:,::-1,::-1])).float()).cuda()
    edges = np.array(edges) - edges[0]
    combined = []
    if not same:
        curr_outputs = torch.cat((model(x), model(x_rc)), dim=1)
        for i in range(len(edges) - 1):
            combined.append(torch.max(curr_outputs[edges[i]:edges[i + 1], :], dim=0, keepdim=True)[0])
    else:
        f_out = model(x)
        rc_out = model(x_rc)
        for i in range(len(edges)-1):
            selected_idx_f = __get_max(f_out, edges[i], edges[i+1])
            selected_idx_rc = __get_max(rc_out, edges[i], edges[i+1])
            curr_outputs = torch.cat((torch.max(f_out[selected_idx_f:min(selected_idx_f+5, edges[i+1]), :],
                                                dim=0, keepdim=True)[0],
                                      torch.max(rc_out[selected_idx_rc:min(selected_idx_rc+5, edges[i+1]), :],
                                                dim=0, keepdim=True)[0]),
                                     dim=1)
            combined.append(curr_outputs)
    out = torch.cat([x for x in combined], dim=0)
    return out


def compute_factor_output(left_data, left_edges, right_data, right_edges,
                          dists, labels, start, evaluation, factor_model, max_size=300,
                          limit_to_one=False, same=False, legacy=False):
    (end, curr_left_data, curr_left_edges, curr_right_data,
     curr_right_edges, curr_dists, curr_labels) = get_data_batch(left_data,
                                                                 left_edges,
                                                                 right_data,
                                                                 right_edges,
                                                                 dists,
                                                                 labels,
                                                                 start,
                                                                 max_size=max_size,
                                                                 limit_to_one=limit_to_one)
    left_out = compute_one_side(curr_left_data, curr_left_edges, factor_model, evaluation, same=same)
    right_out = compute_one_side(curr_right_data, curr_right_edges, factor_model, evaluation, same=same)
    if legacy:
        curr_labels = torch.autograd.Variable(torch.from_numpy(curr_labels).long()).cuda()
    else:
        curr_labels = torch.autograd.Variable(torch.from_numpy(curr_labels).float()).cuda()
    curr_dists = torch.autograd.Variable(torch.from_numpy(np.array(curr_dists, dtype='float32'))).cuda()
    return end, left_out, right_out, curr_dists, curr_labels


def apply_classifier(classifier, left_out, right_out, curr_dists, input_grad=False, use_distance=True):

    if use_distance:
        if len(curr_dists.size()) == 1:
            curr_dists = curr_dists.view(-1, 1)
        combined1 = torch.cat((left_out, right_out, curr_dists), dim=1)
    else:
        combined1 = torch.cat((left_out, right_out), dim=1)

    if input_grad:
        combined1 = torch.nn.Parameter(combined1.data)
        out1 = classifier(combined1)
        return out1, combined1
    else:
        out1 = classifier(combined1)
        return out1


def predict(model, classifier, loss_fn,
            valid_left_data, valid_left_edges,
            valid_right_data, valid_right_edges,
            valid_dists, valid_labels, return_prob=False, use_distance=True,
            use_metrics=True, max_size=300, verbose=0, same=False, legacy=False,
            plot=True):

    model.eval()
    classifier.eval()
    val_err = 0.
    val_samples = 0
    all_probs = []
    last_print = 0
    edge = 0
    with torch.no_grad():
        while edge < len(valid_left_edges) - 1:
            end, left_out, right_out, curr_dists, curr_labels = compute_factor_output(valid_left_data, valid_left_edges,
                                                                                      valid_right_data, valid_right_edges,
                                                                                      valid_dists, valid_labels, edge, True,
                                                                                      model, max_size=max_size, same=same, legacy=legacy)
            if verbose > 0:
                logging.info(str(curr_dists.size()))
            curr_outputs = apply_classifier(classifier, left_out, right_out, curr_dists, use_distance=use_distance)

            loss = loss_fn(curr_outputs, curr_labels)
            if legacy:
                if int(torch.__version__.split('.')[1]) > 2:
                    val_predictions = F.softmax(curr_outputs, dim=1).data.cpu().numpy()
                else:
                    val_predictions = F.softmax(curr_outputs).data.cpu().numpy()
                val_predictions = val_predictions[:,1]
            else:
                val_predictions = torch.sigmoid(curr_outputs).data.cpu().numpy()
            all_probs.append(val_predictions)

            val_err += loss.data.item() * (end - edge)

            val_samples += end - edge
            if verbose > 0 and end - last_print > 10000:
                logging.info(str(end))
            edge = end

    all_probs = np.concatenate(all_probs)

    if use_metrics:
        c_auprc = [metrics.average_precision_score(valid_labels, all_probs), ]
        c_roc = [metrics.roc_auc_score(valid_labels, all_probs), ]

        logging.info("  validation loss:\t\t{:.6f}".format(val_err / val_samples))
        logging.info("  auPRCs: {}".format("\t".join(map(str, c_auprc))))
        logging.info("  auROC: {}".format("\t".join(map(str, c_roc))))
        all_preds = np.zeros(all_probs.shape[0])
        all_preds[all_probs > 0.5] = 1
        logging.info("  f1: {}".format(str(metrics.f1_score(valid_labels, all_preds))))
        logging.info("  precision: {}".format(str(metrics.precision_score(valid_labels, all_preds))))
        logging.info("  recall: {}".format(str(metrics.recall_score(valid_labels, all_preds))))
        logging.info("  accuracy: {}".format(str(metrics.accuracy_score(valid_labels, all_preds))))
        logging.info("  ratio: {}".format(np.sum(valid_labels) / len(valid_labels)))
        one_prec = metrics.precision_score(valid_labels, np.ones(len(valid_labels)))
        precision, recall, _ = metrics.precision_recall_curve(valid_labels, all_probs, pos_label=1)
        fpr, tpr, _ = metrics.roc_curve(valid_labels, all_probs, pos_label=1)
        if plot:
            plt.plot(fpr, tpr)
            plt.plot(recall, precision)
            plt.axhline(one_prec, color='r')
            plt.xlim(-0.05, 1.05)
            plt.show()
    if return_prob:
        return val_err / val_samples, all_probs
    return val_err / val_samples


def predict_finetune(classifier, loss_fn,
                     left_data_store, right_data_store,
                    valid_dists, valid_labels, return_prob=False, use_distance=True,
                    use_metrics=True, verbose=0):

    classifier.eval()
    val_err = 0.
    val_samples = 0
    all_probs = []
    last_print = 0
    edge = 0
    while edge < len(valid_labels):
        end = edge + 1000
        left_out = torch.autograd.Variable(torch.from_numpy(left_data_store[edge:end])).cuda()
        right_out = torch.autograd.Variable(torch.from_numpy(right_data_store[edge:end])).cuda()
        curr_dists = torch.autograd.Variable(torch.from_numpy(valid_dists[edge:end])).cuda()
        curr_labels = torch.autograd.Variable(torch.from_numpy(valid_labels[edge:end]).long()).cuda()
        if verbose > 0:
            logging.info(str(curr_dists.size()))
        curr_outputs = apply_classifier(classifier, left_out, right_out, curr_dists, use_distance=use_distance)

        loss = loss_fn(curr_outputs, curr_labels)
        val_predictions = torch.sigmoid(curr_outputs).data.cpu().numpy()
        all_probs.append(val_predictions)


        val_err += loss.data[0] * (end - edge)

        val_samples += end - edge
        if verbose > 0 and end - last_print > 10000:
            logging.info(str(end))
        edge = end
    all_probs = np.concatenate(all_probs)

    if use_metrics:
        c_auprc = [metrics.average_precision_score(valid_labels, all_probs), ]
        c_roc = [metrics.roc_auc_score(valid_labels, all_probs), ]

        logging.info("  validation loss:\t\t{:.6f}".format(val_err / val_samples))
        logging.info("  auPRCs: {}".format("\t".join(map(str, c_auprc))))
        logging.info("  auROC: {}".format("\t".join(map(str, c_roc))))
        all_preds = np.zeros(all_probs.shape[0])
        all_preds[all_probs > 0.5] = 1
        logging.info("  f1: {}".format(str(metrics.f1_score(valid_labels, all_preds))))
        logging.info("  precision: {}".format(str(metrics.precision_score(valid_labels, all_preds))))
        logging.info("  recall: {}".format(str(metrics.recall_score(valid_labels, all_preds))))
        logging.info("  accuracy: {}".format(str(metrics.accuracy_score(valid_labels, all_preds))))
        logging.info("  ratio: {}".format(np.sum(valid_labels) / len(valid_labels)))
        one_prec = metrics.precision_score(valid_labels, np.ones(len(valid_labels)))
        precision, recall, _ = metrics.precision_recall_curve(valid_labels, all_probs, pos_label=1)
        fpr, tpr, _ = metrics.roc_curve(valid_labels, all_probs, pos_label=1)
        plt.plot(fpr, tpr)
        plt.plot(recall, precision)
        plt.axhline(one_prec, color='r')
        plt.xlim(-0.05, 1.05)
        plt.show()
    if return_prob:
        return val_err / val_samples, all_probs
    return val_err / val_samples


def load_hdf5_data(fn):
    data = h5py.File(fn, 'r')
    left_data = data['left_data']
    right_data = data['right_data']
    left_edges = data['left_edges'][:]
    right_edges = data['right_edges'][:]
    labels = data['labels'][:]
    pairs = data['pairs'][:]
    dists = [[np.log10(abs(p[5] / 5000 - p[2] / 5000 + p[4] / 5000 - p[1] / 5000) * 0.5) / np.log10(2000001 / 5000),] + list(p)[7:]
                   for p in pairs]
    return data,left_data,right_data,left_edges,right_edges,labels,dists


def train(model, classifier, data_pre, model_name, retraining, use_existing=None, epochs=60,
          use_weight_for_training=None, init_lr=0.0002, finetune=False, generate_data=True,
          interval=5000, verbose=0, ranges_to_skip=None, use_distance=True, eps=1e-8, same=False,
          model_dir='/data/protein', legacy=False, plot=False):

    if use_existing is not None:
        model.load_state_dict(torch.load('%s/%s.model.pt' % (model_dir, use_existing)))
        classifier.load_state_dict(torch.load("%s/%s.classifier.pt" % (model_dir,use_existing)))
    elif retraining:
        model.load_state_dict(torch.load('%s/%s.model.pt' % (model_dir,model_name)))
        classifier.load_state_dict(torch.load("%s/%s.classifier.pt" % (model_dir,model_name)))

    model.cuda()
    classifier.cuda()

    if not (finetune and not generate_data):
        (train_data, train_left_data, train_right_data,
         train_left_edges, train_right_edges,
         train_labels, train_dists) = load_hdf5_data("%s_train.hdf5" % data_pre)
        (val_data, valid_left_data, valid_right_data,
         valid_left_edges, valid_right_edges,
         valid_labels, valid_dists) = load_hdf5_data("%s_valid.hdf5" % data_pre)

    if finetune and generate_data:
        model.eval()

        def __generate_factor_outputs(dset, _left_data, _right_data, _left_edges, _right_edges, _labels, _dists):
            _data_store = h5py.File(data_pre + '_%s_factor_outputs.hdf5'%dset, 'w')
            _left_data_store = data_store.create_dataset('left_out', (len(_labels),  model.num_filters[-1]*2),
                                                         dtype='float32', chunks=True, compression='gzip')
            _right_data_store = data_store.create_dataset('right_out', (len(_labels),  model.num_filters[-1]*2),
                                                          dtype='float32', chunks=True, compression='gzip')
            _dist_data_store = data_store.create_dataset('dists', data=_dists, dtype='float32',
                                                         chunks=True, compression='gzip')
            _labels_data_store = data_store.create_dataset('labels', data=_labels, dtype='uint8')
            i = 0
            last_print = 0
            while i < len(train_left_edges) - 1:
                end, _left_out, _right_out, _, _ = compute_factor_output(_left_data, _left_edges,
                                                                      _right_data, _right_edges,
                                                                      _dists, _labels, i,
                                                                      True, factor_model=model, legacy=legacy)
                _left_data_store[i:end] = _left_out.data.cpu().numpy()
                _right_data_store[i:end] = _right_out.data.cpu().numpy()
                if end - last_print > 5000:
                    last_print = end
                    logging.info('generating input : %d / %d', end, len(train_labels))
                i = end
            _data_store.close()

        __generate_factor_outputs('train', train_left_data, train_left_edges, train_right_data, train_right_edges,
                                  train_labels, train_dists)
        __generate_factor_outputs('valid', valid_left_data, valid_left_edges, valid_right_data, valid_right_edges,
                                  valid_labels, valid_dists)

    if finetune:
        data_store = h5py.File(data_pre + '_train_factor_outputs.hdf5', 'r')
        left_data_store = data_store['left_out']
        right_data_store = data_store['right_out']
        dist_data_store = data_store['dists']
        labels_data_store = data_store['labels']
        train_labels = labels_data_store[:]

        valid_data_store = h5py.File(data_pre + '_valid_factor_outputs.hdf5', 'r')
        valid_left_data_store = data_store['left_out']
        valid_right_data_store = data_store['right_out']
        valid_dist_data_store = data_store['dists'][:]
        valid_labels_data_store = data_store['labels'][:]

        logging.info("%s %s %s" % (str(left_data_store.shape), str(right_data_store.shape), str(dist_data_store.shape)))

    rootLogger = logging.getLogger()
    for handler in rootLogger.handlers:
        rootLogger.removeHandler(handler)
    rootLogger.setLevel(logging.INFO)
    logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
    fileHandler = logging.FileHandler('logs/' + model_name + "%s.log"%('_re' if retraining else ''))
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

    logging.info('learning rate: %f, eps: %f' % (init_lr, eps))

    if use_weight_for_training is not None:
        if use_weight_for_training == 'balanced':
            weights = torch.FloatTensor([1, max(1, (len(train_labels) - np.sum(train_labels)) / np.sum(train_labels))]).cuda()
        elif type(use_weight_for_training) == int or type(use_weight_for_training) == float:
            weights = torch.FloatTensor([1, use_weight_for_training]).cuda()
    else:
        weights = torch.FloatTensor([1, 1]).cuda()

    logging.info(str(weights))
    #loss_fn = torch.nn.BCEWithLogitsLoss(weight=weights)
    if legacy:
        loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    else:
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=weights[1:])

    if finetune:
        optimizer = torch.optim.Adam(list(classifier.parameters()),
                                     lr=init_lr, eps=eps,
                                     weight_decay=init_lr * 0.1)
    else:
        optimizer = torch.optim.Adam(list(classifier.parameters()) + list(model.parameters()),
                                     lr=init_lr, eps=eps,
                                     weight_decay=init_lr * 0.1)
    if not finetune:
        best_val_loss = predict(model, classifier, loss_fn,
                                valid_left_data, valid_left_edges,
                                valid_right_data, valid_right_edges,
                                valid_dists, valid_labels, return_prob=False,
                                use_distance=use_distance, verbose=verbose, same=same, 
                                legacy=legacy, plot=plot)
    else:
        best_val_loss = predict_finetune(classifier, loss_fn, valid_left_data_store, valid_right_data_store,
                                         valid_dist_data_store, valid_labels_data_store, return_prob=False,
                                         use_distance=use_distance, verbose=verbose)
    print(best_val_loss)

    last_update = 0
    if ranges_to_skip is not None:
        ranges_to_skip.sort(key=lambda k:(k[0],k[1]))
    for epoch in range(0, epochs):

        start_time = time.time()
        i = 0
        train_loss = 0.
        num_samples = 0

        model.train()
        classifier.train()

        last_print = 0
        curr_loss = 0.
        curr_pos = 0
        while i < len(train_labels):
            if finetune:
                end = i + 400
                left_out = torch.autograd.Variable(torch.from_numpy(left_data_store[i:end])).cuda()
                right_out = torch.autograd.Variable(torch.from_numpy(right_data_store[i:end])).cuda()
                curr_dists = torch.autograd.Variable(torch.from_numpy(dist_data_store[i:end])).cuda()
                curr_labels = torch.autograd.Variable(torch.from_numpy(labels_data_store[i:end]).long()).cuda()
            else:
                end, left_out, right_out, curr_dists, curr_labels = compute_factor_output(train_left_data,
                                                                                          train_left_edges,
                                                                                          train_right_data,
                                                                                          train_right_edges,
                                                                                          train_dists, train_labels, i,
                                                                                          False, model,
                                                                                          same=same, legacy=legacy)

            if ranges_to_skip is not None:
                skip_batch = False
                for (r_s, r_e) in ranges_to_skip:
                    if min(r_e, end) > max(i, r_s):
                        skip_batch = True
                        break
                if skip_batch:
                    i = end
                    continue

            if verbose > 0:
                logging.info(str(curr_dists.size()))
            curr_outputs = apply_classifier(classifier, left_out, right_out, curr_dists, use_distance=use_distance)

            loss = loss_fn(curr_outputs, curr_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            num_samples += end - i
            curr_loss += loss.data.item() * (end - i)
            train_loss += loss.data.item() * (end - i)
            curr_pos += torch.sum(curr_labels).data.item()
            i = end
            if num_samples < 1000 or num_samples - last_print > interval:
                logging.info("%d  %f  %f  %f  %f", i, time.time() - start_time,
                             train_loss / num_samples, curr_loss / (num_samples - last_print),
                             curr_pos*1.0 / (num_samples - last_print))
                curr_pos = 0
                curr_loss = 0
                last_print = num_samples

        logging.info("Epoch {} of {} took {:.3f}s".format(epoch + 1, epochs, time.time() - start_time))
        logging.info("Train loss: %f", train_loss / num_samples)

        if not finetune:
            val_err = predict(model, classifier, loss_fn,
                              valid_left_data, valid_left_edges,
                              valid_right_data, valid_right_edges,
                              valid_dists, valid_labels, return_prob=False,
                              use_distance=use_distance, verbose=verbose, 
                              same=same, legacy=legacy, plot=plot)
        else:
            val_err = predict_finetune(classifier, loss_fn, valid_left_data_store, valid_right_data_store,
                                       valid_dist_data_store, valid_labels_data_store, return_prob=False,
                                       use_distance=use_distance, verbose=verbose)

        if val_err < best_val_loss or epoch == 0:
            best_val_loss = val_err
            last_update = epoch
            logging.info("current best val: %f", best_val_loss)
            torch.save(model.state_dict(),
                       "{}/{}{}.model.pt".format(model_dir, model_name, '_re' if retraining else ''),
                       pickle_protocol=cPickle.HIGHEST_PROTOCOL)
            torch.save(classifier.state_dict(),
                       "{}/{}{}.classifier.pt".format(model_dir, model_name, '_re' if retraining else ''),
                       pickle_protocol=cPickle.HIGHEST_PROTOCOL)
        if epoch - last_update >= 10:
            break
    if not (finetune and not generate_data):
        train_data.close()
        val_data.close()
    fileHandler.close()
    consoleHandler.close()
    rootLogger.removeHandler(fileHandler)
    rootLogger.removeHandler(consoleHandler)


def test(model, classifier, model_name, data_name, return_probs, data_set='valid',
         use_metrics=True, save_probs=False, use_distance=True, num_bins=10,
         data_name_is_filename=False, max_size=300, verbose=0, same=False,
         model_dir='/data/protein', legacy=False, plot=False):

    rootLogger = logging.getLogger()
    for handler in rootLogger.handlers:
        rootLogger.removeHandler(handler)
    rootLogger.setLevel(logging.INFO)
    logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
    fileHandler = logging.FileHandler("logs/%s_%s_%s.log" % (model_name, data_name.split('/')[-1][:20], data_set))
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

    logging.info(model_name)
    logging.info(data_name)
    if not data_name_is_filename:
        (val_data, valid_left_data, valid_right_data,
         valid_left_edges, valid_right_edges,
         valid_labels, valid_dists) = load_hdf5_data("%s_%s.hdf5" % (data_name, data_set))
    else:
        (val_data, valid_left_data, valid_right_data,
         valid_left_edges, valid_right_edges,
         valid_labels, valid_dists) = load_hdf5_data(data_name)

    if legacy:
        loss_fn = torch.nn.CrossEntropyLoss()
    else:
        loss_fn = torch.nn.BCEWithLogitsLoss()
    model.load_state_dict(torch.load('%s/%s.model.pt' % (model_dir,model_name)))
    classifier.load_state_dict(torch.load("%s/%s.classifier.pt" % (model_dir,model_name)))

    model.cuda()
    classifier.cuda()
    ret = predict(model, classifier, loss_fn,
                  valid_left_data, valid_left_edges,
                  valid_right_data, valid_right_edges,
                  valid_dists, valid_labels, return_prob=True,
                  use_distance=use_distance, use_metrics=use_metrics,
                  max_size=max_size, verbose=verbose, same=same, legacy=legacy, plot=plot)

    if return_probs and use_metrics:
        val_probs = ret[1]
        val_probs_by_dist = [[] for _ in range(num_bins)]
        val_labels_by_dist = [[] for _ in range(num_bins)]
        val_preds_by_dist = [[] for _ in range(num_bins)]
        counts_by_dist = np.zeros(num_bins)
        for p, l, d in zip(val_probs, valid_labels, valid_dists):
            idx = int(d[0] *num_bins - 0.000001)
            val_probs_by_dist[idx].append(p)
            val_labels_by_dist[idx].append(l)
            val_preds_by_dist[idx].append(1 if p >= 0.5 else 0)
            counts_by_dist[idx] += 1
        auprcs_by_dist = []
        f1s_by_dist = []
        for ps, ls, prs in zip(val_probs_by_dist, val_labels_by_dist, val_preds_by_dist):
            if len(ls) > 1 and sum(ls) / len(ls) <= 0.5:
                auprcs_by_dist.append(metrics.average_precision_score(ls, ps))
                f1s_by_dist.append(metrics.f1_score(ls, prs))
            else:
                auprcs_by_dist.append(0)
                f1s_by_dist.append(0)
        print(auprcs_by_dist)
        print(f1s_by_dist)

        pos_by_dist = [np.sum(x) for x in val_labels_by_dist]
        pos_frac_by_dist = [a / b for a, b in zip(pos_by_dist, counts_by_dist)]

        pos_by_dist = np.array(pos_by_dist) / np.sum(pos_by_dist)
        counts_by_dist /= np.sum(counts_by_dist)

        if plot:
            plt.plot(auprcs_by_dist, label='aupr')
            plt.plot(f1s_by_dist, label='f1')
            plt.plot(counts_by_dist, label='total count')
            plt.plot(pos_by_dist, label='pos count')
            plt.plot(pos_frac_by_dist, label='pos frac')
            plt.legend()
            plt.show()

    if save_probs:
        print(val_data['pairs'].shape, ret[1].shape)
        print('%s_%s_%s_probs.txt' % (model_name, data_name.split('/')[-1], data_set))
        with open('%s_%s_%s_probs.txt' % (model_name, data_name.split('/')[-1], data_set), 'w') as out:
            out.write('\n'.join(['\t'.join(map(str, list(pair)[:6]+[label, prob] + list(pair)[7:]))
                                 for pair, label, prob in zip(val_data['pairs'][:], valid_labels, ret[1])]))


    val_data.close()
    fileHandler.close()
    consoleHandler.close()
    rootLogger.removeHandler(fileHandler)
    rootLogger.removeHandler(consoleHandler)

    if return_probs:
        return ret[1]
