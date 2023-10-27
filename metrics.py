import torch
import numpy as np
from sklearn import metrics
from torch_geometric.utils import negative_sampling


def sigmoid_array(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x


def softmax_array(x, y):
    newx = []
    newy = []
    for i in range(len(x)):
        e1 = x[i]
        e2 = y[i]
        mx_e = max(e1, e2)
        e1 = e1 - mx_e
        e2 = e2 - mx_e
        new_e1 = np.exp(e1) / (np.exp(e1) + np.exp(e2))
        new_e2 = np.exp(e2) / (np.exp(e1) + np.exp(e2))
        newx.append(new_e1)
        newy.append(new_e2)
    return np.array(newx), np.array(newy)


def eval_threshold(labels_all, preds_all):
    for i in range(int(0.5 * len(labels_all))):
        if preds_all[2 * i] > 0.95 and preds_all[2 * i + 1] > 0.95:
            preds_all[2 * i] = max(preds_all[2 * i], preds_all[2 * i + 1])
            preds_all[2 * i + 1] = preds_all[2 * i]
        else:
            preds_all[2 * i] = min(preds_all[2 * i], preds_all[2 * i + 1])
            preds_all[2 * i + 1] = preds_all[2 * i]
    fpr, tpr, thresholds = metrics.roc_curve(labels_all, preds_all)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    preds_all_ = []
    for p in preds_all:
        if p >= optimal_threshold:
            preds_all_.append(1)
        else:
            preds_all_.append(0)
    return preds_all, preds_all_


def get_roc_score(model, output, pos_edges, neg_edges):
    model.eval()
    if pos_edges.nelement() == 0:
        return 0, 0, 0, 0, 0, 0

    with torch.no_grad():
        pre_loss, positive_penalty, pos_pred, neg_pred = model(
            [output, pos_edges, neg_edges]
        )
        #
        pos_pred = torch.Tensor.numpy(
            torch.squeeze(torch.Tensor.cpu(pos_pred.detach()))
        )
        neg_pred = torch.Tensor.numpy(
            torch.squeeze(torch.Tensor.cpu(neg_pred.detach()))
        )

        pos_pred_soft, neg_pred_soft = softmax_array(pos_pred, neg_pred)

        pred_soft = np.hstack([pos_pred_soft, neg_pred_soft])
        # pred_soft = np.hstack([pos_pred, neg_pred])
        pred_label = np.hstack([np.ones_like(pos_pred), np.zeros_like(neg_pred)])

        _, preds_all_ = eval_threshold(pred_label, pred_soft)
        auc = metrics.roc_auc_score(pred_label, pred_soft)
        ap = metrics.average_precision_score(pred_label, pred_soft)
        acc = metrics.accuracy_score(pred_label, preds_all_)
        f1_score = metrics.f1_score(pred_label, preds_all_)
        recall = metrics.recall_score(pred_label, preds_all_)

    return auc, ap, acc, f1_score, recall, pre_loss
