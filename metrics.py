from scipy.stats import kendalltau
from sklearn.metrics import r2_score


def get_metric_by_epoch(pred_list: list, label_list: list, idx: int, metric):
    final_pred_list = []
    final_label_list = []

    for p, l in zip(pred_list, label_list):
        final_pred_list.append(p[idx])
        final_label_list.append(l[idx])

    return metric(final_label_list, final_pred_list)


def get_final_epoch_kt(pred_list: list, label_list: list):
    return get_metric_by_epoch(pred_list, label_list, -1, kendalltau)


def get_final_epoch_r2(pred_list: list, label_list: list):
    return get_metric_by_epoch(pred_list, label_list, -1, r2_score)


def get_avg_kt(pred_list: list, label_list: list):
    kt_list = []
    p_list = []

    for i in range(pred_list[0].shape[0]):
        kt, p = get_metric_by_epoch(pred_list, label_list, i, kendalltau)
        kt_list.append(kt)
        p_list.append(p)

    return sum(kt_list)/len(kt_list), sum(p_list)/len(p_list)


def get_avg_r2(pred_list: list, label_list: list):
    r_list = []

    for i in range(pred_list[0].shape[0]):
        r = get_metric_by_epoch(pred_list, label_list, i, r2_score)
        r_list.append(r)

    return sum(r_list)/len(r_list)
