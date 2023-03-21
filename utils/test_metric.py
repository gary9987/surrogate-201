import numpy as np
import random


def mAP(predict, label, n: float):
    assert 0. <= n <= 1.
    topn = int(n * len(predict))
    sorted_pred = sorted(predict, reverse=True)
    pred_order = [sorted_pred.index(i) for i in predict]

    sorted_label = sorted(label, reverse=True)
    label_order = [sorted_label.index(i) for i in label]

    pred_binary = []
    for i in range(len(predict)):
        if pred_order[i] < topn and label_order[i] < topn:
            pred_binary.append(1)
        else:
            pred_binary.append(0)

    argsort_pred = (-1 * np.array(predict)).argsort()
    sorted_pred_binary = [pred_binary[i] for i in argsort_pred]
    sorted_pred_binary = sorted_pred_binary[: topn]

    precision_sum = 0.
    correct_cot = 1
    for i in range(topn):
        if sorted_pred_binary[i] == 1:
            precision_sum += (correct_cot / (i+1))
            correct_cot += 1

    return precision_sum / (correct_cot - 1) if correct_cot != 1 else 0.0


def randon_select_data(predict, label, mid_point: int, num_select: int, num_minor: int, minor_bound=None):
    assert num_select > num_minor

    num_major = num_select - num_minor

    pred_list = []
    label_list = []

    for select_type, num in zip(['major', 'minor'], [num_major, num_minor]):
        for _ in range(num):
            rand_idx = random.randint(0, label.shape[0] - 1)
            if select_type == 'major':
                while label[rand_idx] <= mid_point:
                    rand_idx = random.randint(0, label.shape[0] - 1)
            elif select_type == 'minor':
                bound = mid_point if minor_bound is None else minor_bound
                while label[rand_idx] > bound:
                    rand_idx = random.randint(0, label.shape[0] - 1)

            pred_list.append(predict[rand_idx])
            label_list.append(label[rand_idx])

    return pred_list, label_list


def is_misjudgment(pred_list, label_list, mid_point: int, num_select: int, num_judge: int):
    assert num_judge <= num_select
    sorted_idx = sorted(range(len(pred_list)), key=lambda k: pred_list[k])

    # Return True when the minor data appears in the top num_judge data
    for idx in sorted_idx[num_select-num_judge:]:
        if label_list[idx] <= mid_point:
            return True
