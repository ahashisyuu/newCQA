import os
import time
import numpy as np

from OfficialScorer import metrics


def PRF(label: np.ndarray, predict: np.ndarray):
    categories_num = label.max() + 1

    matrix = np.zeros((categories_num, categories_num), dtype=np.int32)

    label_array = [(label == i).astype(np.int32) for i in range(categories_num)]
    predict_array = [(predict == i).astype(np.int32) for i in range(categories_num)]

    for i in range(categories_num):
        for j in range(categories_num):
            matrix[i, j] = label_array[i][predict_array[j] == 1].sum()

    # (1) confusion matrix
    label_sum = matrix.sum(axis=1, keepdims=True)  # shape: (ca_num, 1)
    matrix = np.concatenate([matrix, label_sum], axis=1)  # or: matrix = np.c_[matrix, label_sum]
    predict_sum = matrix.sum(axis=0, keepdims=True)  # shape: (1, ca_num+1)
    matrix = np.concatenate([matrix, predict_sum], axis=0)  # or: matrix = np.r_[matrix, predict_sum]

    # (2) accuracy
    temp = 0
    for i in range(categories_num):
        temp += matrix[i, i]
    accuracy = temp / matrix[categories_num, categories_num]

    # (3) precision (P), recall (R), and F1-score for each label
    P = np.zeros((categories_num,))
    R = np.zeros((categories_num,))
    F = np.zeros((categories_num,))

    for i in range(categories_num):
        P[i] = matrix[i, i] / matrix[categories_num, i]
        R[i] = matrix[i, i] / matrix[i, categories_num]
        F[i] = 2 * P[i] * R[i] / (P[i] + R[i]) if P[i] + R[i] > 0 else 0

    # # (4) micro-averaged P, R, F1
    # micro_P = micro_R = micro_F = accuracy

    # (5) macro-averaged P, R, F1
    macro_P = P.mean()
    macro_R = R.mean()
    macro_F = 2 * macro_P * macro_R / (macro_P + macro_R) if macro_P + macro_R else 0

    return {'matrix': matrix, 'acc': accuracy,
            'each_prf': [P, R, F], 'macro_prf': [macro_P, macro_R, macro_F]}


def print_metrics(metrics, metrics_type, save_dir=None):
    matrix = metrics['matrix']
    acc = metrics['acc']
    each_prf = [[v * 100 for v in prf] for prf in zip(*metrics['each_prf'])]
    macro_prf = [v * 100 for v in metrics['macro_prf']]
    loss = metrics['loss']
    epoch = metrics['epoch']
    lines = ['\n\n**********************************************************************************',
             '*                                                                                *',
             '*                           {}                                  *'.format(
                 time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))),
             '*                                                                                *',
             '**********************************************************************************\n',
             '------------  Epoch {0}, loss {1:.4f}  -----------'.format(epoch, loss),
             'Confusion matrix:',
             '{0:>6}|{1:>6}|{2:>6}|<-- classified as'.format(' ', 'Good', 'Bad'),
             '------|--------------------|{0:>6}'.format('-SUM-'),
             '{0:>6}|{1:>6}|{2:>6}|{3:>6}'.format('Good', *matrix[0].tolist()),
             '{0:>6}|{1:>6}|{2:>6}|{3:>6}'.format('Bad', *matrix[1].tolist()),
             '------|-------------|------',
             '{0:>6}|{1:>6}|{2:>6}|{3:>6}'.format('-SUM-', *matrix[2].tolist()),
             '\nAccuracy = {0:6.2f}%\n'.format(acc * 100),
             'Results for the individual labels:',
             '\t{0:>6}: P ={1:>6.2f}%, R ={2:>6.2f}%, F ={3:>6.2f}%'.format('Good', *each_prf[0]),
             '\t{0:>6}: P ={1:>6.2f}%, R ={2:>6.2f}%, F ={3:>6.2f}%'.format('Bad', *each_prf[1]),
             '\n<<Official Score>>Macro-averaged result:',
             'P ={0:>6.2f}%, R ={1:>6.2f}%, F ={2:>6.2f}%'.format(*macro_prf),
             '--------------------------------------------------\n']

    [print(line) for line in lines]

    if save_dir is not None:
        with open(os.path.join(save_dir, "{}_logs.log".format(metrics_type)), 'a') as fw:
            [fw.write(line + '\n') for line in lines]


def transferring(matrix: np.ndarray, categories_num=3):
    conf_matrix = {'true': {'true': {}, 'false': {}}, 'false': {'true': {}, 'false': {}}}
    if categories_num == 3:
        conf_matrix['true']['true'] = matrix[0, 0]
        conf_matrix['true']['false'] = matrix[0, 1] + matrix[0, 2]
        conf_matrix['false']['true'] = matrix[1, 0] + matrix[2, 0]
        conf_matrix['false']['false'] = matrix[1, 1] + matrix[1, 2] + matrix[2, 1] + matrix[2, 2]
    else:
        conf_matrix['true']['true'] = matrix[0, 0]
        conf_matrix['true']['false'] = matrix[0, 1]
        conf_matrix['false']['true'] = matrix[1, 0]
        conf_matrix['false']['false'] = matrix[1, 1]
    return conf_matrix


def get_pre(eval_id, label, score, reranking_th, ignore_noanswer):

    model_pre = {}
    for cID, relevant, s in zip(eval_id, label, score):
        relevant = 'true' if 0 == relevant else 'false'
        # Process the line from the res file.
        qid = '_'.join(cID.split('_')[0:-1])
        if qid not in model_pre:
            model_pre[qid] = []
        model_pre[qid].append((relevant, s, cID))

    # Remove questions that contain no correct answer
    if ignore_noanswer:
        for qid in model_pre.keys():
            candidates = model_pre[qid]
            if all(relevant == "false" for relevant, _, _ in candidates):
                del model_pre[qid]

    for qid in model_pre:
        # Sort by model prediction score.
        pre_sorted = model_pre[qid]
        max_score = max([score for rel, score, cid in pre_sorted])
        if max_score >= reranking_th:
            pre_sorted = sorted(pre_sorted, key=lambda x: x[1], reverse=True)

        model_pre[qid] = [rel for rel, score, aid in pre_sorted]

    return model_pre


def eval_reranker(eval_id, label, score,
                  th=10,
                  reranking_th=0.0,
                  ignore_noanswer=False):
    model_pre = get_pre(eval_id, label, score,
                        reranking_th=reranking_th, ignore_noanswer=ignore_noanswer)

    # evaluate SVM
    # prec_model = metrics.recall_of_1(model_pre, th)
    # acc_model = metrics.accuracy(model_pre, th)
    # acc_model1 = metrics.accuracy1(model_pre, th)
    # acc_model2 = metrics.accuracy2(model_pre, th)
    mrr_model = metrics.mrr(model_pre, th)
    map_model = metrics.map(model_pre, th)

    avg_acc1_model = metrics.avg_acc1(model_pre, th)

    print("")
    print("*** Official score (MAP for SYS): %5.4f" % map_model)
    print("%13s" % "SYS")
    print("MAP   : %5.4f" % map_model)
    print("AvgRec: %5.4f" % avg_acc1_model)
    print("MRR   : %6.2f" % mrr_model)

    # for i, (p_model, a_model, a_model1, a_model2) in enumerate(
    #         zip(prec_model, acc_model, acc_model1, acc_model2), 1):
    #     print("REC-1@%02d: %6.2f  ACC@%02d: %6.2f  AC1@%02d: %6.2f  AC2@%02d: %4.0f" % (
    #           i, p_model, i, a_model, i, a_model1, i, a_model2))
    print('----------------------------------------------------------------')
    print('----------------------------------------------------------------\n')
    return map_model, avg_acc1_model, mrr_model

