"""Metrics to evaluate ICD-9 coding models.
"""

import pandas as pd

def basic_micro_metrics(true_labels, predictions, delimiter=';'):
    '''Computes basic micro metrics (those not requiring scores).

    Micro metrics are averaged across all predictions and labels.

    Arguments
    ---------
    true_labels : interable
        Contains sets of true labels as strings or container that can be cast to
        a set.
    predictions : iterable
        Contains sets of predictions as strings or container that can be cast to
        a set.
    delimiter : char, optional
        A delimiter to use if the labels are formatted as strings.

    Returns
    -------
    metrics : dict
        A dictionary containing micro-averaged precision, recall, and f1 score.
    '''

    true_pos = 0
    false_pos = 0
    false_neg = 0

    true_labels = pd.Series(true_labels)
    predictions = pd.Series(predictions)

    # Check if contents are Strings and split if needed
    if isinstance(true_labels[0], str):
        true_labels = true_labels.fillna('')
        true_labels = true_labels.str.split(';')
    if isinstance(predictions[0], str):
        true_labels = predictions.fillna('')
        true_labels = predictions.str.split(';')

    true_labels = true_labels.apply(set)
    predictions = predictions.apply(set)

    # Get counts to compute metrics
    for true_set, predict_set in zip(true_labels, predictions):
        true_pos += len(true_set.intersection(predict_set))
        false_pos += len(predict_set - true_set)
        false_neg += len(true_set - predict_set)

    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    f1 = 2 * (precision * recall) / (precision + recall)

    metrics = {'micro-precision':precision, 'micro-recall':recall, 'micro-f1':f1}

    return metrics

def basic_macro_metrics(true_labels, predictions, delimiter=';', per_label=False):
    '''Computes basic micro metrics (those not requiring scores).

    Micro metrics are averaged across all predictions and labels.

    Arguments
    ---------
    true_labels : interable
        Contains sets of true labels as strings or container that can be cast to
        a set.
    predictions : iterable
        Contains sets of predictions as strings or container that can be cast to
        a set.
    delimiter : char, optional
        A delimiter to use if the labels are formatted as strings.
    per_label : boolean, optional
        Whether to return the metrics for each individual label.

    Returns
    -------
    metrics : dict
        A dictionary containing macro-averaged precision, recall, and f1 score.

    per_label_metrics : dict
        A dictionary containing per-label metrics, only if specified by per_label.
    '''
    metrics = {}
    per_label_metrics = {}

    code_idx = {}
    codes = []
    true_pos = []
    false_pos = []
    false_neg = []

    true_labels = pd.Series(true_labels)
    predictions = pd.Series(predictions)

    # Check if contents are Strings and split if needed
    if isinstance(true_labels[0], str):
        true_labels = true_labels.fillna('')
        true_labels = true_labels.str.split(';')
    if isinstance(predictions[0], str):
        true_labels = predictions.fillna('')
        true_labels = predictions.str.split(';')

    true_labels = true_labels.apply(set)
    predictions = predictions.apply(set)

    # Helper to get or assign code to array index
    def get_index(label):
        if label in code_idx:
            temp_idx = code_idx[label]
        else:
            temp_idx = len(codes)
            code_idx[label] = temp_idx
            codes.append(label)
            true_pos.append(0)
            false_pos.append(0)
            false_neg.append(0)
        return temp_idx

    # Get counts to compute metrics
    for true_set, predict_set in zip(true_labels, predictions):
        for label in true_set:
            # Get array index
            temp_idx = get_index(label)
            # Check for true positives and false negatives
            if label in predict_set:
                true_pos[temp_idx] += 1
            else:
                false_neg[temp_idx] += 1
        for label in predict_set:
            # Get arratrue_labelsy index
            temp_idx = get_index(label)
            # Check for false positives
            if label not in true_set:
                false_pos[temp_idx] += 1

    # Create dataframe for computation of results
    df = pd.DataFrame({'codes':codes,
                       'tp':true_pos,
                       'fp':false_pos,
                       'fn':false_neg})

    # Compute macro_results using 1e-10 to avoid divide-by-zero
    df['precision'] = df['tp'] / (df['tp'] + df['fp'] + 1e-10)
    df['recall'] = df['tp'] / (df['tp'] + df['fn'] + 1e-10)
    df['f1'] = 2 * df['precision'] * df['recall'] / (df['precision'] + df['recall'] + 1e-10)
    df['support'] = df['tp'] + df['fn']

    metrics = {'macro-precision':df['precision'].mean(),
               'macro-recall':df['recall'].mean(),
               'macro-f1':df['recall'].mean()}

    if per_label:
        results = (metrics, df.to_dict(orient='list'))
    else:
        results = metrics

    return results
