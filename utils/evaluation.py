import pandas as pd
import numpy as np
import sklearn.metrics as skmetrics

from typing import List


def compute_metrics(y_true_list: List, y_pred_list: List, labels=['train', 'test']) -> pd.DataFrame:
    results = []
    for y_true, y_pred, split in zip(y_true_list, y_pred_list, labels):
        fpr, tpr, thresholds = skmetrics.roc_curve(y_true, y_pred)
        best_thresh = thresholds[np.argmax(tpr-fpr)]
        
        y_pred_bin = [1 if x > best_thresh else 0 for x in y_pred]
        
        results.append({
            'split': split,
            'auc': skmetrics.roc_auc_score(y_true, y_pred),
            'pr': skmetrics.average_precision_score(y_true, y_pred),
            'accuracy': skmetrics.accuracy_score(y_true, y_pred_bin),
            'precision': skmetrics.precision_score(y_true, y_pred_bin, pos_label=1, zero_division=0),
            'recall': skmetrics.recall_score(y_true, y_pred_bin, pos_label=1, zero_division=0),
            'specificity': skmetrics.recall_score(y_true, y_pred_bin, pos_label=0, zero_division=0),
            'f1-score': skmetrics.f1_score(y_true, y_pred_bin, zero_division=0)
        })
        
    return pd.DataFrame.from_records(results)