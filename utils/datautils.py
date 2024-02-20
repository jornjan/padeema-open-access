import pandas as pd
import numpy as np

import sklearn.metrics as skmetrics
from typing import List, Tuple, Dict


def get_positive_data(input: pd.DataFrame, target_col='infection'):
    any_infec = input.groupby(level=0)[target_col].any()
    pos_any_infec = any_infec[any_infec].index.tolist()
    return input.loc[pos_any_infec].copy()


def get_negative_data(input: pd.DataFrame, target_col='infection'):
    any_infec = input.groupby(level=0)[target_col].any()
    pos_any_infec = any_infec[~any_infec].index.tolist()
    return input.loc[pos_any_infec].copy()


def subsample_healthy_sequences(input: pd.DataFrame, target_col: str, shuffle=True) -> pd.DataFrame:
    def compute_new_los(healthy_los: pd.DataFrame, pos_probs: pd.Series) -> pd.Series:
        healthy_los = healthy_los.copy()
        healthy_los['new_los'] = healthy_los['los']
        
        max_day = 16
        min_day = pos_probs.index.min()
        pos_probs = pos_probs.reindex(range(min_day, max_day+1 ), fill_value=0)
        
        n_per_day = (pos_probs*len(healthy_los)).round()
        for day in range(max_day,min_day,-1):
            temp_upns = healthy_los[healthy_los['new_los']==day].index.values.tolist()
            n_downsample = int(len(temp_upns) - n_per_day.loc[day])
            
            if n_downsample <= 0: continue
            
            downssample_upns = np.random.choice(temp_upns, n_downsample, replace=False)
            healthy_los.loc[downssample_upns, 'new_los'] = healthy_los.loc[downssample_upns, 'new_los'] - 1

        return healthy_los
    
    def subsample_sequences(input: pd.DataFrame, new_seq_lengths: pd.Series):
        cid = input.index.get_level_values(0)[0]
        sel_day = new_seq_lengths.loc[cid]
        return input.iloc[input.index.get_level_values(1) <= sel_day]
        
    positive_data = get_positive_data(input, target_col)
    negative_data = get_negative_data(input, target_col)

    # Compute distribution of length of stay (los) in positive cases
    positive_los_counts = positive_data.groupby(level=0).size().value_counts().sort_index()
    positive_los_probs = positive_los_counts/positive_los_counts.sum()
    
    los_neg_df = pd.DataFrame(negative_data.groupby(level=0).size(), columns=['los'])
    los_neg_df = compute_new_los(los_neg_df, positive_los_probs)
    
    # Shorten sequences
    negative_data_out = negative_data.groupby(level=0).apply(subsample_sequences, los_neg_df['new_los'])
    negative_data_out = negative_data_out.droplevel(0)
    
    df_out = pd.concat([negative_data_out, positive_data])
    # [Optionally] shuffle the patients
    if shuffle:
        upn_list = df_out.index.get_level_values(0).unique().tolist()
        np.random.shuffle(upn_list)
        df_out = df_out.loc[upn_list]

    return df_out


def compl_wide_to_long(input: pd.DataFrame, target: str, max_los=15) -> pd.DataFrame:
    assert target in input.columns.tolist(), 'Target variable not found in complication dataframe'
    assert f'{target}_day' in input.columns.tolist(), 'Target day not found in complication dataframe'
    
    target_day = input[f'{target}_day'].values[0]
    los = input['los'].values[0]
    final_day = np.nanmin([max_los, los, target_day])
    if pd.isna(target_day) or final_day<target_day:
        final_label = 0
    else:
        final_label = 1
    
    day_index = pd.Index(range(1, int(final_day)+1), name='day')
    df_out = pd.DataFrame(index=day_index)
    df_out[target] = 0
    df_out.loc[final_day, target] = final_label

    return df_out

def build_run_compl_df(wide_compl_df: pd.DataFrame,
                       target: str='infection',
                       min_los: int=1,
                       shift: int=-1):
    long_compl_df = wide_compl_df.groupby('CID').apply(compl_wide_to_long, target=target)
    
    # Min length of stay
    seq_len_df = long_compl_df.groupby('CID').size()
    sel_cids = seq_len_df[seq_len_df>=min_los].index.tolist()
    target_df = long_compl_df.loc[sel_cids].copy()
    
    # Subsampling
    target_df = subsample_healthy_sequences(target_df, target)
    target_df = target_df.groupby('CID').shift(periods=shift).dropna()
    target_df = target_df.astype(int)
    
    return target_df




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



class DailyMedianImputer():
    """
    Uses the median value for per day to fill missing values.
    """    
    
    def __init__(self, fill_vals: pd.DataFrame=None) -> None:
        """Initialize the imputer, optionally with preset imputation values.

        Parameters
        ----------
        fill_vals : pd.DataFrame, optional
            Set the values to use for imputation.
        """        
        self.fill_vals: pd.DataFrame = fill_vals
    
    def fit(self, input: pd.DataFrame) -> None:
        """Fits the imputation values to the input. This overwrites the previous values.

        Parameters
        ----------
        input : pd.DataFrame
            Dataframe to fit. Must have a column/index named `day`.
        """        
        self.fill_vals = input.groupby('day').median()
        self.fill_vals = self.fill_vals.fillna(method='ffill', axis=1) # Ensure there are no NaN values left
        
    def transform(self, input: pd.DataFrame) -> pd.DataFrame:
        """Impute the missing values according to the fitted imputation values.

        Parameters
        ----------
        input : pd.DataFrame
            Dataframe to transform

        Returns
        -------
        pd.DataFrame
            Dataframe with values imputed
        """
        assert self.fill_vals is not None, "Imputer must be fitted before transformation"        
        output = input.fillna(self.fill_vals)
        return output.fillna(method='ffill', axis=0)
    
    def fit_transform(self, input: pd.DataFrame) -> pd.DataFrame:
        """ First fit the model to the input and then transform it.

        Parameters
        ----------
        input : pd.DataFrame
            Dataframe to fit and transform

        Returns
        -------
        pd.DataFrame
            Dataframe with values imputed
        """        
        self.fit(input)
        return self.transform(input)