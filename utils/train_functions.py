import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler

from utils import datautils

from tqdm import tqdm
from typing import Dict, List, Tuple, Union



##########################################################################################
# Logistic Regression Train Function
##########################################################################################
def train_lr_model(compl_df: pd.DataFrame,
                   data_dfs: List[pd.DataFrame],
                   target: str,
                   min_los: int=0,
                   shift: int=-1,
                   lr_C: float=0.001,
                   n_runs: int = 10,
                   verbose=True
                   ) -> pd.DataFrame:
    """ Train multiple `LogisticRegression` models with different train-test splits.
    Can take multiple Dataframes as input, which will be concatenated for each sample.
    
    *Caution*
    Models are trained with all days for uncomplicated patients and only the day of complication for complicated patients. 
    This may lead to severe class imbalance

    Parameters
    ----------
    compl_df : pd.DataFrame
        Target complication DataFrame in wide format
    data_dfs : List[pd.DataFrame]
        Input dataframes, one for each modality, links with target DataFrame through CID
    target : str, optional
        Column name of target variable in `compl_df`
    min_los : int, optional
        Minimum number of days until complication or discharge, by default 0
    shift : int, optional
        Prediction window defined by relative shift, 0=0-days ahead, -1=1-day ahead, -2=2-days ahead, etc..., by default -1
    lr_C : float, optional
        Inverse regularization strength, by default 0.001
    n_runs : int, optional
        Number of cross-validation runs, total_runs=4*n_runs, by default 10
    verbose : bool, optional
        Whether or not to display a `tqdm` progressbar, by default True

    Returns
    -------
    pd.DataFrame
        Performance metrics for all trained models.
    """
    
    results = pd.DataFrame()

    skf = StratifiedKFold(n_splits=4, shuffle=True)
    if verbose: pbar = tqdm(total=skf.get_n_splits()*n_runs)
    for _ in range(n_runs):
        # Prepare data for this cross-validation run
        run_infec_df = datautils.build_run_compl_df(compl_df, target=target, min_los=min_los, shift=shift)
        neg_compl_df = datautils.get_negative_data(run_infec_df, target_col=target)
        pos_compl_df = datautils.get_positive_data(run_infec_df, target_col=target) \
            .reset_index(level=1) \
            .groupby(level=0).last() \
            .set_index('day', append=True)
            
            
        neg_run_data = neg_compl_df.copy()
        pos_run_data = pos_compl_df.copy()
        for input_data in data_dfs: neg_run_data = neg_run_data.join(input_data, how='left')
        for input_data in data_dfs: pos_run_data = pos_run_data.join(input_data, how='left')
        run_df = pd.concat([neg_run_data, pos_run_data]).sort_index()

        
        for train_idx, test_idx in skf.split(run_df, run_df[target]):
            train_df = run_df.iloc[train_idx].copy()
            test_df = run_df.iloc[test_idx].copy()
            
            # Prepare train and test data
            imputer = datautils.DailyMedianImputer() # Ensures no missing values
            train_df = imputer.fit_transform(train_df)
            test_df = imputer.transform(test_df)
            
            train_df = train_df.drop_duplicates()
            test_df = test_df.drop_duplicates()

            y_train = train_df.pop(target)
            y_test = test_df.pop(target)
            
            # Scale input values
            scaler = RobustScaler()
            X_train = scaler.fit_transform(train_df)
            X_test = scaler.transform(test_df)
            
            # Fit model
            clf = LogisticRegression(C=lr_C, class_weight='balanced')
            clf.fit(X_train, y_train)
            
            # Retrieve predictions
            pred_train = clf.predict_proba(X_train)[:, clf.classes_.tolist().index(1)]
            pred_test = clf.predict_proba(X_test)[:, clf.classes_.tolist().index(1)]
            
            # Compute results
            run_results = datautils.compute_metrics([y_train.to_numpy(), y_test.to_numpy()],
                                                     [pred_train, pred_test])
            
            run_results['n_pos'] = y_test.sum()
            run_results['n_neg'] = len(y_test) - y_test.sum()
            results = pd.concat([results, run_results], ignore_index=True)
            
            if verbose: pbar.update()
    if verbose: pbar.close()
    
    results['target'] = target
    results['window'] = f'{24*abs(shift)}h' 
    
    return results