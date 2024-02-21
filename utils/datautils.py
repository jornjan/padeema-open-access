import pandas as pd
import numpy as np

from utils import imputers

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


def build_datadict(data: Dict[str, pd.DataFrame],
                   target: str,
                   min_los: int,
                   shift: int) -> Dict[str, pd.DataFrame]:
    data = {key: df.copy() for key, df in data.items()}
    datadict = dict()
    
    target_df = build_run_compl_df(data['compl'], target=target, min_los=min_los, shift=shift)
    datadict['target'] = target_df
    
    # lab
    lab_df = data.get('lab', None)
    if lab_df is not None:
        lab_df = target_df.join(lab_df, how='left')
        lab_df = lab_df.groupby(level=0).fillna(method='ffill')
        datadict['lab'] = lab_df
    
    # vitals 
    vitals_df = data.get('vitals', None)
    if vitals_df is not None:
        vitals_df = target_df.join(vitals_df, how='left')
        vitals_df = vitals_df.groupby(level=0).fillna(method='ffill')
        datadict['vitals'] = vitals_df
        
    # image
    image_df = data.get('img', None)
    if image_df is not None:
        image_df = target_df.join(image_df, how='left')
        image_df = image_df.groupby(level=0).fillna(method='ffill')
           
        datadict['img'] = image_df
        
    # static
    static_df = data.get('static', None)
    if static_df is not None:
        datadict['static'] = target_df.groupby(level=0).last().join(static_df, how='left')
        
    return datadict

def impute_data(train_data: Dict[str, pd.DataFrame],
                test_data: Dict[str, pd.DataFrame],
                target: str) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    keys = list(train_data.keys())
    
    if 'lab' in keys:
        imputer = imputers.DailyMedianImputer()
        lab_train, lab_test = train_data['lab'], test_data['lab']
        
        labcols = lab_train.columns.drop(target).tolist()
        lab_train[labcols] = imputer.fit_transform(lab_train[labcols])
        lab_test[labcols] = imputer.transform(lab_test[labcols])
        
        train_data['lab'] = lab_train
        test_data['lab'] = lab_test
        
        
    if 'vitals' in keys:
        imputer = imputers.DailyMedianImputer()
        vitals_train, vitals_test = train_data['vitals'], test_data['vitals']
        
        vitcols = vitals_train.columns.drop(target).tolist()
        vitals_train[vitcols] = imputer.fit_transform(vitals_train[vitcols])
        vitals_test[vitcols] = imputer.transform(vitals_test[vitcols])
        
        train_data['vitals'] = vitals_train
        test_data['vitals'] = vitals_test
        

    if 'static' in keys:
        imputer = imputers.MedianImputer()
        static_train, static_test = train_data['static'], test_data['static']
        
        try:
            statcols = static_train.columns.drop(target).tolist()
            static_train[statcols] = imputer.fit_transform(static_train[statcols])
            static_test[statcols] = imputer.transform(static_test[statcols])
        except:
            print(static_train.dtypes)
            print(imputer.fill_vals)
            raise(ValueError('failed imputation'))
        
        train_data['static'] = static_train
        test_data['static'] = static_test
        
    return train_data, test_data