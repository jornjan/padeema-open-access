import pandas as pd

from tensorflow import keras
from tensorflow.keras.optimizers import Adam
import tensorflow_addons as tfa

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler

from utils import datautils, custom_models, generators, imputers, evaluation

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
            imputer = imputers.DailyMedianImputer() # Ensures no missing values
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
            run_results = evaluation.compute_metrics([y_train.to_numpy(), y_test.to_numpy()],
                                                     [pred_train, pred_test])
            
            run_results['n_pos'] = y_test.sum()
            run_results['n_neg'] = len(y_test) - y_test.sum()
            results = pd.concat([results, run_results], ignore_index=True)
            
            if verbose: pbar.update()
    if verbose: pbar.close()
    
    results['target'] = target
    results['window'] = f'{24*abs(shift)}h' 
    
    return results


##########################################################################################
# Unimodal train functions
##########################################################################################

def train_resnet(train_df: pd.DataFrame,
                 test_df: pd.DataFrame,
                 target: str,
                 lr: float = 1e-4) -> keras.Model:
    img_dir = 'example_data/'
    
    def convert_to_tabular(input: pd.DataFrame, target: str) -> pd.DataFrame:
        neg_df = datautils.get_negative_data(input, target_col=target)
        neg_df = neg_df.reset_index(drop=True).dropna()
        
        pos_df = datautils.get_positive_data(input, target_col=target)
        pos_df = pos_df.groupby(level=0).fillna(method='ffill')
        pos_df = pos_df.groupby(level=0).last().reset_index(drop=True).dropna()
        
        return pd.concat([pos_df, neg_df], ignore_index=True)

    train_df = convert_to_tabular(train_df, target)
    test_df = convert_to_tabular(test_df, target)
    
    train_gen = generators.ImageGenerator(train_df, image_dir=img_dir, target_col=target, shuffle=True)
    test_gen = generators.ImageGenerator(test_df, image_dir=img_dir, target_col=target, shuffle=False)
            
    model = custom_models.build_resnet()
    model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr), metrics=[keras.metrics.AUC(name='auc')])
    model.fit(train_gen,
              epochs=25,
              validation_data=test_gen,
              verbose=0)
    
    return model


def train_image_model(train_df: pd.DataFrame,
                      test_df: pd.DataFrame,
                      lr: float,
                      target: str = 'infection') -> Tuple[pd.DataFrame, keras.Model]:
    img_dir = 'example_data/'
    
    resnet = train_resnet(train_df, test_df, target)
    model = custom_models.build_cnnlstm(resnet=resnet)
    model.get_layer('td_resnet').trainable = False
    
    train_gen = generators.SeqImageGenerator(train_df, image_dir=img_dir, target_col=target, shuffle=True)
    test_gen = generators.SeqImageGenerator(test_df, image_dir=img_dir, target_col=target, shuffle=False)

    model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr), metrics=[keras.metrics.AUC(name='auc')])
    model.fit(train_gen,
              epochs=25,
              validation_data=test_gen,
              verbose=0)
    
    results = evaluation.compute_metrics([train_gen.get_labels(), test_gen.get_labels()],
                                        [model.predict(train_gen, verbose=0),  model.predict(test_gen, verbose=0)])
    results.insert(0, 'model', 'img')
    return results, model
    

def train_lab_model(train_df: pd.DataFrame,
                    test_df: pd.DataFrame,
                    lr: float,
                    target: str = 'infection') -> Tuple[pd.DataFrame, keras.Model]:
    
    labcols = train_df.columns.drop(target).tolist()

    train_gen = generators.LabSeqGenerator(train_df, target, batch_size=32, shuffle=True)
    test_gen = generators.LabSeqGenerator(test_df, target, batch_size=32, shuffle=False)
    
    model = custom_models.LabLSTM(len(labcols), bd=False)
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr), metrics=[keras.metrics.AUC(name='auc')])

    model.fit(
        train_gen,
        epochs=15,
        validation_data=test_gen,
        verbose=0
    )

    results = evaluation.compute_metrics([train_gen.get_labels(), test_gen.get_labels()],
                                           [model.predict(train_gen, verbose=0),  model.predict(test_gen, verbose=0)])
    results.insert(0, 'model', 'lab')
    return results, model


def train_vitals_model(train_df: pd.DataFrame,
                       test_df: pd.DataFrame,
                       lr: float,
                       target: str = 'infection',
                       imputer = imputers.DailyMedianImputer()) -> Tuple[pd.DataFrame, keras.Model]:
    vitalcols = train_df.columns.drop(target).tolist()
    
    train_gen = generators.VitalsSeqGenerator(train_df, target, batch_size=32, shuffle=True)
    test_gen = generators.VitalsSeqGenerator(test_df, target, batch_size=32, shuffle=False)
    
    model = custom_models.LabLSTM(len(vitalcols), 12, bd=False)
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr), metrics=[keras.metrics.AUC(name='auc')])
    model.fit(
        train_gen,
        epochs=15,
        validation_data=test_gen,
        verbose=0
    )
    
    results = evaluation.compute_metrics([train_gen.get_labels(), test_gen.get_labels()],
                                        [model.predict(train_gen, verbose=0),  model.predict(test_gen, verbose=0)])
    results.insert(0, 'model', 'vitals')
    return results, model


def train_static_model(train_df: pd.DataFrame,
                       test_df: pd.DataFrame,
                       lr: float,
                       target: str = 'infection') -> Tuple[pd.DataFrame, keras.Model]:
    train_df, test_df = train_df.copy(), test_df.copy()
    static_cols = train_df.columns.drop(target).tolist()
    
    cweight_dict = generators.get_classweights(train_df[target])
    
    X_train, X_test = train_df[static_cols].astype(float).to_numpy(), test_df[static_cols].astype(float).to_numpy()
    y_train, y_test = train_df[target].astype(int).to_numpy(), test_df[target].astype(int).to_numpy()

    model = custom_models.StaticModel(n_features=len(static_cols))
    model.compile(optimizer=keras.optimizers.Adam(lr), loss='binary_crossentropy', metrics=[keras.metrics.AUC(name='auc')])
    model.fit(
        X_train, y_train,
        epochs=30,
        validation_data=(X_test, y_test),
        verbose=0,
        batch_size=32,
        class_weight=cweight_dict
        )
    
    results = evaluation.compute_metrics([y_train, y_test],
                                 [model.predict(X_train, verbose=0),  model.predict(X_test, verbose=0)])
    results.insert(0, 'model', 'static')
    return results, model


def train_unimodal_models(train_dict: Dict[str, pd.DataFrame],
                          test_dict: Dict[str, pd.DataFrame],
                          lr_dict: Dict[str, float],
                          target: str,
                          pbar=None) -> Tuple[pd.DataFrame, 
                                              Dict[str, keras.Model]]:
    model_train_func_mapper = {
        'img': train_image_model,
        'lab': train_lab_model,
        'vitals': train_vitals_model,
        'static': train_static_model
    }
    
    modalities = list(train_dict.keys())
    modalities.remove('target')
    
    results = pd.DataFrame()
    model_dict = dict()
    for modality in modalities:
        if pbar is not None: pbar.set_description(f'Training {modality} model')
        
        train_func = model_train_func_mapper[modality]
        temp_results, temp_model = train_func(train_dict[modality], test_dict[modality], lr_dict[modality], target)
        results = pd.concat([results, temp_results], ignore_index=True)
        model_dict[modality] = temp_model
        
    return results, model_dict


##########################################################################################
# Multimodal train functions
##########################################################################################

def copy_model(model: keras.Model) -> keras.Model:
    model_clone = keras.models.clone_model(model)
    model_clone.build(model.input_shape)
    model_clone.set_weights(model.get_weights())
    return model_clone


def get_multi_optimizer(mm_model: keras.Model,
                        model_dict: Dict[str, keras.Model], 
                        lr_dict: Dict[str, float],
                        factor: float) -> tfa.optimizers.MultiOptimizer:
    opt_list = list()
    
    # Set unimodal learning rates
    for name, um_model in model_dict.items():
        lr = lr_dict[name] * factor
        opt_list.append( (keras.optimizers.Adam(lr), um_model) )
    
    # Set multimodal learning rate
    for idx, layer in enumerate(mm_model.layers):
        if layer.name == 'fusion_layer':
            lr = lr_dict['final'] * factor
            opt_list.append( (keras.optimizers.Adam(lr), mm_model.layers[idx:]) )
    
    return tfa.optimizers.MultiOptimizer(opt_list)


def train_mmodal_model(train_dict: Dict[str, pd.DataFrame],
                       test_dict:  Dict[str, pd.DataFrame],
                       model_dict: Dict[str, keras.Model],
                       lr_dict: Dict[str, float],
                       fusion: str,
                       target: str,
                       ft_factor: int = None,
                       multi_loss: bool = False) -> Tuple[pd.DataFrame, keras.Model]:
    modalities = list(train_dict.keys())
    modalities.remove('target')
    
    train_gen = generators.MultimodalGenerator(train_dict, modalities, target, repeat_target=multi_loss, shuffle=True)
    test_gen = generators.MultimodalGenerator(test_dict, modalities, target, repeat_target=multi_loss, shuffle=False)
    
    model_dict = {key: copy_model(model) for key, model in model_dict.items()}
    full_model = custom_models.build_multimodal_model(model_dict, fusion=fusion, include_um_outputs=multi_loss)
    
    for name in model_dict.keys():
        full_model.get_layer(f'{name}_model').trainable = False
    
    loss = 'binary_crossentropy'
    
    full_model.compile(optimizer=Adam(lr_dict[fusion]), loss=loss, metrics=[keras.metrics.AUC(name='auc')])
    full_model.fit(train_gen,
                   epochs=30,
                   validation_data=test_gen,
                   verbose=0)

    train_labels, test_labels = train_gen.get_labels(), test_gen.get_labels()
    train_preds, test_preds = full_model.predict(train_gen, verbose=0),  full_model.predict(test_gen, verbose=0)
        
    results_df = evaluation.compute_metrics([train_labels, test_labels],
                                           [train_preds, test_preds])
    results_df.insert(0, 'model', fusion)
    
    # Optionally do finetuning
    if ft_factor is not None: 
        for name in model_dict.keys():
            full_model.get_layer(f'{name}_model').trainable = True

        lr_dict['final'] = lr_dict[fusion]
        optimizer = get_multi_optimizer(full_model, model_dict, lr_dict, ft_factor)
        
        full_model.compile(optimizer=optimizer, loss=loss, metrics=[keras.metrics.AUC(name='auc')])
        full_model.fit(train_gen,
                        epochs=10,
                        validation_data=test_gen,
                        verbose=0)
        
        train_labels, test_labels = train_gen.get_labels(), test_gen.get_labels()
        train_preds, test_preds = full_model.predict(train_gen, verbose=0),  full_model.predict(test_gen, verbose=0)
        if multi_loss:
            train_preds, test_preds = train_preds[:,0], test_preds[:,0]
            
        ft_results = evaluation.compute_metrics([train_labels, test_labels],
                                               [train_preds, test_preds])
        
        ft_results.insert(0, 'model', f'ft_{fusion}')
        results_df = pd.concat([results_df, ft_results], ignore_index=True)
        
    return results_df, full_model


def mmodal_training_run(raw_data_dict: Dict[str, pd.DataFrame],
                        lr_dict: Dict[str, float],
                        target: str,
                        min_los: int = 1,
                        shift: int = -1,
                        ft_factor: float = None,
                        multi_loss: bool = False,
                        break_early: bool = False,
                        verbose: bool = False) -> pd.DataFrame:
    # Create run data dict
    run_data_dict = datautils.build_datadict(raw_data_dict, target, min_los, shift)
    
    ## Cross val runs
    splitting_df = run_data_dict['target'].groupby(level=0).last()
    
    results = pd.DataFrame()
    skf = StratifiedKFold(n_splits=4, shuffle=True)
    
    pbar = tqdm(total=skf.get_n_splits()) if verbose else None
    for train_idx, test_idx in skf.split(splitting_df, splitting_df.to_numpy()):
        train_cid = splitting_df.iloc[train_idx].index.tolist()
        test_cid = splitting_df.iloc[test_idx].index.tolist()
        
        train_dict = {key: temp_df.loc[train_cid].copy() for key, temp_df in run_data_dict.items()}
        test_dict = {key: temp_df.loc[test_cid].copy() for key, temp_df in run_data_dict.items()}
        
        # Impute data
        train_dict, test_dict = datautils.impute_data(train_dict, test_dict, target)

        # Train unimodal models
        um_results, um_models = train_unimodal_models(train_dict, test_dict, lr_dict, target, pbar=pbar)
        
            
        # Train multimodal models
        if verbose: pbar.set_description(f'Training late model')
        late_results, late_model = train_mmodal_model(train_dict, test_dict, um_models, lr_dict, 'late', target, ft_factor=ft_factor, multi_loss=multi_loss)
        
        
        # pbar.set_description(f'Training mid model')
        mid_results, mid_model = train_mmodal_model(train_dict, test_dict, um_models, lr_dict, 'mid', target, ft_factor=ft_factor)
        
        results = pd.concat([results, um_results, late_results, mid_results], ignore_index=True)
        
        if verbose: pbar.update()
        if break_early: break
    if verbose: pbar.close()
    
    return results