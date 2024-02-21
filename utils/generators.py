import pandas as pd
import numpy as np
from typing import List, Dict, Callable

from tensorflow import keras
from sklearn.utils import compute_class_weight



#####################################################################################
# Utility Functions
#####################################################################################
def get_classweights(input: pd.Series):
    class_weights = compute_class_weight(class_weight='balanced', classes=[0, 1], y=input.to_numpy())
    cweight_dict = {
        0: class_weights[0],
        1: class_weights[1]
    }
    return cweight_dict




#####################################################################################
# Generators
#####################################################################################

class ImageGenerator(keras.utils.Sequence):
    def __init__(self,
                 df: pd.DataFrame,
                 image_dir: str,
                 target_col: str = 'infection',
                 batch_size: int = 32,
                 shuffle: bool = True) -> None:
        """ Generator to create image batches, from a `pd.DataFrame` containing at least two columns: `fname` and `<target_col>` 
        
        Parameters
        ----------
        df : pd.DataFrame
            `pd.DataFrame` containing at least two columns: `fname` and `<target_col>` 
        image_dir : str
            Path to directory that holds the images
        target_col : str, optional
            Target column for prediction, by default `infection`
        batch_size : int, optional, by default 32
        shuffle : bool, optional, 
            Shuffle data between epochs, by default True
        """  
        assert 'fname' in df.columns.tolist(), 'DataFrame must have a column called: fname'
        assert target_col in df.columns.tolist(), f'DataFrame must have the target column: {target_col}'
        
        self.df: pd.DataFrame = df.copy().sort_index()
        self.index_list: list = self.df.index.tolist()
        
        self.target_col: str = target_col
        self.image_dir: str = image_dir
        self.batch_size: int = batch_size
        self.shuffle: bool = shuffle
        
        self.df[self.target_col] = self.df[self.target_col].astype(int) # Make sure target col has right type
        
        self.cweight_dict = get_classweights(self.df[target_col])
        self.on_epoch_end()
    
    def __len__(self) -> int:      
        return int(np.ceil(len(self.index_list) / self.batch_size))

    def on_epoch_end(self):
        "Conditionally shuffles indexes every epoch"
        if self.shuffle:
            np.random.shuffle(self.index_list)
            
    def get_labels(self) -> np.ndarray:
        return self.df.loc[self.index_list, self.target_col].to_numpy()
    
    def get_fnames(self) -> np.ndarray:
        return self.df['fname'].to_numpy()
    
    def _load_img(self, fname: str):
        if pd.isna(fname):
            im_out = np.zeros((256, 256, 1))
        else:
            im_path = f'{self.image_dir}/{fname}'
            im_out = keras.preprocessing.image.load_img(im_path, color_mode='grayscale')
            im_out = keras.preprocessing.image.img_to_array(im_out)
        return im_out
    
    def __getitem__(self, index):
        idx_batch = self.index_list[index*self.batch_size:(index+1)*self.batch_size]
        
        fname_list = self.df.loc[idx_batch, 'fname'].tolist()
        X_batch = np.array([self._load_img(f) for f in fname_list])
        
        y_batch = self.df.loc[idx_batch, self.target_col].to_numpy()
        
        # Balance classes
        sweight_batch = np.fromiter((self.cweight_dict[y] for y in y_batch), dtype=float) 
        
        return X_batch, y_batch, sweight_batch
        
        
class SeqImageGenerator(keras.utils.Sequence):
    def __init__(self,
                 df: pd.DataFrame,
                 image_dir: str,
                 target_col: str = 'infection',
                 batch_size: int = 32,
                 shuffle: bool=True) -> None:
        """ Generator to create batches of image sequences of varying length. 
        Sequences are pre-padded to match the length of the largest sequence.

        Parameters
        ----------
        df : pd.DataFrame
            Multi-index dataframe with `CID` as the outer index and `day` as the inner index
        image_dir : str
            Path to directory that holds the images
        include_ct: bool
            Indicate whether we include CT image information
        target_col : str, optional
            Target column for prediction, by default `infection`
        batch_size : int, optional, by default 32
        shuffle : bool, optional, 
            Shuffle data between epochs, by default True
        """        
        assert 'fname' in df.columns.tolist(), 'fname column required in dataframe'

        self.df = df.copy().sort_index()
        self.df[target_col] = self.df[target_col].astype(int)  # Make sure target col has right type
        self.cid_list: list = self.df.index.get_level_values('CID').unique().tolist()

        self.batch_size: int = batch_size
        self.target_col: str = target_col
        self.image_dir: str = image_dir
        self.shuffle: bool = shuffle
        self.cweight_dict: Dict[int, float] = get_classweights(self.df.groupby(level=0).last()[target_col])
        
        self.on_epoch_end()
        
    def __len__(self) -> int:      
        return int(np.ceil(len(self.cid_list) / self.batch_size))

    def on_epoch_end(self):
        "Conditionally shuffles indexes every epoch"
        if self.shuffle:
            np.random.shuffle(self.cid_list)
            
    def get_cids(self) -> np.ndarray:
        return np.asarray(self.cid_list)
    
    def get_labels(self) -> np.ndarray:
        labels = [self.df.loc[cid, self.target_col].to_numpy() for cid in self.cid_list]        
        labels = [y[-1] for y in labels]
        return np.array(labels)
    
    def _load_img(self, fname: str):
        if pd.isna(fname):
            im_out = np.zeros((256, 256, 1))
        else:
            im_path = f'{self.image_dir}/{fname}'
            im_out = keras.preprocessing.image.load_img(im_path, color_mode='grayscale')
            im_out = keras.preprocessing.image.img_to_array(im_out)
        return im_out

    def __getitem__(self, index):
        cid_batch = self.cid_list[index*self.batch_size:(index+1)*self.batch_size]
        
        X_batch = list()
        for cid in cid_batch:
            fname_list = self.df.loc[cid, 'fname'].tolist()
            temp_ims = np.array([self._load_img(f) for f in fname_list])
            
            X_batch.append(temp_ims)
        
        X_batch = keras.preprocessing.sequence.pad_sequences(X_batch, padding='pre', dtype=float, value=0.)
        
        
        # Get target values
        y_batch = [self.df.loc[cid, self.target_col].to_numpy() for cid in cid_batch]        
        y_batch = [y[-1] for y in y_batch]
        y_batch = np.array(y_batch)
        
        # Balance classes
        sweight_batch = np.fromiter((self.cweight_dict[y] for y in y_batch), dtype=float) 
        
        return X_batch, y_batch, sweight_batch
        
    
class LabSeqGenerator(keras.utils.Sequence):
    def __init__(self, 
                 df: pd.DataFrame, 
                 target_col: str,
                 batch_size=32,
                 shuffle=True,
                 padding_val=0.) -> None:
        
        self.df: pd.DataFrame = df.copy().sort_index()
        self.df[target_col] = self.df[target_col].astype(int)  # Make sure target col has right type
        
        self.cid_list: list = self.df.index.get_level_values('CID').unique().tolist()
        
        self.target_col: str = target_col
        self.value_cols: List[str] = self.df.columns.drop(target_col).tolist()

        self.batch_size: int = len(self.cid_list) if batch_size is None else batch_size
        
        self.shuffle: bool = shuffle
        self.padding_val: float = padding_val
        self.cweight_dict: Dict[int, float] = get_classweights(self.df.groupby(level=0).last()[target_col])
        
        self.on_epoch_end()
    
    def __len__(self) -> int:
        return int(np.ceil(len(self.cid_list) / self.batch_size))

    def on_epoch_end(self):
        "Conditionally shuffles indexes every epoch"
        if self.shuffle:
            np.random.shuffle(self.cid_list)
            
    def get_cids(self) -> np.ndarray:
        return np.asarray(self.cid_list)
    
    def get_labels(self) -> np.ndarray:
        labels = [self.df.loc[cid, self.target_col].to_numpy() for cid in self.cid_list]        
        labels = [y[-1] for y in labels]
        return np.array(labels)
    
    def __getitem__(self, index):
        # Get the cids in this batch
        cid_batch = self.cid_list[index*self.batch_size:(index+1)*self.batch_size]

        # Get input values
        X_batch = [self.df.loc[cid, self.value_cols].to_numpy() for cid in cid_batch]
        X_batch = keras.preprocessing.sequence.pad_sequences(X_batch, padding='pre', dtype=float, value=self.padding_val)
        
        # Get target values
        y_batch = [self.df.loc[cid, self.target_col].to_numpy() for cid in cid_batch]        
        y_batch = [y[-1] for y in y_batch]
        y_batch = np.array(y_batch)
        
        # Balance classes
        sweight_batch = np.fromiter((self.cweight_dict[y] for y in y_batch), dtype=float) 
        
        return X_batch, y_batch, sweight_batch

    def to_dataframe(self) -> pd.DataFrame:
        X_full = [self.df.loc[cid].to_numpy() for cid in self.cid_list]
        X_full = keras.preprocessing.sequence.pad_sequences(X_full, maxlen=13, padding='pre', dtype=float, value=self.padding_val)
    
        
        m_index = pd.MultiIndex.from_product([self.cid_list, np.arange(1,X_full.shape[1]+1)], names=self.df.index.names)
        df_out = pd.DataFrame(index=m_index, data=X_full.reshape((X_full.shape[0]*X_full.shape[1], X_full.shape[2])), columns=self.df.columns)
        # df_out[self.target_col] = y_full
        return df_out
        
        
class VitalsSeqGenerator(keras.utils.Sequence):
    def __init__(self, 
                df: pd.DataFrame, 
                target_col: str,
                batch_size=32,
                shuffle=True,
                padding_val=0.) -> None:
        
        self.df: pd.DataFrame = df.copy().sort_index()
        self.df[target_col] = self.df[target_col].astype(int)  # Make sure target col has right type
        
        self.cid_list: list = self.df.index.get_level_values('CID').unique().tolist()
        
        self.target_col: str = target_col
        self.value_cols: List[str] = self.df.columns.tolist()
        self.value_cols.remove(self.target_col)

        self.batch_size: int = batch_size
        self.shuffle: bool = shuffle
        self.padding_val: float = padding_val
        self.cweight_dict: Dict[int, float] = get_classweights(self.df.groupby(level=0).last()[target_col])
        
        self.on_epoch_end()
    
    def __len__(self) -> int:
        "Number of batches per epoch"
        return int(np.ceil(len(self.cid_list) / self.batch_size))

    def on_epoch_end(self):
        "Conditionally shuffles indexes every epoch"
        if self.shuffle:
            np.random.shuffle(self.cid_list)
            
    def get_cids(self) -> np.ndarray:
        return np.asarray(self.cid_list)
    
    def get_labels(self) -> np.ndarray:
        labels = [self.df.loc[cid, self.target_col].to_numpy() for cid in self.cid_list]        
        labels = [y[-1] for y in labels]
        return np.array(labels)
    
    def __getitem__(self, index):
        # Get the cid's in this batch
        cid_batch = self.cid_list[index*self.batch_size:(index+1)*self.batch_size]

        # Get input values
        X_batch = [self.df.loc[cid, self.value_cols].to_numpy() for cid in cid_batch]
        X_batch = keras.preprocessing.sequence.pad_sequences(X_batch, padding='pre', dtype=float, value=self.padding_val)
        
        # Get target values
        y_batch = [self.df.loc[cid, self.target_col].to_numpy() for cid in cid_batch]        
        y_batch = [y[-1] for y in y_batch]
        y_batch = np.array(y_batch)
        
        # Balance classes
        sweight_batch = np.fromiter((self.cweight_dict[y] for y in y_batch), dtype=float) 
        
        return X_batch, y_batch, sweight_batch
    
    
class MultimodalGenerator(keras.utils.Sequence):
    def __init__(self,
                 data_dict: Dict[str, pd.DataFrame],
                 modalities: List[str],
                 target_col: str,
                 padding_val: float=0.0,
                 batch_size: int=32,
                 repeat_target: bool=False,
                 shuffle: bool=True) -> None:
        self.data_dict = data_dict
        
        self.target_df = self.data_dict['target'].groupby(level=0).last()[target_col].astype(int)
        self.cid_list = self.target_df.index.tolist()
        
        self.modalities = modalities
        self._getitem_mapper = self._init_getitem_mapper(self.modalities)
        
        self.image_dir = 'example_data/'
        self.padding_val = padding_val
        self.batch_size = batch_size
        self.repeat_target = repeat_target
        self.shuffle=shuffle
        self.cweight_dict: Dict[int, float] = get_classweights(self.target_df)
        
        if 'static' in modalities: self.static_cols = self.data_dict['static'].columns.drop(target_col).tolist()
        if 'lab' in modalities: self.lab_cols = self.data_dict['lab'].columns.drop(target_col).tolist()
        if 'vitals' in modalities: self.vitals_cols = self.data_dict['vitals'].columns.drop(target_col).tolist()
        if 'img' in modalities: self.include_ct = 'ct' in self.data_dict['img'].columns.tolist()
        
        self.on_epoch_end()
        
    def __len__(self) -> int:
        "Number of batches per epoch"
        return int(np.ceil(len(self.cid_list) / self.batch_size))

    def on_epoch_end(self):
        "Conditionally shuffles indexes every epoch"
        if self.shuffle:
            np.random.shuffle(self.cid_list)
    
    def get_cids(self) -> np.ndarray:
        return np.asarray(self.cid_list)
    
    def get_labels(self) -> np.ndarray:
        return self.target_df.loc[self.cid_list].to_numpy()
    
    def _init_getitem_mapper(self, modalities: List[str]) -> Dict[str, Callable]:
        dict_out = {}
        for key in modalities:
            if key=='static': dict_out['static']=self.__getstatic__
            elif key=='vitals': dict_out['vitals']=self.__getvitals__
            elif key=='img': dict_out['img']=self.__getimg__
            elif key=='lab': dict_out['lab']=self.__getlab__
            else: raise ValueError('Unknown modality type')
        return dict_out
    
    def __getitem__(self, index):
        cid_batch = self.cid_list[index*self.batch_size:(index+1)*self.batch_size]
        X = {}
        for key, func in self._getitem_mapper.items():
            if key == 'img':
                X = X | func(cid_batch)
            else:
                X[key] = func(cid_batch)
        
        y_batch = self.target_df[cid_batch].to_numpy()
        
        # Balance classes
        sweight_batch = np.fromiter((self.cweight_dict[y] for y in y_batch), dtype=float) 
        
        if self.repeat_target:
            y_batch = np.expand_dims(y_batch, axis=-1)
            y_batch = np.repeat(y_batch, len(self.modalities)+1, axis=-1)
        return X, y_batch, sweight_batch
    
    def __getstatic__(self, cid_batch: List[int]) -> np.ndarray:
        return self.data_dict['static'].loc[cid_batch, self.static_cols].astype(float).to_numpy()
    
    def __getvitals__(self, cid_batch: List[int]) -> np.ndarray:
        X_batch = [self.data_dict['vitals'].loc[cid, self.vitals_cols].to_numpy() for cid in cid_batch]
        X_batch = keras.preprocessing.sequence.pad_sequences(X_batch, padding='pre', dtype=float, value=self.padding_val)
        return X_batch
    
    def _load_img(self, fname: str):
        if pd.isna(fname):
            im_out = np.zeros((256, 256, 1))
        else:
            im_path = f'{self.image_dir}/{fname}'
            im_out = keras.preprocessing.image.load_img(im_path, color_mode='grayscale')
            im_out = keras.preprocessing.image.img_to_array(im_out)
        return im_out
    
    def __getimg__(self, cid_batch: List[int]) -> Dict[str, np.ndarray]:
        img_batch = {}
        
        X_batch = list()
        for cid in cid_batch:
            fname_list = self.data_dict['img'].loc[cid, 'fname'].tolist()
            temp_ims = np.array([self._load_img(f) for f in fname_list])
            X_batch.append(temp_ims)
        img_batch['img'] = keras.preprocessing.sequence.pad_sequences(X_batch, padding='pre', dtype=float, value=0.)
          
        return img_batch
    
    def __getlab__(self, cid_batch: List[int]) -> np.ndarray:
        X_batch = [self.data_dict['lab'].loc[cid, self.lab_cols].to_numpy() for cid in cid_batch]
        X_batch = keras.preprocessing.sequence.pad_sequences(X_batch, padding='pre', dtype=float, value=self.padding_val)
        return X_batch
        
        
        