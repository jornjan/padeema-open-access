import pandas as pd

from tqdm import tqdm
from itertools import product

from utils.train_functions import mmodal_training_run


####################################################################
# Settings
####################################################################

targets = ['aleak', 'pneu']
shifts = [0, -1, -2]
min_los = 4
n_runs = 3

# Learning rates
mm_factor = 0.5
um_lr_dict = {
    'lab': 1e-2,
    'vitals': 1e-2,
    'img': 5e-3,
    'static': 1e-3,
    'late': 1,
    'mid': 1
}
mm_lr_dict = {key: value*mm_factor for key, value in um_lr_dict.items()}



####################################################################
# Load Data
####################################################################

# Complications
compl_df = pd.read_csv('example_data/compl_data.csv').set_index(['CID'])

# Lab results
lab_df = pd.read_csv('example_data/lab_data.csv').set_index(['CID', 'day'])
lab_df = lab_df.groupby('CID').ffill()

# Vital signs
vitals_df = pd.read_csv('example_data/vitals_data.csv').set_index(['CID', 'day', 'hour'])
vitals_df = pd.concat([vitals_df[['temp', 'af', 'hr']].groupby(['CID', 'day']).max(), 
                        vitals_df[['bp']].groupby(['CID', 'day']).min()], 
                        axis=1)

# Pre-operative patient characteristics (static data)
static_df = pd.read_csv('example_data/static_data.csv').set_index(['CID'])
static_df = static_df.fillna(static_df.median())


# Images
img_df = pd.read_csv('example_data/img_data.csv').set_index(['CID', 'day'])


data_dict = {
    'compl': compl_df,
    'lab': lab_df,
    'vitals': vitals_df,
    'img': img_df,
    'static': static_df
}


####################################################################
# Experiments
####################################################################

results = pd.DataFrame()
pbar = tqdm(total=n_runs*len(targets)*len(shifts))
for run, target, shift in product(range(n_runs), targets, shifts):
    temp_results = mmodal_training_run(data_dict, um_lr_dict, target, shift=shift, ft_factor=0.5, min_los=4)
    temp_results['target'] = target
    temp_results['window'] = f'{24*abs(shift)}h'
    
    results = pd.concat([results, temp_results], ignore_index=True)
    pbar.update()
    
pbar.close()
results.to_csv('example_results/full_results.csv')
print(results.groupby(['target', 'window', 'split', 'model']).mean().sort_values(['target', 'window','split', 'auc'], ascending=False))