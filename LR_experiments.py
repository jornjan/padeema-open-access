import pandas as pd
from utils import train_functions

from tqdm import tqdm
from itertools import product



####################################################################
# Settings
####################################################################

targets = ['aleak', 'pneu']
shifts = [0, -1, -2]
min_los = 4
n_runs = 10

Cvals = {
    'lab': 0.001,
    'vitals': 0.5,
    'labvitals': 0.001,
    'static': 0.5,
    'full': 0.001
}


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




####################################################################
# Experiments
####################################################################



# Lab results
lab_results = pd.DataFrame()
for target, shift in tqdm(product(targets, shifts), total=len(targets)*len(shifts)):
    temp_results = train_functions.train_lr_model(compl_df, [lab_df],
                                                  min_los=min_los, target=target,
                                                  shift=shift, lr_C=Cvals['lab'],
                                                  n_runs=n_runs, verbose=False)
    lab_results = pd.concat([lab_results, temp_results], ignore_index=True)

print('Lab Results:')
print(lab_results.groupby(['target', 'window', 'split']).mean())



# Vital signs
vitals_results = pd.DataFrame()
for target, shift in tqdm(product(targets, shifts), total=len(targets)*len(shifts)):
    temp_results = train_functions.train_lr_model(compl_df, [vitals_df],
                                                  min_los=min_los, target=target,
                                                  shift=shift, lr_C=Cvals['vitals'],
                                                  n_runs=n_runs, verbose=False)
    vitals_results = pd.concat(
        [vitals_results, temp_results], ignore_index=True)

print('Vital Signs Results:')
print(vitals_results.groupby(['target', 'window', 'split']).mean())



# Pre-operative patient characteristics (static data)
static_results = pd.DataFrame()
for target, shift in tqdm(product(targets, shifts), total=len(targets)*len(shifts)):
    temp_results = train_functions.train_lr_model(compl_df, [static_df],
                                                  min_los=min_los, target=target,
                                                  shift=shift, lr_C=Cvals['static'],
                                                  n_runs=n_runs, verbose=False)
    static_results = pd.concat(
        [static_results, temp_results], ignore_index=True)

print('Static Data Results:')
print(static_results.groupby(['target', 'window', 'split']).mean())



# Lab results + Vital signs 
labvitals_results = pd.DataFrame()
for target, shift in tqdm(product(targets, shifts), total=len(targets)*len(shifts)):
    temp_results = train_functions.train_lr_model(compl_df, [lab_df, vitals_df],
                                                  min_los=min_los, target=target,
                                                  shift=shift, lr_C=Cvals['labvitals'],
                                                  n_runs=n_runs, verbose=False)
    labvitals_results = pd.concat(
        [labvitals_results, temp_results], ignore_index=True)

print('Lab+Vitals Results:')
print(labvitals_results.groupby(['target', 'window', 'split']).mean())



# Lab results + Vital signs + Static data
full_lr_results = pd.DataFrame()
for target, shift in tqdm(product(targets, shifts), total=len(targets)*len(shifts)):
    temp_results = train_functions.train_lr_model(compl_df, [lab_df, vitals_df, static_df],
                                                  min_los=min_los, target=target,
                                                  shift=shift, lr_C=Cvals['full'],
                                                  n_runs=n_runs, verbose=False)
    full_lr_results = pd.concat(
        [full_lr_results, temp_results], ignore_index=True)

print('Lab+Vitals+static Results:')
print(full_lr_results.groupby(['target', 'window', 'split']).mean())
