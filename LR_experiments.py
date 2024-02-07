import pandas as pd
from oesolib import train_functions

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
zgt_compl = pd.read_pickle('/mnt/data/jvbeld/ZGT/processed/ZGT_complications.pkl')

# Lab results
zgt_lab = pd.read_pickle('/mnt/data/jvbeld/ZGT/processed/ZGT_lab.pkl')
zgt_lab = zgt_lab.groupby('CID').ffill()

# Vital signs
zgt_vitals = pd.read_pickle('/mnt/data/jvbeld/ZGT/processed/ZGT_vitals.pkl')
zgt_vitals = pd.concat([zgt_vitals[['temp', 'af', 'hr']].groupby(['CID', 'day']).max(), 
                        zgt_vitals[['bp']].groupby(['CID', 'day']).min()], 
                        axis=1)

# Pre-operative patient characteristics (static data)
zgt_static = pd.read_pickle('/mnt/data/jvbeld/ZGT/processed/ZGT_static.pkl')
zgt_static = zgt_static.fillna(zgt_static.median())




####################################################################
# Experiments
####################################################################



# Lab results
lab_results = pd.DataFrame()
for target, shift in tqdm(product(targets, shifts), total=len(targets)*len(shifts)):
    temp_results = train_functions.train_lr_model(zgt_compl, [zgt_lab],
                                                  min_los=min_los, target=target,
                                                  shift=shift, lr_C=Cvals['lab'],
                                                  n_runs=n_runs, verbose=False)
    lab_results = pd.concat([lab_results, temp_results], ignore_index=True)

print('Lab Results:')
print(lab_results.groupby(['target', 'window', 'split']).mean())



# Vital signs
vitals_results = pd.DataFrame()
for target, shift in tqdm(product(targets, shifts), total=len(targets)*len(shifts)):
    temp_results = train_functions.train_lr_model(zgt_compl, [zgt_vitals],
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
    temp_results = train_functions.train_lr_model(zgt_compl, [zgt_static],
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
    temp_results = train_functions.train_lr_model(zgt_compl, [zgt_lab, zgt_vitals],
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
    temp_results = train_functions.train_lr_model(zgt_compl, [zgt_lab, zgt_vitals, zgt_static],
                                                  min_los=min_los, target=target,
                                                  shift=shift, lr_C=Cvals['full'],
                                                  n_runs=n_runs, verbose=False)
    full_lr_results = pd.concat(
        [full_lr_results, temp_results], ignore_index=True)

print('Lab+Vitals+static Results:')
print(full_lr_results.groupby(['target', 'window', 'split']).mean())
