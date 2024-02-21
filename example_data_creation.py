import pandas as pd
from random import randint
import numpy as np
from os import path

output_folder = 'example_data/'


####################################################################
# Settings
####################################################################

# Number of patients in example dataset
n_patients = 100

# Set minimum and maximum length of stay
min_los, max_los = 7, 14

# Set minimum day at which a complication can occur
min_complday = 4

# Anastomotic leakage probability
p_aleak = 0.2

# Pneumonia probability
p_pneu = 0.3

# Maximum values for lab measurements
LABMAXVALS = {
    'amy': 500,
    'crp': 500,
    'leuc': 30
}

# Maximum and minimum values for vital signs
VITALSMAXVALS = {
    'hr': 200,
    'temp': 42,
    'bp': 200,
    'af': 50
}

VITALSMINVALS = {
    'hr': 40,
    'temp': 35,
    'bp': 70,
    'af': 5
}


####################################################################
# Create complication dataframe
####################################################################

pat_list = []

for i in range(n_patients):
    cid = f'PAT{i+1:0>3}'

    # Generate random length of stay
    los = randint(7, 14)

    # Randomly assign anastomotic leakage
    aleak = np.random.choice([0, 1], p=[1 - p_aleak, p_aleak])
    aleak_day = np.nan if aleak==0 else randint(min_complday, los)

    # Randomly assign pneumonia
    pneu = np.random.choice([0, 1], p=[1 - p_pneu, p_pneu])
    pneu_day = np.nan if pneu==0 else randint(min_complday, los)

    pat_list.append(
        {
            'CID': cid,
            'los': los,
            'aleak': aleak,
            'aleak_day': aleak_day,
            'pneu': pneu,
            'pneu_day': pneu_day,
        }
    )

# Save data to CSV file
compl_df = pd.DataFrame.from_records(pat_list).set_index('CID')
compl_df.to_csv(path.join(output_folder, 'compl_data.csv'))



####################################################################
# Generate Static data
####################################################################



pat_list = []
for cid, pdata in compl_df.iterrows():
    pat_list.append(
        {
            'CID': cid,
            'age': randint(45, 85),
            'sex': randint(0, 1),
            'weight': randint(60, 120),
            'height': randint(160, 190),
            'asa': randint(1, 4),
            'surgery_type': randint(0, 1),
            'comdiam': randint(0, 1),
            'comlong': randint(0, 1),
            'comcard': randint(0, 1),
        }
    )

# Save to CSV
static_df = pd.DataFrame.from_records(pat_list).set_index('CID')
static_df.to_csv(path.join(output_folder, 'static_data.csv'))
static_df



####################################################################
# Generate Lab data
####################################################################

lab_df = pd.DataFrame()
for cid, pdata in compl_df.iterrows():
    los = int(pdata.loc['los'])

    # Create random lab values between 0 and 1 for each day
    temp_df = pd.DataFrame(np.random.random(size=(los, 3)), columns=['amy', 'crp', 'leuc'], index=range(1, los+1))

    # Multiply with lab ranges
    temp_df = temp_df.mul(LABMAXVALS).round(0)


    temp_df['day'] = temp_df.index
    temp_df['CID'] = cid
    lab_df = pd.concat([lab_df, temp_df], ignore_index=True)

# Save to CSV
lab_df = lab_df.set_index(['CID', 'day'])
lab_df.to_csv(path.join(output_folder, 'lab_data.csv'))



####################################################################
# Generate vitals data
####################################################################


vitals_df = pd.DataFrame()
for cid, pdata in compl_df.iterrows():
    los = int(pdata.loc['los'])

    # Create random lab values between 0 and 1 for each day for each of the 3 timeslots
    temp_df = pd.DataFrame(np.random.random(size=(3*los, 4)), columns=['hr', 'temp', 'bp', 'af'], 
                           index=pd.MultiIndex.from_product([range(1, los+1), [0, 8, 16]], names=['day', 'hour']))
    
    # Scale to realistic values
    temp_df = temp_df.mul(VITALSMAXVALS).add(VITALSMINVALS).round(0)
    temp_df = temp_df.reset_index()


    temp_df['CID'] = cid
    vitals_df = pd.concat([vitals_df, temp_df], ignore_index=True)

# Save to CSV
vitals_df = vitals_df.set_index(['CID', 'day', 'hour'])
vitals_df.to_csv(path.join(output_folder, 'vitals_data.csv'))



####################################################################
# Generate Image data
####################################################################

image_df = pd.DataFrame()
for cid, pdata in compl_df.iterrows():
    los = int(pdata.loc['los'])

    # Assign the template image on each day for each patient
    temp_df = pd.DataFrame(index=range(1, los+1))

    temp_df['day'] = temp_df.index
    temp_df['fname'] = 'thorax_example_img.png'
    temp_df['CID'] = cid

    image_df = pd.concat([image_df, temp_df], ignore_index=True)

# Save to CSV
image_df = image_df.set_index(['CID', 'day'])
image_df.to_csv(path.join(output_folder, 'img_data.csv'))