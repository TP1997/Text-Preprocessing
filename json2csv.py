# Use pytorch_py3.8.8
    # Convert downloaded large json-file to multiple smaller csv-files.
    # Sort rows in decreasing order by reviewTime 

import os
import pandas as pd
import datetime
import json
import glob
import logging
import shutil

logging.basicConfig(
    # filename='out.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def to_date(d):
    mdy = [i.strip(',') for i in d.split()]
    return str(datetime.date(int(mdy[2]), int(mdy[0]), int(mdy[1])))

# All_Beauty
# AMAZON_FASHION
# CDs_and_Vinyl
# Cell_Phones_and_Accessories
# Digital_Music
# Electronics
# Industrial_and_Scientific
# Luxury_Beauty
# Musical_Instruments
# Software
# Video_Games

data_name = 'All_Beauty'
data_path = f'/home/tuomas/Python/Gradu/data_processing/datasets/Amazon review data/Original/{data_name}/'
os.chdir(data_path)
chunksize = 100000
df_json = pd.read_json(data_name+'.json', encoding='utf-8', lines=True, chunksize=chunksize)

sizes1 = []
filenames1 = []
reviewTimes = []
for j, chunk in enumerate(df_json):
    chunk = chunk.dropna(subset=['reviewText'])
    chunk.reviewTime = chunk.reviewTime.str.extract('([0-9]{4})', expand=False)
    #chunk['reviewTime'] = chunk['reviewTime'].apply(to_date)
    chunk = chunk.astype({'overall':'category', 'reviewTime':'category', 'reviewText':'object'})
    # Sort created csv rows by reviewTime
    chunk = chunk.sort_values(by=['reviewTime'], ascending=False)
    reviewTimes += list(set(chunk['reviewTime']))
    
    # Save
    fn = data_path + data_name+f'-{j}.csv'
    chunk.to_csv(fn, index=False, columns=['overall','reviewTime','reviewText'])
    sizes1.append(chunk.shape[0])
    filenames1.append(fn.split('/')[-1])   

reviewTimes = list(set(reviewTimes))
reviewTimes.sort(reverse=True)

#%% Sort all chunks csv-files to decreasing order by reviewTime

savepath = '/home/tuomas/Python/Gradu/data_processing/datasets/Amazon review data/Original/All_Beauty/Sorted/'
df_new = pd.DataFrame(columns=['overall', 'reviewTime', 'reviewText'])
j=0
sizes = []
filenames2 = []
for rt in reviewTimes:
    for i, fn in enumerate(filenames1):
        df = pd.read_csv(fn)
        df = df[df['reviewTime']==int(rt)]
        df_new = df_new.append(df, ignore_index=True)
        #print(df_new.shape[0])
        if(df_new.shape[0] > chunksize):
            fn = data_name+f'-{j}.csv'
            df_new.to_csv(savepath+fn, index=False)
            sizes.append(df_new.shape[0])
            df_new = pd.DataFrame(columns=['overall', 'reviewTime', 'reviewText'])
            filenames2.append(fn)
            j += 1
            
if df_new.shape[0] > 0:
    fn = data_name+f'-{j}.csv'
    df_new.to_csv(savepath+fn, index=False)
    sizes.append(df_new.shape[0])
    filenames2.append(fn)

# Write sizes.json
sizes_json = { filenames2[j]:sizes[j] for j,_ in enumerate(filenames2)}
with open(savepath+'sizes.json', 'w') as f:
    json.dump(sizes_json, f, indent=4)
    
#%% Remove old csv's and replace them by the sorted ones

for fn in filenames1:
    fn = data_path+fn
    os.remove(fn) 
    
for fn in filenames2:
    shutil.move(savepath+fn, data_path+fn)

shutil.move(savepath+'sizes.json', data_path+'sizes.json')

#%%

