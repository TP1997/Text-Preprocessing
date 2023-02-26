# Use pytorch_py3.8.8
# Copy text data from csv-file to txt-file to form a raw training file for the further processing.

# Before run specify following variables:
    # data_name
    # by_reviewTime
    # random_order
    # train_size
    # save_dir
    # fctr (optional)
    
import os
import pandas as pd
import numpy as np
import glob
import re
import json
import logging

logging.basicConfig(
    # filename='out.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


# All_Beauty                    All_Beauty_sentences
# AMAZON_FASHION                AMAZON_FASHION_sentences
# CDs_and_Vinyl                 CDs_and_Vinyl_sentences
# Cell_Phones_and_Accessories   Cell_Phones_and_Accessories_sentences
# Digital_Music                 Digital_Music_sentences
# Electronics                   Electronics_sentences
# Industrial_and_Scientific     Industrial_and_Scientific_sentences
# Luxury_Beauty                 Luxury_Beauty_sentences
# Musical_Instruments           Musical_Instruments_sentences
# Software                      Software_sentences
# Video_Games                   Video_Games_sentences

data_name = 'All_Beauty_sentences'
by_reviewTime = True
random_order = False
year = '2012'
train_size = 20000

# Define source and destination paths
#data_dir = f'{data_name}' if random_order else f'By_year/{data_name}'
data_path = f'/home/tuomas/Python/Gradu/data_processing/datasets/Amazon review data/Preprocessed_common/{data_name}/'
save_dir = '20k'
save_path = f'/home/tuomas/Python/Gradu/data_processing/datasets/Amazon review data/Training_data/{data_name}/{save_dir}/'
try:
    os.mkdir(save_path)
except FileExistsError:
    pass

# Fetch information about csv-sizes
sizes = None
with open(data_path + 'sizes.json', 'r') as f:
    sizes = json.load(f)
    

#%% 

def get_sampling_counts2(values: np.array, train_size, prop=False):
    if prop:
        
        pass
    else:
        r_min = 1.0
        r_max = sizes['total'] if random_order else sizes[year]
        t_min = 1.0
        t_max = train_size
        
        values = values.astype('float')
        values -= r_min
        values /= (r_max-r_min)
        values *= (t_max-t_min)
        values += t_min
        
        return values.astype(int)
        
def write_training_file(fn_source, fn_dest, n_samples):
    # Sample dataframes based on their specific counts & save to training file 
    filenames.sort(key=lambda x: int(re.findall(r'-\d+', x.split('/')[-1])[0][1:]))
    with open(save_path + fn_dest, 'w') as trainfile:
          for i, fn in enumerate(filenames):
              logger.info(f"Processing {(i+1)}/{len(filenames)}")
              df = pd.read_csv(fn)
              if n_samples[fn] < df.shape[0]:
                  df = df.sample(n = n_samples[fn], random_state=rs, replace=False)
              for txt in df.reviewText:
                  #print(txt)
                  trainfile.write(txt + '\n')
        

rs = np.random.RandomState(1234)
fctr = 4.5

if by_reviewTime:
    # Calculate review counts of separate dataframes.
    reviews_left = int(train_size*fctr)
    with open(save_path + 'train_raw.txt', 'w') as trainfile:
        for fn, v in zip(sizes.keys(), sizes.values()):
            df = pd.read_csv(fn)
            amount = min(v, reviews_left) #v if v<reviews_left else reviews_left
            for txt in df[:amount].reviewText:
                trainfile.write(txt + '\n')
                
            reviews_left -= v
            if reviews_left < 1:
                break
        
elif random_order:
    # Calculate review counts of separate dataframes.
    df_sizes = np.array(list(sizes.values()))
    # Get dataframe-specific review counts to fetch.
    df_sizes = get_sampling_counts2(df_sizes, train_size*fctr)
    df_sizes = {k:v for k,v in zip(sizes.keys(),df_sizes)}
    # Sample dataframes based on their specific counts & save to training file 
    filenames = glob.glob(data_path + '*.csv')
    write_training_file(filenames, 'train_raw.txt', df_sizes)
    
else:
    # Calculate review counts of separate dataframes.
    df_sizes = np.ones(int((sizes[year] - (sizes[year]%sizes['chunksize']))/sizes['chunksize']))*sizes['chunksize']
    if sizes[year]%sizes['chunksize'] > 0:
        df_sizes = np.append(df_sizes, sizes[year]%sizes['chunksize'])
    
    # Get dataframe-specific review counts to fetch.
    df_sizes = get_sampling_counts2(df_sizes, train_size*1.3)
        
    filenames = glob.glob(data_path + f'*{year}-*.csv')
    write_training_file(filenames, f'train_{year}_raw.txt', df_sizes)



#%%
reviews_left=100000
for fn, v in zip(sizes.keys(), sizes.values()):
    amount =  min(v, reviews_left) #v if v<reviews_left else reviews_left
    print(f'{fn} : {amount}')
    reviews_left -= amount
    if reviews_left < 1:
        break









