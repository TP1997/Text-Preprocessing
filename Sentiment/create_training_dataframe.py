# Use pytorch_py3.8.8
# Create a single dataframe used for training

import os
import logging
import numpy as np
import glob 
import pandas as pd

logging.basicConfig(
    # filename='out.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__) 

# All_Beauty                    All_Beauty_sentence
# AMAZON_FASHION                AMAZON_FASHION_sentence
# CDs_and_Vinyl                 CDs_and_Vinyl_sentence
# Cell_Phones_and_Accessories   Cell_Phones_and_Accessories_sentence
# Digital_Music                 Digital_Music_sentence
# Electronics                   Electronics_sentence
# Industrial_and_Scientific     Industrial_and_Scientific_sentence
# Luxury_Beauty                 Luxury_Beauty_sentence
# Musical_Instruments           Musical_Instruments_sentence
# Software                      Software_sentence
# Video_Games                   Video_Games_sentence

data_name = 'All_Beauty_sentence'
train_size = 30000

# Define source and destination paths
data_path = f'/home/tuomas/Python/Gradu/data_processing/datasets/Amazon review data/Preprocess_common2/{data_name}/'
save_dir = '20k'
save_path = f'/home/tuomas/Python/Gradu/data_processing/datasets/Amazon review data/Training_data/{data_name}/{save_dir}/'
try:
    os.makedirs(save_path)
except FileExistsError:
    pass

filenames = glob.glob(data_path + '*.csv')
filenames.sort()

#%%
fctr = 3.5
reviews_left = int(train_size*fctr)

df_train = []
for i,fn in enumerate(filenames):
    logger.info(f"Processing {(i+1)}/{len(filenames)}")
    df = pd.read_csv(fn)[:reviews_left]
    df_train.append(df)
    reviews_left -= df.shape[0]
    if reviews_left <= 0:
        break

df_train = pd.concat(df_train)
df_train.to_csv(save_path+"train_raw.csv", index=False)


