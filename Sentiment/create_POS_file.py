# Use pytorch_py3.8.8

# Create a new dataframe that consist POS-tagged text from train_StanfordPOS.csv

import pandas as pd
import logging
from nltk.tag.stanford import StanfordPOSTagger
from nltk.tokenize import word_tokenize
from multiprocessing import Pool
import numpy as np

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
data_dir = '20k'
path = f'/home/tuomas/Python/Gradu/data_processing/datasets/Amazon review data/Training_data/{data_name}/{data_dir}/'
filename = 'train_StanfordPOS.csv'

path_to_model = '/home/tuomas/Java/lib/stanford-postagger-full-2020-11-17/models/english-bidirectional-distsim.tagger'
path_to_jar = '/home/tuomas/Java/lib/stanford-postagger-full-2020-11-17/stanford-postagger.jar'
stanford_tagger = StanfordPOSTagger(path_to_model, path_to_jar)

#%%
df_orig = pd.read_csv(path+'train_StanfordPOS.csv')
df_orig['reviewText'] = df_orig['reviewText'].apply(eval)
#df_orig = np.array_split(df_orig, 100)

df_POS = pd.DataFrame(columns=['doc_id','POS_tag'])

cntr = 0
for df in np.array_split(df_orig, 10):
    df_temp = pd.DataFrame(columns=['doc_id','POS_tag'])
    SPOS = stanford_tagger.tag_sents(df['reviewText'])
    df_temp['doc_id'] = df['doc_id']
    df_temp['POS_tag'] = SPOS
    
    df_POS = df_POS.append(df_temp, ignore_index=True)
    
    cntr += df.shape[0]
    logger.info(f"{cntr}/{df_orig.shape[0]}")
    
df_POS.to_csv(path+'train_POS.csv', index=False)

#%%
df_POS = pd.read_csv(path+'train_POS.csv')
df_POS['POS_tag']=df_POS['POS_tag'].apply(eval)


#%%
df_orig = pd.read_csv(path+'train_StanfordPOS.csv')
df_orig['reviewText'] = df_orig['reviewText'].apply(eval)

df_POS = pd.DataFrame(columns=['doc_id','POS_tag'])

processes = 16
cntr = 0
for df in np.array_split(df_orig, 10):
    df_split = np.array_split(df, processes)
    pool = Pool(processes)
    #df_split = pool.map(remove_nonalpha_chars, df_split)
    
#%% Testing the speed of Stanford POS-tagger vs default nltk POS-tagger
import time
import nltk
t1 = 'What is the airspeed of an unladen swallow?'
t2 = 'Must be higher than the speed of light, right?'
t3 = t1+' '+t2

#%%
start_time = time.time()
spos1 = stanford_tagger.tag(word_tokenize(t1))
spos2 = stanford_tagger.tag(word_tokenize(t2))
spos3 = stanford_tagger.tag(word_tokenize(t3))
print(f'Stanford POS: {time.time()-start_time} seconds')

#%%
start_time = time.time()
npos1 = nltk.pos_tag(word_tokenize(t1))
npos2 = nltk.pos_tag(word_tokenize(t2))
npos3 = nltk.pos_tag(word_tokenize(t3))
print(f'nltk POS: {time.time()-start_time} seconds')















