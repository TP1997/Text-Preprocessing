# Use pytorch_py3.8.8
# Operations:
    # Remove non-ascii words.
    # Remove non-english texts.
    
import logging
import glob
import nltk.data
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
from multiprocessing import Pool
import langdetect
from langdetect import DetectorFactory
from nltk.tokenize import word_tokenize
import os

DetectorFactory.seed = 0
logging.basicConfig(
    # filename='out.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

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
save_path = f'/home/tuomas/Python/Gradu/data_processing/datasets/Amazon review data/Preprocess_common1/{data_name}/'
filenames = glob.glob(data_path + '*.csv')
filenames.sort()
filenames=[filenames[0]]

try:
    os.makedirs(save_path)
except FileExistsError:
    pass

#%%
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
stop = stopwords.words('english')

remove_nonascii_pattern = '([a-zA-Z]*[\u0080-\uFFFF]+[a-zA-Z]+|[a-zA-Z]+[\u0080-\uFFFF]+[a-zA-Z]*|[\u0080-\uFFFF]+)'

def remove_words(df, pattern=None): # If lemmatize: must use for loop
    df['reviewText'] = df['reviewText'].str.replace(pattern, '', regex=True)
    df['reviewText'] = df['reviewText'].str.split()
    df['reviewText'] = df['reviewText'].str.join(sep=' ')
    #df['reviewText'] = df['reviewText'].str.lower()
    return df

def remove_nonenglish_texts(df):
    is_english = []
    for rt in df['reviewText']:
        try:
            lang = langdetect.detect(rt)
            is_english.append(lang=='en')
        except:
            is_english.append(False)
            
    return df[is_english]

def add_SPOS_indices(df):
    positions = []
    for rt in df['reviewText']:
        rt_len = word_tokenize(rt)
        positions.append(np.arange(len(rt_len)).tolist())
        #df['SPOS_idx'] = np.arange(rt_len).tolist()
        
    df['SPOS_idx'] = positions
    return df
        
    
processes = 16
curr_docid = 0
for i, fn in enumerate(filenames):
    logger.info(f"Processing {(i+1)}/{len(filenames)}")
    df_split = pd.read_csv(fn)
    # Add document id column (for sentence tokenization)
    df_split.insert(0,'doc_id',np.arange(curr_docid, curr_docid+df_split.shape[0]))
    curr_docid += df_split.shape[0]
    # Add new column to indicate word positions in StanfordPOS dataframe (used in later preprocessing steps)
    #df_split.insert(loc=4,column='SPOS_idx',value=np.nan)
    
    df_split = np.array_split(df_split, processes)
    pool = Pool(processes)
    
    logger.info("Removing non-ascii words...")
    df_split = pool.starmap(remove_words, list(zip(df_split, [remove_nonascii_pattern]*processes)))
    
    logger.info("Removing non-english texts...")
    df_split = pool.map(remove_nonenglish_texts, df_split)
    
    logger.info("Adding word positions...")
    df_split = pool.map(add_SPOS_indices, df_split)
    
    logger.info("Saving...")
    pool.close()
    pool.join()
    df_split = pd.concat(df_split)
    df_split['reviewText'].replace('', np.nan, inplace=True) 
    df_split = df_split.dropna(subset=['reviewText'])
    fn = save_path + fn.split('/')[-1][:-4]+'_StanfordPOS.csv'
    df_split.to_csv(fn, index=False)
    
    
logger.info("Done!")
    
    
#%%
fn='/home/tuomas/Python/Gradu/data_processing/datasets/Amazon review data/Preprocessed_common/All_Beauty/All_Beauty-0_StanfordPOS.csv'
df = pd.read_csv(fn)

df['SPOS_idx']=df['SPOS_idx'].apply(eval)
    
    
    
    
    
    
    
    
    
    
    
    
    