# Use pytorch_py3.8.8
# Operations:
    # Tokenize to sentences
    # Remove non-ascii words.
        # Lowercase
    # Remove non-english texts.
    # Parse texts.
    # Remove non-alphabetic words.
import os

import nltk.data
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag.stanford import StanfordPOSTagger

from multiprocessing import Pool

import pandas as pd
import numpy as np
import logging
import langdetect
from sklearn.feature_extraction.text import CountVectorizer
from gensim.corpora import Dictionary
import gensim
gensim_logger = logging.getLogger('gensim').setLevel(logging.WARNING)
import glob
import re
import json

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
prefix = ''
to_sentences = False
if to_sentences:
    prefix = '_sentences'
data_path = f'/home/tuomas/Python/Gradu/data_processing/datasets/Amazon review data/Original/{data_name}/'
save_path = f'/home/tuomas/Python/Gradu/data_processing/datasets/Amazon review data/Preprocessed_common/{data_name+prefix}/'
filenames = glob.glob(data_path + '*.csv')
filenames.sort()

# Set Stanford pos-tagger
path_to_model = '/home/tuomas/Java/lib/stanford-postagger-full-2020-11-17/models/english-bidirectional-distsim.tagger'
path_to_jar = '/home/tuomas/Java/lib/stanford-postagger-full-2020-11-17/stanford-postagger.jar'
stanford_tagger = StanfordPOSTagger(path_to_model, path_to_jar)

#%%
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
stop = stopwords.words('english')

remove_nonascii_pattern = '([a-zA-Z]*[\u0080-\uFFFF]+[a-zA-Z]+|[a-zA-Z]+[\u0080-\uFFFF]+[a-zA-Z]*|[\u0080-\uFFFF]+)'
remove_nonalpha_pattern = r'([a-zA-Z]*[^a-zA-Z ]+[a-zA-Z]+|[a-zA-Z]+[^a-zA-Z ]+[a-zA-Z]*|[^a-zA-Z ]+)'

# These are used in the second step of preprocessing.
#args = {'minlen_word':3,
 #       'minlen_sentence':5,
  #      'mincnt_word':5,
   #     'min_df':0
    #    }


def remove_words(df, pattern=None): # If lemmatize: must use for loop
    df['reviewText'] = df['reviewText'].str.replace(pattern, '', regex=True)
    df['reviewText'] = df['reviewText'].str.split()
    df['reviewText'] = df['reviewText'].str.join(sep=' ')
    df['reviewText'] = df['reviewText'].str.lower()
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

def pos_tag_text(text : list):
    tag_dict = {"J":wordnet.ADJ,
                "N":wordnet.NOUN,
                "V":wordnet.VERB,
                "R":wordnet.ADV}
    tagged_text = [(w, tag_dict.get(pos[0].upper(), wordnet.NOUN)) for w,pos in nltk.pos_tag(text)]
    return tagged_text
    
def stanford_pos_tagging(text : str):
    tagged_words = stanford_tagger.tag(word_tokenize(text))
    tagged_text = ' '.join([f'{w}:{pos}' for w,pos in tagged_words])
    return tagged_text
    
def pos_tag_original_documents(df):
    df['origPOS'] = df['reviewText'].apply(stanford_pos_tagging)
    return df
    
def remove_stopwords(text):
    text_token = word_tokenize(text)
    text_rmstop = [w for w in text_token if w not in stop]
    return text_rmstop

def lemmatize_words(text : list):
    lmtzr = WordNetLemmatizer()
    text_lmtz = []
    for w,pos in pos_tag_text(text):
        text_lmtz.append(lmtzr.lemmatize(w, pos))
    return text_lmtz

def parse_dataframe(df):
    #df['reviewText'] = df['reviewText'].apply(parse_text)
    df['reviewText'] = df['reviewText'].apply(remove_stopwords)
    #df['reviewText'] = df['reviewText'].apply(lemmatize_words)
    df['reviewText'] = df['reviewText'].str.join(sep=' ')
    return df

def split_sentences(df):
    df_new = pd.DataFrame(columns=df.columns)
    for _,row in df.iterrows():
        sentences = sent_tokenize(row['reviewText'])
        for s in sentences:
            newrow = row[:3]
            newrow['reviewText'] = s
            df_new = df_new.append(newrow, ignore_index=True)
    
    return df_new

def combine_sentences(df):
    
    df_new = pd.DataFrame(columns=df.columns)
    
    # Group by document ids
    doc_ids = list(set(df['doc_id']))
    df = df.groupby('doc_id')
    
    for did in doc_ids:
        group = df.get_group(did)
        newrow = group.iloc[0][:3]
        sentences = list(group['reviewText'])
        sentences = ' . '.join(sentences)
        newrow['reviewText'] = sentences
        df_new = df_new.append(newrow, ignore_index=True)
        
    return df_new


#filenames = ['/home/tuomas/Python/Gradu/data_processing/datasets/Amazon review data/Original/All_Beauty/All_Beauty-0_test.csv']
processes = 16
sizes = []
fns = []
for i, fn in enumerate(filenames):
    logger.info(f"Processing {(i+1)}/{len(filenames)}")
    
    df_split = pd.read_csv(fn)
    #if to_sentences:
    # Add document id column (for sentence tokenization)
    df_split.insert(0,'doc_id',np.arange(df_split.shape[0]))
    df_split = np.array_split(df_split, processes)
        
    # Create process pool
    pool = Pool(processes)
        
    if to_sentences:
        logger.info("Tokenizing to sentences...")
        df_split = pool.map(split_sentences, df_split)
        
    logger.info("Removing non-ascii words...")    
    df_split = pool.starmap(remove_words, list(zip(df_split, [remove_nonascii_pattern]*processes))) #df_split,remove_nonascii_pattern)
    
    logger.info("Removing non-english texts...")
    df_split = pool.map(remove_nonenglish_texts, df_split)
    
    # Savepoint 1: Save current modifications for Stanford POS-tagging
    logger.info("Savepoint 1...")
    pool.close()
    pool.join()
    df_split = pd.concat(df_split)
    fn = save_path + fn.split('/')[-1][:-4]+'_StanfordPOS.csv'
    df_split.to_csv(fn, index=False)
    
    # Continue with preprocessing
    df_split = np.array_split(df_split, processes)
    pool = Pool(processes)
    
    #logger.info("Pos-tagging for MaxEnt-classifier...")
    #df_split = pool.map(pos_tag_original_documents, df_split)
    
    logger.info("Parsing texts...") # Removes puncuations
    df_split = pool.map(parse_dataframe, df_split)
    
    logger.info("Removing non-alphabetic words...")
    df_split = pool.starmap(remove_words, list(zip(df_split, [remove_nonalpha_pattern]*processes)))
      
    if to_sentences:
        logger.info("Combining sentences...")
        df_split = pool.map(combine_sentences, df_split)
    
    pool.close()
    pool.join()
    # Savepoint 2: Save preprocessed dataframe
    logger.info("Savepoint 2...")
    df_split = pd.concat(df_split)
    df_split['reviewText'].replace('', np.nan, inplace=True) 
    df_split = df_split.dropna(subset=['reviewText'])
    fn = save_path + fn.split('/')[-1][:-4]+'.csv'
    df_split.to_csv(fn, index=False)
    
    sizes.append(df_split.shape[0])
    fns.append(fn)
    
# Write sizes.json

sizes_json = { fns[j]:sizes[j] for j,_ in enumerate(fns)}
sizes_json['total'] = sum(sizes)
with open(save_path+'sizes.json', 'w') as f:
    json.dump(sizes_json, f, indent=4)


#%%
a1="small and compact. Perfect for trimming! I wouldn't recommend usign for a close shave however."
a2="Best value for the Braun refills. My go to source."

















