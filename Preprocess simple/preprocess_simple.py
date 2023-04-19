import glob
import logging
import re
import os
import pandas as pd
import numpy as np
from multiprocessing import Pool
import langdetect
from langdetect import DetectorFactory
#import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag

#nltk.download('wordnet')
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
save_name = 'Test'
save_path = f'/home/tuomas/Python/Gradu/data_processing/datasets/Amazon review data/Preprocess simple/Training data/All_Beauty/{save_name}/'
try:
    os.makedirs(save_path)
except FileExistsError:
    pass
filenames = glob.glob(data_path + '*.csv')
filenames.sort()
#filenames=[filenames[0]]

#%% Preprocessing (1) - Global parameters
remove_nonascii_words_pat = r'\b[\w\'-]*[^\x00-\x7F]+[\w\'-]*\b'
remove_nonascii_char_pat = r'[^\x00-\x7F]'
ascii_pat = remove_nonascii_char_pat

remove_nonalpha_words_pat = r'([a-zA-Z]*[^a-zA-Z ]+[a-zA-Z]+|[a-zA-Z]+[^a-zA-Z ]+[a-zA-Z]*|[^a-zA-Z ]+)'
remove_nonalpha_char_pat = r"[^a-zA-Z' ]" # Don't remove "/'" and " " terms.
alpha_pat = remove_nonalpha_char_pat

tag_dict = {"J":wordnet.ADJ,
            "N":wordnet.NOUN,
            "V":wordnet.VERB,
            "R":wordnet.ADV}

lmtzr = WordNetLemmatizer()

# Create custom stopword list which includes also neutral terms, e.g. "wouldn't" -> ("wouldn't", "would").
def create_stopword_list(custom=True):
    stop = []
    if custom:
        for sw in stopwords.words('english'):
            sw = word_tokenize(sw)+[sw]
            stop += sw
        stop = list(set(stop))
    else:
        stop = stopwords.words('english')
    
    stop.append("'s")
    stop.append("'m")
    stop.remove("n't")
    return stop
        
stop = create_stopword_list(True)

minlen_word = 3

processes = 16
chunk_size_part1 = 1000
data_size_part1 = 10000#00

#s = '6Â 918! Â417Â 712. Hello all!' 
#s = "compact perfect trim n't recommend"
#sdf = pd.DataFrame(data={'text':[s]})
#sdf['text'].str.replace(remove_nonalpha_char_pat, '', regex=True)

#%% Preprocessing (1) - Run
def remove_pattern(df, pat):
    df['reviewText'] = df['reviewText'].str.replace(pat, '', regex=True)
    df['reviewText'] = df['reviewText'].str.split()
    df['reviewText'] = df['reviewText'].str.join(sep=' ')
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
    
def lemmatize_text(text : list):
    tagged_text = [(w, tag_dict.get(pos[0].upper(), wordnet.NOUN)) for w,pos in pos_tag(text)]
    text_lmtz = [lmtzr.lemmatize(w, pos) for w, pos in tagged_text]
    return text_lmtz
    
def lemmatize_dataframe(df):
    df['reviewText'] = df['reviewText'].apply(word_tokenize)
    df['reviewText'] = df['reviewText'].apply(lemmatize_text)
    df['reviewText'] = df['reviewText'].str.join(sep=' ')
    return df
    
def remove_stopwords_dataframe(df):
    df['reviewText'] = df['reviewText'].str.lower()
    df['reviewText'] = df['reviewText'].str.split() #df['reviewText'] = df['reviewText'].apply(word_tokenize)
    df['reviewText'] = df['reviewText'].apply(lambda wlist: [w for w in wlist if w not in stop and len(w)>=minlen_word])
    df['reviewText'] = df['reviewText'].str.join(sep=' ')
    return df    
    
processes = 16
dfs = []
for i, fn in enumerate(filenames):
    logger.info(f"Processing {(i+1)}/{len(filenames)}")
    i=0
    file_ready = False
    while not file_ready:
        df_split = pd.read_csv(fn)[i*chunk_size_part1:(i+1)*chunk_size_part1]
        if df_split.shape[0] == 0:
            file_ready = True
            continue
        df_split = np.array_split(df_split, processes)
        pool = Pool(processes)
    
        logger.info("Removing non-ascii characters/words...")
        #df_split = pool.map(remove_pattern, df_split)
        df_split = pool.starmap(remove_pattern, list(zip(df_split, [ascii_pat]*processes)))
    
        logger.info("Removing non-english texts...")
        df_split = pool.map(remove_nonenglish_texts, df_split)
    
        logger.info("Lemmatizing words")
        df_split = pool.map(lemmatize_dataframe, df_split)
        
        logger.info("Removing non-alphabetic characters/words...")
        df_split = pool.starmap(remove_pattern, list(zip(df_split, [alpha_pat]*processes)))
    
        logger.info("Removing stopwords and short words...")
        df_split = pool.map(remove_stopwords_dataframe, df_split)
    
    
        pool.close()
        pool.join()
        df_split = pd.concat(df_split)
        df_split['reviewText'].replace('', np.nan, inplace=True) 
        df_split = df_split.dropna(subset=['reviewText'])
        dfs.append(df_split)
    
        data_size_part1 -= df_split.shape[0]
        if data_size_part1 <= 0:
            break
        i+=1
    if data_size_part1 <= 0:
        break
    
df_part1 = pd.concat(dfs)
df_part1.to_csv(save_path+'df_p1.csv')

#%% Preprocessing (2) - Global parameters
prune_rules = {'mindf_word': (True, 5),             # Remove words having document frequency smaller than "mindf_word".X
               'minf_word': (True,10),              # Remove words occurred less than "mincount_word" times.
               'least_freq_quant': (False, 0.99),   # Remove least frequent % of words.X !!! ZIP'F LAW !!!
               'most_freq_quant': (False, 0.01),    # Remove most frequent % of words. !!! KEEP THIS FALSE !!!
               'minlen_doc': (True, 4),             # Remove documents shorter than "minlen_doc"X
               }

train_size = 25000
df_part1['reviewText_wid'] = ""

#%% Preprocessing (2) - Run
from gensim.corpora import Dictionary

def dataframe_generator(df):
    for _,row in df.iterrows():
        yield row['reviewText'].split()

dictnr = Dictionary(dataframe_generator(df_part1))
w2id = dictnr.token2id
#word_doc_freq = {dictnr[k]:v for k,v in zip(dictnr.dfs.keys(),dictnr.dfs.values())}
#word_tot_freq = {dictnr[k]:v for k,v in zip(dictnr.cfs.keys(),dictnr.cfs.values())}


def remove_words_dataframe(df):
    df['reviewText'] = df['reviewText'].str.split()
    df['reviewText'] = df['reviewText'].apply(lambda wlist: [w for w in wlist if dictnr.dfs[w2id[w]]>=prune_rules['mindf_word'][1] or dictnr.cfs[w2id[w]]>=prune_rules['minf_word'][1]])
    #df['reviewText'] = df['reviewText'].str.join(sep=' ')
    return df   

def remove_documents_dataframe(df):
    df['reviewText'] = df['reviewText'].apply(lambda wlist: wlist if len(wlist)>=prune_rules['minlen_doc'][1] else [])
    #df['reviewText'] = df['reviewText'].str.join(sep=' ')
    return df
    
def to_wid_dataframe(df):
    df['reviewText_wid'] = df['reviewText'].apply(dictnr.doc2idx)
    #df['reviewText_wid'] = df['reviewText_wid'].apply(lambda wids: ' '.join(str(wid) for wid in wids))
    df['reviewText'] = df['reviewText'].str.join(sep=' ')
    return df
    
df_split = np.array_split(df_part1, processes)
pool = Pool(processes)
logger.info("Remove words having small document frequency or total frequency...")
df_split = pool.map(remove_words_dataframe, df_split)

logger.info("Remove short documents...")
df_split = pool.map(remove_documents_dataframe, df_split)

logger.info("Transform words into word indices...")
df_split = pool.map(to_wid_dataframe, df_split)

pool.close()
pool.join()
df_split = pd.concat(df_split)
df_split['reviewText'].replace('', np.nan, inplace=True) 
df_split = df_split.dropna(subset=['reviewText'])

df_split.to_csv(save_path+'df_p2.csv')
dictnr.save(save_path+'dictionary.dict')
    
    
    
    
    
    
    
    
    
    
    
    