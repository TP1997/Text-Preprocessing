# Use pytorch_py3.8.8
# Operations:
    # Tokenize to sentences
    # Parse texts.
    # Remove non-alphabetic words.
    
import logging
import pandas as pd
import glob
import numpy as np
from multiprocessing import Pool
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, wordnet
import re
import os
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
import nltk

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
to_sentences = True
if to_sentences:
    prefix = 'sentence'
    
data_path = f'/home/tuomas/Python/Gradu/data_processing/datasets/Amazon review data/Preprocess_common1/{data_name}/'
save_path = f'/home/tuomas/Python/Gradu/data_processing/datasets/Amazon review data/Preprocess_common2/{data_name}_{prefix}/'
filenames = glob.glob(data_path + '*_StanfordPOS.csv')
filenames.sort()
#filenames=[filenames[0]]

try:
    os.makedirs(save_path)
except FileExistsError:
    pass

#%%
stop = stopwords.words('english')
stop.remove('not')
stop.append("'s")
stop.append("'m")
remove_nonalpha_words_pattern = r'([a-zA-Z]*[^a-zA-Z ]+[a-zA-Z]+|[a-zA-Z]+[^a-zA-Z ]+[a-zA-Z]*|[^a-zA-Z ]+)'
remove_nonalpha_chars_pattern = r"[^a-zA-Z' ]"
ps = PorterStemmer()
nltk.download('wordnet')
lmtzr = WordNetLemmatizer()
tag_dict = {"J":wordnet.ADJ,
            "N":wordnet.NOUN,
            "V":wordnet.VERB,
            "R":wordnet.ADV}

def split_sentences(df):
    df_new = pd.DataFrame(columns=df.columns)
    for _,row in df.iterrows():
        sentences = sent_tokenize(row['reviewText'])
        SPOS_idx = row['SPOS_idx']
        i = 0
        for s in sentences:
            n_words = len(word_tokenize(s))
            newrow = row[:3]
            newrow['reviewText'] = word_tokenize(s) #s
            newrow['SPOS_idx'] = SPOS_idx[i:(i+n_words)]
            i+=n_words
            df_new = df_new.append(newrow, ignore_index=True)
    
    return df_new
                
def remove_nonalpha_chars(df):
    df_new = pd.DataFrame(columns=df.columns)
    for _,row in df.iterrows():
        #SPOS_idx = row['SPOS_idx']
        words_new = []
        SPOS_idx_new = []
        for spos_idx, w in zip(row['SPOS_idx'], row['reviewText']): #zip(row['SPOS_idx'], word_tokenize(row['reviewText'])):
            w = re.sub(remove_nonalpha_chars_pattern, "", w)
            if len(w)>0:
                words_new.append(w.lower())
                SPOS_idx_new.append(spos_idx)
                #SPOS_idx_new.append(SPOS_idx[idx])
        
        newrow = row[:3]
        newrow['reviewText'] = words_new #' '.join(words_new)
        newrow['SPOS_idx'] = SPOS_idx_new
        df_new = df_new.append(newrow, ignore_index=True)
        
    return df_new
   
def remove_stopwords(df):
    df_new = pd.DataFrame(columns=df.columns)
    for _,row in df.iterrows():
        words_new = []
        #SPOS_idx = row['SPOS_idx']
        SPOS_idx_new = []
        for spos_idx, w in zip(row['SPOS_idx'], row['reviewText']): #zip(row['SPOS_idx'],word_tokenize(row['reviewText'])):
            if w not in stop:
                words_new.append(w)
                SPOS_idx_new.append(spos_idx)
                #SPOS_idx_new.append(SPOS_idx[idx])

        newrow = row[:3]
        newrow['reviewText'] = words_new #' '.join(words_new)
        newrow['SPOS_idx'] = SPOS_idx_new
        df_new = df_new.append(newrow, ignore_index=True)
    
    return df_new

def stem_text(text):
    text_token = text #word_tokenize(text)
    text_token = [ps.stem(w) for w in text_token]
    return text_token #' '.join(text_token)
    
def stem_dataframe(df):
    df['reviewText'] = df['reviewText'].apply(stem_text)
    return df

def lemmatize_text(text : list):
    tagged_text = [(w, tag_dict.get(pos[0].upper(), wordnet.NOUN)) for w,pos in nltk.pos_tag(text)]
    text_lmtz = []
    for w, pos in tagged_text:
        text_lmtz.append(lmtzr.lemmatize(w, pos))
    return text_lmtz

def lemmatize_dataframe(df):
    df['reviewText'] = df['reviewText'].apply(lemmatize_text)
    return df
    
# Is this needed?         
def combine_sentences(df):
    df_new = pd.DataFrame(columns=df.columns)
    
    # Remove empty sentences
    df['reviewText'] = df['reviewText'].apply(lambda wlist: np.nan if len(wlist)==0 else wlist)
    df = df.dropna(subset=['reviewText'])
    
    # Group by document ids
    doc_ids = list(set(df['doc_id']))
    df = df.groupby('doc_id')
    
    for did in doc_ids:
        group = df.get_group(did)
        newrow = group.iloc[0][:3]
        
        #for sentence, spox_idxs in zip(list(group['reviewText']), list(group['SPOS_idx'])):
            
        sentences = list(group['reviewText'])
        sentences = [ s+['.'] for s in sentences] #' . '.join(sentences)
        sentences = [y for x in sentences for y in x][:-1]
        newrow['reviewText'] = sentences
        
        SPOS_idxs = list(group['SPOS_idx'])
        SPOS_idxs = [y for x in SPOS_idxs for y in x]
        newrow['SPOS_idx'] = SPOS_idxs
        
        df_new = df_new.append(newrow, ignore_index=True)
        
    return df_new

#%%

processes = 16
sizes = []
fns = []
for i, fn in enumerate(filenames):
    logger.info(f"Processing {(i+1)}/{len(filenames)}")
    
    df_split = pd.read_csv(fn)
    df_split['SPOS_idx']=df_split['SPOS_idx'].apply(eval)
    df_split = np.array_split(df_split, processes)
    pool = Pool(processes)
    
    if to_sentences:
        logger.info("Tokenizing to sentences...")
        df_split = pool.map(split_sentences, df_split)
    
    logger.info("Removing non-alphabetic characters...")
    #df_split = pool.starmap(remove_nonalpha_chars, list(zip(df_split, [remove_nonalpha_chars_pattern]*processes)))
    df_split = pool.map(remove_nonalpha_chars, df_split)
    
    logger.info("Removing stopwords...")
    df_split = pool.map(remove_stopwords, df_split)
    
    #logger.info("Stemming words")
    #df_split = pool.map(stem_dataframe, df_split)
    
    logger.info("Lemmatizing words")
    df_split = pool.map(lemmatize_dataframe, df_split)
    
    if to_sentences:
        logger.info("Combining sentences...")
        df_split = pool.map(combine_sentences, df_split)
    
    logger.info("Saving...")
    pool.close()
    pool.join()
    df_split = pd.concat(df_split)
    df_split['reviewText'] = df_split['reviewText'].apply(lambda wlist: np.nan if len(wlist)==0 else wlist)
    df_split = df_split.dropna(subset=['reviewText'])
    fn = save_path + data_name + f'-{i}{prefix}.csv'
    df_split.to_csv(fn, index=False)
    
logger.info("Done!")

#%%

#%%
a = [['aa','aa','aa','aa'],['bb','bb','bbv'],['cc','cc','cc','cc','cc']]
b = [ e+['.'] for e in a]#[:-1]
b = [y for x in b for y in x][:-1]

#%%
x = "div id ´´ ´´ class"

#%%
a1="notsmall and compact. Perfect for trimming! I wouldn't recommend usign for a close shave however."
a2="Best value for the Braun refills. My go to source."
a3='What is the airspeed of an unladen swallow?'

#%%
from nltk.tag.stanford import StanfordPOSTagger
# Set Stanford pos-tagger
path_to_model = '/home/tuomas/Java/lib/stanford-postagger-full-2020-11-17/models/english-bidirectional-distsim.tagger'
path_to_jar = '/home/tuomas/Java/lib/stanford-postagger-full-2020-11-17/stanford-postagger.jar'
stanford_tagger = StanfordPOSTagger(path_to_model, path_to_jar)

tagged_words1 = stanford_tagger.tag(word_tokenize(a1))
tagged_words2 = stanford_tagger.tag(a1.split())

#%%
remove_nonalpha_pattern2 = r"([a-zA-Z']*[^a-zA-Z' ]+[a-zA-Z']+|[a-zA-Z']+[^a-zA-Z' ]+[a-zA-Z']*|[^a-zA-Z' ]+)"
remove_nonalpha_chars = r"[^a-zA-Z' ]"
import re
re.sub(remove_nonalpha_pattern2, "", a1)
re.sub(remove_nonalpha_chars, "", "!!..,")    
    
    
#%%
fn1 = '/home/tuomas/Python/Gradu/data_processing/datasets/Amazon review data/Preprocess_common1/All_Beauty/All_Beauty-0_StanfordPOS.csv'
df1 = pd.read_csv(fn1)
fn2 = '/home/tuomas/Python/Gradu/data_processing/datasets/Amazon review data/Preprocess_common2/All_Beauty/All_Beauty-0.csv'
#fn2 = '/home/tuomas/Python/Gradu/data_processing/datasets/Amazon review data/Preprocess_common2/All_Beauty/All_Beauty-0sentence.csv'
df2 = pd.read_csv(fn2)
df2['SPOS_idx']=df2['SPOS_idx'].apply(eval)
i = 1111
words1 = word_tokenize(df1.loc[df1['doc_id']==i]['reviewText'].iloc[0])
words2 = word_tokenize(' '.join(df2.loc[df2['doc_id']==i]['reviewText'].iloc[0].split('.')))#word_tokenize(' '.join(df2.loc[df2['doc_id']==i]['reviewText']))
spos_idx = df2.loc[df2['doc_id']==i]['SPOS_idx'].iloc[0]

#%%
for i,w2 in zip(spos_idx, words2):
    print(f'{w2} : {words1[i]}')
    
    
#%%
test_list1 = [1, 2, 3]
test_list2 = [4,5,6]
test_list3 = [7,8,9]
test_list4 = [7,8,9]
  
# using list comprehension to concat
res_list = [y for x in [test_list1, test_list2, test_list3,test_list4] for y in x]

#%%
from nltk.tag.stanford import StanfordPOSTagger
path_to_model = '/home/tuomas/Java/lib/stanford-postagger-full-2020-11-17/models/english-bidirectional-distsim.tagger'
path_to_jar = '/home/tuomas/Java/lib/stanford-postagger-full-2020-11-17/stanford-postagger.jar'
stanford_tagger = StanfordPOSTagger(path_to_model, path_to_jar)

stanford_tagger.tag(word_tokenize(a1))

#%% Old code
def remove_nonalpha_chars3(df, pattern=None):
    df['reviewText'] = df['reviewText'].str.replace(pattern, '', regex=True)
    df['reviewText'] = df['reviewText'].str.split()
    df['reviewText'] = df['reviewText'].str.join(sep=' ')
    df['reviewText'] = df['reviewText'].str.lower()
    return df

def remove_nonalpha_chars2(df, pattern=None):
    for i in range(df.shape[0]):
        rt = df.iloc[i]['reviewText']
        rt_new = []
        SPOS_idx_new = []
        for sposidx, w in enumerate(word_tokenize(rt)):
            w_ = re.sub(remove_nonalpha_chars_pattern, "", w)
            if len(w_)>0:
                rt_new.append(w_)
                SPOS_idx_new.append(sposidx)
    
def remove_stopwords_old(text):
    text_token = word_tokenize(text)
    text_rmstop = [w for w in text_token if w not in stop]
    return text_rmstop

def parse_dataframe(df):
    df['reviewText'] = df['reviewText'].str.lower()
    df['reviewText'] = df['reviewText'].apply(remove_stopwords)
    #df['reviewText'] = df['reviewText'].apply(lemmatize_words)
    df['reviewText'] = df['reviewText'].str.join(sep=' ')
    return df
    
    
    
    
    
