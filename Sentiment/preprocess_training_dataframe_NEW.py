# Use pytorch_py3.8.8

# Processing operations include:
    # See prune_rules
    # Generate conf.json
    
# Before run specify following variables:
    # data_name
    # data_dir
    # prune_rules
    # train_size
    # w2v_args
    
import logging
import pandas as pd
from gensim.corpora import Dictionary
import numpy as np
import json
from nltk.tokenize import word_tokenize
import glob

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
filename = 'train_raw.csv'

#%% Define pruning rules.
prune_rules = {'minlen_word':(True, 2),             # Remove words shorter than "minlen_word" characters.X
               'mindf_word': (True, 5),            # Remove words having document frequency smaller than "mindf_word".X
               'mincount_word': (True,10),          # Remove words occurred less than "mincount_word" times.
               'least_freq_quant': (False, 0.99),   # Remove least frequent % of words.X !!! ZIP'F LAW !!!
               'most_freq_quant': (False, 0.01),    # Remove most frequent % of words. !!! KEEP THIS FALSE !!!
               'minlen_doc': (True, 4),             # Remove documents shorter than "minlen_doc"X
               'minlen_sentence': (False, 4)        # Remove sentences shorter than "minlen_sentence"X
               }
sentence_separator = '.'
train_size = 25000

#%% Prune words following predefined rules.

def file_generator(fn):
    with open(fn, 'r') as f:
        for l in f:
            yield l.split()
            
def dataframe_generator_x(fn):
    df = pd.read_csv(fn)[:train_size]
    df['reviewText']=df['reviewText'].apply(eval)
    for _,row in df.iterrows():
        yield row['reviewText']
        
def dataframe_generator(df):
    for _,row in df.iterrows():
        yield row['reviewText']

df = pd.read_csv(path+filename)[:train_size]
df['SPOS_idx']=df['SPOS_idx'].apply(eval)
df['reviewText']=df['reviewText'].apply(eval)

# Create dictionary.
dictnr = Dictionary(dataframe_generator(df))
# Document frequencies of words.
word_doc_freq = {dictnr[k]:v for k,v in zip(dictnr.dfs.keys(),dictnr.dfs.values())}
# Word total frequency
word_tot_freq = {dictnr[k]:v for k,v in zip(dictnr.cfs.keys(),dictnr.cfs.values())}
lb = np.quantile(np.array(list(word_tot_freq.values())), prune_rules['least_freq_quant'][1])
ub = np.quantile(np.array(list(word_tot_freq.values())), 1-prune_rules['most_freq_quant'][1])

df_new = pd.DataFrame(columns=df.columns)
maxlen = 0
avglen = 0
size = 0
for _,row in df.iterrows():
    words_new = ['!']
    SPOS_idx_new = []
    sentence_symbols = 0
    did = row['doc_id']
    i = 0
    for w in row['reviewText']: #zip(row['reviewText'],row['SPOS_idx']):#zip(word_tokenize(row['reviewText']),row['SPOS_idx']):
        spos_idx = row['SPOS_idx'][i]
        prune = False
        if (w==sentence_separator) or (w==sentence_separator and words_new[-1]!=sentence_separator): # Do not add trailing .'s e.g., "w_1 . . w_2"
            #w = ' . '
            words_new.append(w)
            sentence_symbols += 1
            prune = False
            continue
        if prune_rules['minlen_word'][0] and (len(w) < prune_rules['minlen_word'][1]): # Remove word if
            prune = True
        if prune_rules['mindf_word'][0] and (word_doc_freq[w] < prune_rules['mindf_word'][1]): # Remove word if
            #print('x')
            prune = True
        if prune_rules['mincount_word'][0] and (word_tot_freq[w] < prune_rules['mincount_word'][1]): # Remove word if
            #print('x')
            prune = True
        if prune_rules['least_freq_quant'][0] and (word_tot_freq[w] < lb): # Remove word if
            prune = True
        if prune_rules['most_freq_quant'][0] and (word_tot_freq[w] > ub): # Remove word if
            prune = True

        if not prune:
            words_new.append(w)
            SPOS_idx_new.append(spos_idx)
        i+=1
            
    # Remove trailing .'s
    if words_new[-1]==sentence_separator:
        words_new = words_new[:-1]
        sentence_symbols -= 1
    
    # Remove documents with less than minlen_doc words, otherwise add document to dataframe.
    if (not prune_rules['minlen_doc'][0]) or (prune_rules['minlen_doc'][0] and (len(words_new)-sentence_symbols-1) >= prune_rules['minlen_doc'][1]):
        maxlen = max(maxlen, (len(words_new)-sentence_symbols-1))
        avglen += (len(words_new)-sentence_symbols-1)
        # Add processed document
        newrow = row[:3]
        newrow['reviewText'] = words_new[1:] #' '.join(words_new[1:])
        newrow['SPOS_idx'] = SPOS_idx_new
        df_new = df_new.append(newrow, ignore_index=True)
        size += 1
        
    if size%1000==0:
        logger.info(f"{size}/{train_size}")

    if size >= train_size:
        pass#break
        
fn = path + 'train.csv'
df_new.to_csv(fn, index=False)
        
#%% Generate data_conf.json

# Create dictionary to fetch the vocabulary size
dictnr = Dictionary(dataframe_generator(df_new))

conf_json = {'train_file_csv': path+'train.csv',
             'size': size,
             'maxlen': maxlen,
             'avglen': avglen/size,
             'vocab_size':len(dictnr)}

with open(path+'data_conf.json', 'w', encoding='utf-8') as f:
    json.dump(conf_json, f, indent=4)


#%% Save pruning rules for further inspection.

with open(path+'prune_rules.json', 'w', encoding='utf-8') as f:
    json.dump(prune_rules, f, indent=4)

#%% Fetch documents in *StanfordPOS corresponding to documents in training file

dn = data_name.strip('_sentence')
stanfordPOS_path = f'/home/tuomas/Python/Gradu/data_processing/datasets/Amazon review data/Preprocess_common1/All_Beauty/{dn}'
filenames = glob.glob(stanfordPOS_path + '*_StanfordPOS.csv')
filenames.sort()
filename=filenames[0]
df_SPOS = pd.read_csv(filename)

# Get those documents occurring in final train.csv
train_doc_ids = df_new['doc_id']
df_SPOS = df_SPOS.loc[df_SPOS['doc_id'].isin(train_doc_ids)]
df_SPOS['reviewText'] = df_SPOS['reviewText'].apply(word_tokenize)
df_SPOS.to_csv(path + 'train_StanfordPOS.csv', index=False)

#%%

#%%
dictnr = Dictionary(dataframe_generator(df_new))
# Document frequencies of words.
word_doc_freq = {dictnr[k]:v for k,v in zip(dictnr.dfs.keys(),dictnr.dfs.values())}
print(np.min(list(word_doc_freq.values())))
# Word total frequency
word_tot_freq = {dictnr[k]:v for k,v in zip(dictnr.cfs.keys(),dictnr.cfs.values())}
print(np.min(list(word_tot_freq.values())))


#%%
dfpos = pd.read_csv('/home/tuomas/Python/Gradu/data_processing/datasets/Amazon review data/Training_data/All_Beauty_sentence/20k/train_StanfordPOS.csv')
dfpos['SPOS_idx'] = dfpos['SPOS_idx'].apply(eval)
dfpos['reviewText'] = dfpos['reviewText'].apply(eval)
#dftrain = pd.read_csv('/home/tuomas/Python/Gradu/data_processing/datasets/Amazon review data/Training_data/All_Beauty_sentence/20k/train_raw.csv')
dftrain = pd.read_csv('/home/tuomas/Python/Gradu/data_processing/datasets/Amazon review data/Training_data/All_Beauty_sentence/20k/train.csv')
dftrain['SPOS_idx'] = dftrain['SPOS_idx'].apply(eval)
dftrain['reviewText'] = dftrain['reviewText'].apply(eval)
#%%
i = 153
wordspos = dfpos.loc[dfpos['doc_id']==i]['reviewText'].iloc[0]
wordstrain = dftrain.loc[dftrain['doc_id']==i]['reviewText'].iloc[0]
spos_idx = dftrain.loc[dftrain['doc_id']==i]['SPOS_idx'].iloc[0]

#%%
i=0
for w in wordstrain:
    if w=='.':
        continue
    print(f'{w} : {wordspos[spos_idx[i]]}')
    i+=1



