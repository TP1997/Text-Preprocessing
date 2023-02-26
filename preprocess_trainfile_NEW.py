# Use pytorch_py3.8.8
# Create bigrams and save them to file.
# Processing operations include:
    # See prune_rules
    # Create word embeddings
    # Generate conf.json
    
# Before run specify following variables:
    # data_name
    # data_dir
    # prune_rules
    # train_size
    # w2v_args

import os
import json
import logging
import glob
import codecs
import numpy as np
import gensim
from gensim.corpora import Dictionary

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
data_dir = '20k'
path = f'/home/tuomas/Python/Gradu/data_processing/datasets/Amazon review data/Training_data/{data_name}/{data_dir}/'
filename = 'train_raw.txt'

#%% Define pruning rules.
prune_rules = {'minlen_word':(True, 2),         # Remove words shorter than "minlen_word" characters.X
               'mindf_word': (True, 10),        # Remove words having document frequency smaller than "mindf_word".X
               'mincount_word': (True,15),     # Remove words occurred less than "mincount_word" times.
               'least_freq_quant': (False, 0.99), # Remove least frequent % of words.X !!! ZIP'F LAW !!!
               'most_freq_quant': (False, 0.01), # Remove most frequent % of words. !!! KEEP THIS FALSE !!!
               'minlen_doc': (True, 4),         # Remove documents shorter than "minlen_doc"X
               'minlen_sentence': (False, 4)     # Remove sentences shorter than "minlen_sentence"X
               }
sentence_separator = '.'
train_size = 20000

#%% Create bigrams and save them to file
# Identify bigram procedures.

def file_generator(fn):
    with open(fn, 'r') as f:
        for l in f:
            yield l.split()

def build_bigram_from_file(fn):
    filegen = file_generator(fn)
    bigram = gensim.models.Phrases(filegen, min_count=prune_rules['mincount_word'][1])#, threshold=1)
    bigram_model = gensim.models.phrases.Phraser(bigram)

    return bigram_model

def bigram_to_file(fn_r, fn_w):
    bigram_model = build_bigram_from_file(fn_r)
    
    with open(fn_r, 'r') as infile, open(fn_w, 'w') as outfile:
        for l in infile:
            newline = bigram_model[l.split()]
            newline = ' '.join(newline)
            outfile.write(newline+'\n')

bigram_to_file(path+filename, path+'train_bg.txt')

#%% Prune words following predefined rules.

def prune_words(fn_r, fn_w):
    # Create dictionary.
    dictnr = Dictionary(file_generator(fn_r))
    # Document frequencies of words.
    word_doc_freq = {dictnr[k]:v for k,v in zip(dictnr.dfs.keys(),dictnr.dfs.values())}
    # Word total frequency
    word_tot_freq = {dictnr[k]:v for k,v in zip(dictnr.cfs.keys(),dictnr.cfs.values())}
    lb = np.quantile(np.array(list(word_tot_freq.values())), prune_rules['least_freq_quant'][1])
    ub = np.quantile(np.array(list(word_tot_freq.values())), 1-prune_rules['most_freq_quant'][1])
    
    size = 0
    maxlen = 0
    avglen = 0
    with open(fn_r, 'r') as infile, open(fn_w, 'w') as outfile:
        for l in infile:
            words = []
            sentence_symbols = 0
            for w in l.split():
                prune = False
                if w==sentence_separator:
                    #w = ' . '
                    words.append(w)
                    sentence_symbols += 1
                    prune = False
                    continue
                if prune_rules['minlen_word'][0] and (len(w) < prune_rules['minlen_word'][1]):
                    prune = True
                if prune_rules['mindf_word'][0] and (word_doc_freq[w] < prune_rules['mindf_word'][1]):
                    prune = True
                if prune_rules['mincount_word'][0] and (word_tot_freq[w] < prune_rules['mincount_word'][1]):
                    prune = True
                if prune_rules['least_freq_quant'][0] and (word_tot_freq[w] < lb):
                    prune = True
                if prune_rules['most_freq_quant'][0] and (word_tot_freq[w] > ub):
                    prune = True
                    
                if not prune:
                    words.append(w)
                    
            # Remove documents with less than minlen_doc words.
            if prune_rules['minlen_doc'][0] and (len(words)-sentence_symbols) >= prune_rules['minlen_doc'][1]:
                maxlen = max(maxlen, (len(words)-sentence_symbols))
                avglen += (len(words)-sentence_symbols)
                newline = ' '.join(words)
                for sentence in newline.split(sentence_separator):
                    if sentence != sentence_separator and len(sentence) >= 2:
                        outfile.write(sentence.strip()+'\n')
                #outfile.write(newline+'\n')
                outfile.write('\n')
                size += 1
                
            if size >= train_size:
                break
            
    return size, maxlen, avglen/size, lb

size, maxlen, avglen, lb = prune_words(path+'train_raw.txt', path+'train.txt')

#%% Create word embeddings
class DocumentGenerator():
    def __init__(self, filename):
        self.filename = filename
    
    def __iter__(self):
        document = ''
        for line in codecs.open(self.filename, "r", encoding="utf-8"):
            if line!='\n':
                document += ' '+line
            if line=='\n':
                yield document.strip().split()
                document = ''
            
def create_word2vec(path, args):
    dg = DocumentGenerator(path)
    model = gensim.models.Word2Vec(corpus_file = path, 
                                   vector_size=args['emb_dim'], 
                                   window=args['emb_window'],
                                   min_count=1, 
                                   workers=16, 
                                   sg=1,
                                   negative=args['neg_size'], 
                                   max_vocab_size=args['vocab_size'])
    vocab_size = len(model.wv)
    model.save(path[:-3]+'w2v')
    return vocab_size

    
w2v_args = {'emb_dim':200,
            'emb_window':5,
            'min_count':None,
            'neg_size':5,
            'vocab_size':None}

vocab_size = create_word2vec(path+'train.txt', w2v_args)


#%% Generate data_conf.json

conf_json = {'train_file_text': path+'train.txt',
             'wv_file': path+'train.w2v',
             'size': size,
             'maxlen': maxlen,
             'avglen': avglen,
             'vocab_size':vocab_size}

with open(path+'data_conf.json', 'w', encoding='utf-8') as f:
    json.dump(conf_json, f, indent=4)

#%% Save pruning rules for further inspection.

with open(path+'prune_rules.json', 'w', encoding='utf-8') as f:
    json.dump(prune_rules, f, indent=4)


#%%
l='first sentence .  .  . ...      .     fourth sentence'
words = []
sentence_symbols = 0
for w in l.split():
    words.append(w)
    
newline = ' '.join(words)
for sentence in newline.split('.'):
    if sentence != '.' and len(sentence) >= 2:
        print(sentence.strip())




















