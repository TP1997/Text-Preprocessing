# Use pytorch_py3.8.8
# Modify the preprocessed trainfile to cross-collection LDA format:
    # Build & save combined vocabulary. 
    # Replace words in documents with vocabulary indices.
    # Create data_conf.json file
    
# Before run specify following variables:
    # data_names
    # data_dirs

import os
import json
from gensim.corpora import Dictionary
import shutil

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

data_names = ['All_Beauty','Luxury_Beauty']
data_dirs = ['20k','20k']

paths = [f'/home/tuomas/Python/Gradu/data_processing/datasets/Amazon review data/Training_data/{dn}/{dd}/' for dn,dd in zip(data_names,data_dirs)]

sd1 = '&'.join(data_names)
sd2 = '&'.join(data_dirs)
savepath = f'/home/tuomas/Python/Gradu/data_processing/datasets/Amazon review data/Training_data/Cross_collection/{sd1}/{sd2}/'
try:
    os.makedirs(savepath)
except FileExistsError:
    pass

data_confs = []
for p in paths:
    with open(p + 'data_conf.json', 'r') as f:
        dc = json.load(f)
        data_confs.append(dc)
        
#%%
def file_generator(paths):
    for p in paths:
        with open(p, 'r') as f:
            for l in f:
                yield l.split()

def transform_texts(data_confs, savepath, data_names, dictnr):
    num_words = []
    for i in range(len(data_confs)):
        with open(savepath+f'{data_names[i]}_train_vidx.txt', 'w') as outfile:
            nw = 0
            for doc in file_generator([data_confs[i]['train_file_text']]):
                doc_vidx = ' '.join([str(j) for j in dictnr.doc2idx(doc)])
                outfile.write(doc_vidx+'\n')
                nw += len(doc)
            num_words.append(nw)
            
    return num_words
            
# Create combined vocabulary
dictnr = Dictionary(file_generator([dc['train_file_text'] for dc in data_confs]))

# Replace words in documents with vocabulary indices
num_words = transform_texts(data_confs, savepath, data_names, dictnr)

#%% Save vocabulary
with open(savepath+'token2idx.json', 'w', encoding='utf-8') as f:
    json.dump(dictnr.token2id, f, indent=4)
    
with open(savepath+'idx2token.json', 'w', encoding='utf-8') as f:
    print(dictnr[0])
    json.dump(dictnr.id2token, f, indent=4)
  
#%% Copy original training files into savepath
for i in range(len(data_confs)):
    shutil.copyfile(data_confs[i]['train_file_text'], savepath+f'{data_names[i]}_train.txt')
    
#%% Create data_conf
data_conf_cc = {}
for i in range(len(data_confs)):
    data_conf_cc[f'{data_names[i]}_train_file_text'] = savepath+f'{data_names[i]}_train.txt'
    data_conf_cc[f'{data_names[i]}_train_file_vidx'] = savepath+f'{data_names[i]}_train_vidx.txt'
    data_conf_cc[f'{data_names[i]}_size'] = str(data_confs[i]['size'])
    data_conf_cc[f'{data_names[i]}_num_words'] = str(num_words[i])
    
data_conf_cc['num_words_total'] = str(dictnr.num_pos)
data_conf_cc['token2idx_file'] = savepath+'token2idx.json'
data_conf_cc['idx2token_file'] = savepath+'idx2token.json'
data_conf_cc['vocab_size'] = str(len(dictnr.token2id))

with open(savepath+'data_conf.json', 'w', encoding='utf-8') as f:
    json.dump(data_conf_cc, f, indent=4)











