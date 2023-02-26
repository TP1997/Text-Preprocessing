# Use pytorch_py3.8.8
# Modify the preprocessed trainfile to LDA format:
    # Replace words in documents with vocabulary indices.
    # Build & save vocabulary.
    
# Before run specify following variables:
    # data_name
    # data_dir
    
import os
import json
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
filename = 'train.txt'

data_conf = None
with open(path + 'data_conf.json', 'r') as f:
    data_conf = json.load(f)
    
#%% Transform documents: words -> vocabulary indices
def file_generator(fn):
    with open(fn, 'r') as f:
        for l in f:
            yield l.split()
            
dictnr = Dictionary(file_generator(path+filename))

with open(path+'train_vidx.txt', 'w') as outfile:
    for doc in file_generator(path+filename):
        if doc!='\n':
            doc_vidx = ' '.join([str(i) for i in dictnr.doc2idx(doc)])
            outfile.write(doc_vidx+'\n')
        else:
            outfile.write('\n')

#%% Optional: Save sentences as sequence of (w:count(w)), where count(w) is the count of word w in sentence

with open(path+'train_vidx_counts.txt', 'w') as outfile:
    for doc in file_generator(path+filename):
        if doc!='\n':
            doc_vidx =[str(i) for i in dictnr.doc2idx(doc)]
            doc_vidx_count = ''
            for vidx in doc_vidx:
                count = doc_vidx.count(vidx)
                count = str(vidx)+':'+str(count)
                doc_vidx_count += count+' '
            outfile.write(doc_vidx_count.strip()+'\n')
        else:
            outfile.write('\n')


#%% Save vocabulary
with open(path+'token2idx.json', 'w', encoding='utf-8') as f:
    json.dump(dictnr.token2id, f, indent=4)
    
with open(path+'idx2token.json', 'w', encoding='utf-8') as f:
    print(dictnr[0])
    json.dump(dictnr.id2token, f, indent=4)
    
#%% Update data_conf
data_conf['num_words'] = dictnr.num_pos
data_conf['train_file_vidx'] = path+'train_vidx.txt'
data_conf['token2idx_file'] = path+'token2idx.json'
data_conf['idx2token_file'] = path+'idx2token.json'
data_conf['train_file_vidx_count'] = path+'train_vidx_counts.txt'

with open(path+'data_conf.json', 'w', encoding='utf-8') as f:
    json.dump(data_conf, f, indent=4)





