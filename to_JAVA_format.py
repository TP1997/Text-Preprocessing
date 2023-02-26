# Use pytorch_py3.8.8
# Write train.csv and train_word_probabilities.csv to text files that JAVA program can read

import logging
import pandas as pd
from gensim.corpora import Dictionary

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
train_path = f'/home/tuomas/Python/Gradu/data_processing/datasets/Amazon review data/Training_data/{data_name}/{data_dir}/'

#%%
# Write training dataframe to text file
def dataframe_generator(df):
    for _,row in df.iterrows():
        yield row['reviewText']
        
df = pd.read_csv(train_path+'train.csv')
df['reviewText']=df['reviewText'].apply(eval)
df['SPOS_idx']=df['SPOS_idx'].apply(eval)

df_proba = pd.read_csv(train_path+'train_word_probabilities.csv')
df_proba['word_prob']=df_proba['word_prob'].apply(eval)

# Create dictionary.
dictnr = Dictionary(dataframe_generator(df))

# Write training files
separate_sentences = True
with open(train_path+'train_vidx.txt', 'w') as outfile_vidx, open(train_path+'train_text.txt', 'w') as outfile_text, open(train_path+'train_proba.txt', 'w') as outfile_proba:
    for _,row in df.iterrows():
        did = row['doc_id']
        word_proba = df_proba.loc[df_proba['doc_id']==did]['word_prob'].iloc[0]
        i=0
        for w in row['reviewText']:
            if w=='.':
                if separate_sentences:
                    outfile_text.write('\n')
                    outfile_vidx.write('\n')
                    outfile_proba.write('\n')
                continue
            outfile_text.write(w+' ')
            outfile_vidx.write(str(dictnr.token2id[w])+' ')
            outfile_proba.write(str(word_proba[i])+' ')
            i+=1
        outfile_text.write('\n\n')
        outfile_vidx.write('\n\n')
        outfile_proba.write('\n\n')
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

