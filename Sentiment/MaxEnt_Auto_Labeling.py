# Use pytorch_py3.8.8

# Create auto-labeled training data for Maximum Entropy classifier
# Output labeled training data in following form:
    # Let w_t be the target word which label (topic-word (0), opinion-word (1)) we are going to sort out
    # x = [ POS(w_{t-1}), POS(w_{t}), POS(w_{t+1}), w2vFeature(w_t)], y = 0 or 1

import logging
import pandas as pd
import glob
from nltk.tokenize import word_tokenize
from gensim.corpora import Dictionary
import numpy as np
from nltk.stem import PorterStemmer
import string
from itertools import groupby
import gensim
from nltk.tag.stanford import StanfordPOSTagger

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
def dataframe_generator(fn):
    df = pd.read_csv(fn)
    df['reviewText']=df['reviewText'].apply(eval)
    for _,row in df.iterrows():
        yield row['reviewText']

# Read the dataframe whose words are used to select opinion words 
# Create dictionary.
#dictnr = Dictionary(dataframe_generator(train_path+'train_StanfordPOS.csv'))
dictnr = Dictionary(dataframe_generator(train_path+'train.csv'))
# Document frequencies of words.
word_doc_freq = {dictnr[k]:v for k,v in zip(dictnr.dfs.keys(),dictnr.dfs.values())}
# Word total frequency
word_tot_freq = {dictnr[k]:v for k,v in zip(dictnr.cfs.keys(),dictnr.cfs.values())}

avg_df = np.mean(list(word_doc_freq.values()))

#%% 
# Extract opinion-word examples
# 1) Randomly select a set of M opinion words from a general opinion lexicon. 
#    selected words are required to have relatively large document frequency 
#    in the target corpus.
def get_opinion_word_examples(fn, M=None, min_df=0, stem_opinion_words=False):
    # stem_opinion_words = True if dictionary is builded on stemmed corpus
    opinion_word_examples =  []
    all_opinion_words = []
    with open(fn, 'r') as lexicon_file:
        for ow in lexicon_file:
            ow = ow.strip()
            #ow_setmmed = ps.stem(ow) if stem_opinion_words else ow
            df = word_doc_freq.get(ow, 0)
            if df>=min_df:
                opinion_word_examples.append(ow)
            if df>0:
                all_opinion_words.append(ow)
                
    opinion_word_examples = list(set(opinion_word_examples))
    
    if M!=None:
        opinion_word_examples = np.random.choice(opinion_word_examples, 2*M).tolist()
    
    # Return non-stemmed opinion words (stem separately if needed)
    return list(set(opinion_word_examples))[:M], all_opinion_words

opinion_lexicon_path = '/home/tuomas/Python/Gradu/data_processing/Opinion_lexicon/Bing_Liu/'
# These are used as target words w_t yielding opinion-lablel (1)
opinion_word_examples, all_opinion_words = get_opinion_word_examples(opinion_lexicon_path+'opinion_words.txt', M=100, 
                                                                     min_df=2*avg_df, stem_opinion_words=False)

#%% 
# Extract topic-word examples
# 2) Randomly choose a set of documents (or sentences) S such that each 
#    document in S contains at least one opinion word in "opinion_words"
# 3) Label word as an topic-word if it is not in "opinion_words" and appears
#    next to a known opinion-word w_o∈"opinion_words" in a document (or sentence)
#    contained in S

# Use Stanford POS data
def get_topic_word_examples(fn, opinion_words, N=None):
    topic_word_examples = []
    for words in dataframe_generator(fn):
        #prev = 
        for i, w in enumerate(words):
            if w in string.punctuation:
                continue
            w = w.lower()
            im = 1 if i>0 else 0
            ip = 1 if i<(len(words)-1) else 0            
            if (w not in opinion_words) and (words[i-im] in opinion_words or words[i+ip] in opinion_words):
                    topic_word_examples.append(w)
                    
    topic_word_examples = list(set(topic_word_examples))
    # Select randomly N topic words
    if N!=None:
        topic_word_examples = np.random.choice(topic_word_examples, 2*N).tolist()

    return list(set(topic_word_examples))[:N]

topic_word_examples = get_topic_word_examples(train_path+'train_StanfordPOS.csv', all_opinion_words, N=100)

#%% 


def get_topic_word_examples_tr(fn_train, fn_SPOS, opinion_words, N=None):
    df_train = pd.read_csv(fn_train)
    df_train['reviewText'] = df_train['reviewText'].apply(eval)
    df_train['SPOS_idx'] = df_train['SPOS_idx'].apply(eval)
    df_SPOS = pd.read_csv(fn_SPOS)
    df_SPOS['reviewText'] = df_SPOS['reviewText'].apply(eval)
    
    #opinion_words_stem = [ps.stem(ow) for ow in opinion_words]
    topic_word_examples = []
    for train_text, train_SPOS, SPOS_text in zip(df_train['reviewText'],df_train['SPOS_idx'],df_SPOS['reviewText']):
        si = 0
        #print(SPOS_text)
        for w in train_text:
            if w=='.':
                continue
            # Get neighbourding words in the original corpus
            im = 1 if train_SPOS[si]>0 else 0
            ip = 1 if train_SPOS[si]<(len(SPOS_text)-1) else 0
            #print(f'{train_SPOS[si]-im}-{train_SPOS[si]}-{train_SPOS[si]+ip}')
            w_SPOS_prev = SPOS_text[train_SPOS[si]-im]
            w_SPOS_next = SPOS_text[train_SPOS[si]+ip]
            if (w not in opinion_words) and (w_SPOS_prev in opinion_words or w_SPOS_next in opinion_words):
                if word_doc_freq.get(w)>=avg_df:
                    topic_word_examples.append(w)
            si+=1
        print('...')
        
    topic_word_examples = list(set(topic_word_examples))
    # Select randomly N topic words
    if N!=None:
        topic_word_examples = np.random.choice(topic_word_examples, 2*N).tolist()

    return list(set(topic_word_examples))[:N]

topic_word_examples2 = get_topic_word_examples_tr(train_path+'train.csv', train_path+'train_StanfordPOS.csv', 
                                                  all_opinion_words, N=100)

#%%
# Train word embeddings Part 1: Write dataframe into text file (SKIP)
with open(train_path+'w2v_train.txt', 'w') as w2v_train:
    for rt in dataframe_generator(train_path+'train.csv'):
        for sentence in [list(group) for k, group in groupby(rt, lambda x: x == ".") if not k]:
            sentence = ' '.join(sentence)
            w2v_train.write(sentence+'\n')

#%%
# Train word embeddings Part 1: Write dataframe into text file
with open(train_path+'w2v_train.txt', 'w') as w2v_train:
    for rt in dataframe_generator(train_path+'train.csv'):
        words = []
        for w in rt:
            if w=='.':
                continue
            words.append(w)
        words = ' '.join(words)
        w2v_train.write(words+'\n')
            
#%%
# Train word embeddings Part 2: Train word embeddings
w2v_args = {'emb_dim':200,
            'emb_window':5,
            'min_count':None,
            'neg_size':5,
            'epochs':10,
            'vocab_size':None}

model = gensim.models.Word2Vec(corpus_file = train_path+'w2v_train.txt', 
                               vector_size=w2v_args['emb_dim'], 
                               window=w2v_args['emb_window'],
                               min_count=1, 
                               workers=16, 
                               sg=1,
                               negative=w2v_args['neg_size'], 
                               max_vocab_size=w2v_args['vocab_size'],
                               epochs=w2v_args['epochs'])

model.save(train_path+'w2v_EAME.w2v')

#%% 
# Compute w2vFeature for each word w_t ∈ opinion_word_examples + topic_word_examples2

def w2vFeature(w_t, N):
    score = 0
    for w_n, cos in model.wv.most_similar(w_t, topn=N):
        sgn = 1 if w_n in all_opinion_words else -1
        score += sgn*cos
        
    return np.floor(score)
            
#%%
# Obtain the POS + w2vFeature representations each word w_t ∈ opinion_word_examples + topic_word_examples2

ws = []
features = []
labels = []

df = pd.read_csv(train_path+'train.csv')
df['reviewText']=df['reviewText'].apply(eval)
df['SPOS_idx']=df['SPOS_idx'].apply(eval)

df_POS = pd.read_csv(train_path+'train_POS.csv')
df_POS['POS_tag']=df_POS['POS_tag'].apply(eval)

cntr = 0
for _,row in df.iterrows():
    train_SPOS = row['SPOS_idx']
    did = row['doc_id']
    POS_tag = df_POS.loc[df_POS['doc_id']==did]['POS_tag'].iloc[0]
    si=0
    for w in row['reviewText']:
        label = None
        if w=='.':
            continue 
        if w in topic_word_examples2:
            label = 0
        elif w in opinion_word_examples:
            label = 1
        
        if label != None: # w is training example, obtain its features
            # Get neighbourding POS-tags in the original corpus
            POS_prev = ('!',None)
            if train_SPOS[si]>0:
                POS_prev = POS_tag[train_SPOS[si]-1]
            POS_curr = POS_tag[train_SPOS[si]]
            POS_next = ('!',None)
            if train_SPOS[si]<(len(POS_tag)-1):
                POS_next = POS_tag[train_SPOS[si]+1]
            # Calculate w2vFeature
            w2vf = w2vFeature(w, N=5)
            # Add feature representation of the word
            ws.append(w)
            features.append([POS_prev, POS_curr, POS_next, w2vf])
            labels.append(label)
        si += 1
        
    cntr += 1
    if cntr%1000==0:
        logger.info(f"{cntr}/{df.shape[0]}")

#%% 
# Transform features to numerical scale
penn_treebank = {'CC':1,'CD':2,'DT':3,'EX':4,'FW':5,'IN':6,'JJ':7,'JJR':8,'JJS':9,'LS':10,
                 'MD':11,'NN':12,'NNS':13,'NNP':14,'NNPS':15,'PDT':16,'POS':17,'PRP':18,'PRP$':19,'RB':20,
                 'RBR':21,'RBS':22,'RP':23,'SYM':24,'TO':25,'UH':26,'VB':27,'VBD':28,'VBG':29,'VBN':30,
                 'VBP':31,'VBZ':32,'WDT':33,'WP':34,'WP$':35,'WRB':36,None:37}
x = len(penn_treebank)+1

features_numeric = []
for f in features:
    features_numeric.append([penn_treebank.get(f[0][1], x), penn_treebank.get(f[1][1], x), penn_treebank.get(f[2][1], x), f[3]])
    
#%%
# Select a balanced subset of features & labels
size = 50 # size-opinion words and size-topic words
opinion_word_idx = np.where(np.array(labels)==1)[0]
opinion_word_idx = np.random.choice(opinion_word_idx,size,replace=False)
opinionw_ws = np.array(ws)[opinion_word_idx].tolist()
opinionw_features = np.array(features_numeric, dtype='int')[opinion_word_idx]
# Add label information
opinionw_features = np.hstack((opinionw_features, np.ones((opinionw_features.shape[0], 1), dtype=opinionw_features.dtype)))

topic_word_idx = np.where(np.array(labels)==0)[0]
topic_word_idx = np.random.choice(topic_word_idx,size,replace=False)
topicw_ws = np.array(ws)[topic_word_idx].tolist()
topicw_features = np.array(features_numeric, dtype='int')[topic_word_idx]
# Add label information
topicw_features = np.hstack((topicw_features, np.zeros((topicw_features.shape[0], 1), dtype=topicw_features.dtype)))

MaxEnt_training_features = np.vstack((opinionw_features,topicw_features))

#%%
# Save features and corresponding labels
np.save(train_path+'MaxEnt_training_features.npy', MaxEnt_training_features)

#%%
# Save all_opinion_words to file
with open(train_path+'all_opinion_words.txt', 'w') as f:
    for ow in all_opinion_words:
        f.write(ow+'\n')

#%%
#%%
path_to_model = '/home/tuomas/Java/lib/stanford-postagger-full-2020-11-17/models/english-bidirectional-distsim.tagger'
path_to_jar = '/home/tuomas/Java/lib/stanford-postagger-full-2020-11-17/stanford-postagger.jar'
stanford_tagger = StanfordPOSTagger(path_to_model, path_to_jar)

df = pd.read_csv(train_path+'train.csv')
df['reviewText']=df['reviewText'].apply(eval)
df['SPOS_idx']=df['SPOS_idx'].apply(eval)

df_SPOS = pd.read_csv(train_path+'train_StanfordPOS.csv')
df_SPOS['reviewText']=df_SPOS['reviewText'].apply(eval)
df_SPOS['SPOS_idx']=df_SPOS['SPOS_idx'].apply(eval)

cnt = 0
for _,row in df.iterrows():
    train_SPOS = row['SPOS_idx']
    did = row['doc_id']
    SPOS_rt = df_SPOS.loc[df_SPOS['doc_id']==did]['reviewText'].iloc[0]
    SPOS = stanford_tagger.tag(SPOS_rt)
    si=0
    for w in row['reviewText']:
        if w=='.':
            continue 
        #print(f'{train_SPOS[si]-im}-{train_SPOS[si]}-{train_SPOS[si]+ip}')
        POS_prev = None
        if train_SPOS[si]>0:
            POS_prev = SPOS[train_SPOS[si]-1]
        POS_curr = SPOS[train_SPOS[si]]
        POS_next = None
        if train_SPOS[si]<(len(SPOS_rt)-1):
            POS_next = SPOS[train_SPOS[si]+1]
            
        print(f'In training data: {w}')
        print(f'Corresponding POS-tags: {POS_prev}, {POS_curr}, {POS_next}')
        print()
        si += 1
    cnt += 1
    print('######################## New document #############################')
    if cnt==4:
        break
            
#%%
df = pd.read_csv(train_path+'train.csv')
df['reviewText']=df['reviewText'].apply(eval)
df['SPOS_idx']=df['SPOS_idx'].apply(eval)

df_SPOS = pd.read_csv(train_path+'train_StanfordPOS.csv')
df_SPOS['reviewText']=df_SPOS['reviewText'].apply(eval)
df_SPOS['SPOS_idx']=df_SPOS['SPOS_idx'].apply(eval)

cntr = 0
for _,row in df.iterrows():
    train_SPOS = row['SPOS_idx']
    did = row['doc_id']
    SPOS_rt = df_SPOS.loc[df_SPOS['doc_id']==did]['reviewText'].iloc[0]
    SPOS = stanford_tagger.tag(SPOS_rt)
    #for w in row['reviewText']:
     #   pass
    cntr+=1
    if cntr%10==0:
        #print(f"{cntr}/{df.shape[0]}")
        logger.info(f"{cntr}/{df.shape[0]}")
            
#%%

df_SPOS = pd.read_csv(train_path+'train_StanfordPOS.csv')
df_SPOS['reviewText']=df_SPOS['reviewText'].apply(eval)
df_SPOS['SPOS_idx']=df_SPOS['SPOS_idx'].apply(eval)

cntr = 0
for _,row in df_SPOS.iterrows():
    SPOS = stanford_tagger.tag(row['reviewText'])
    cntr+=1
    if cntr%10==0:
        #print(f"{cntr}/{df.shape[0]}")
        logger.info(f"{cntr}/{df.shape[0]}")
    if cntr==50:
        break
    
#%%
df_SPOS = pd.read_csv(train_path+'train_StanfordPOS.csv')
df_SPOS['reviewText']=df_SPOS['reviewText'].apply(eval)
df_SPOS['SPOS_idx']=df_SPOS['SPOS_idx'].apply(eval)

docs = df_SPOS['reviewText'][:10]
SPOS = stanford_tagger.tag_sents(docs)
            
            
            
            
            
            
            
            
            
            
            


