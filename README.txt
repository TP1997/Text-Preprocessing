BHM = Bayesian Hierarchial Model

New execution order:

    1. json2csv.py                          (Original -> Original)
    2. preprocess_common1.py                (Original -> Preprocessed_common1)
    3. preprocess_common2.py                (Preprocessed_common1 -> Preprocessed_common2)
    4. create_training_dataframe.py         (Preprocessed_common2 -> Training_data)
    5. preprocess_training_dataframe_NEW.py (Training_data -> Training_data)






###################### ###################### ###################### ###################### ######################
New execution order:

Common for every method:
    1. json2csv.py                  (Original -> Original)
    2. preprocess_common1.py        (Original -> Preprocessed_common)
    3. preprocess_common2.py        (Preprocessed_common -> Preprocessed_common)
    2. preprocess_common.py         (Original -> Preprocessed_common)
    optional. (group_by_year.py)
    3. csv2txt.py                   (Preprocessed_common -> Training_data)
    4. preprocess_trainfile_NEW.py  (Training_data -> Training_data)

Specific for BHM-models:
    5. to_LDA_format.py             (Training_data -> Training_data)


Specific for cross-collection BHM-models:
    5. to_LDA_cc_format.py          (Training_data -> Training_data/Cross_collection)

Specific for JAE-ABAE:
5. TFIDF_document_vectors.py



