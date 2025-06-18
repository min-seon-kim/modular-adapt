import re
import pandas as pd
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from strategies import update_model_by_strategy
from models import OnlineOODDetector

def regex_(text):
    pattern = '(http|ftp|https)://(?:[-\w.]|(?:%[\da-fA-F]{2}))+/(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
    text = re.sub(pattern, '', text)
    pattern = '(http|ftp|https)://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
    text = re.sub(pattern, '', text)
    pattern = '(http|ftp|https):// (?:[-\w.]|(?:%[\da-fA-F]{2}))+'
    text = re.sub(pattern, '', text)
    only_english = re.sub('[^ a-zA-Z]', '', text)
    only_english = only_english.lower()

    if bool(only_english and only_english.strip()) and len(only_english) >= 10:
        return only_english
    return False


def compare_drop(top_data, last_data):
    top_n = len(top_data)
    last_n = len(last_data)
    range_ = top_n - last_n
    if range_ < 0:
        last_data = last_data.sample(n=(last_n + range_), random_state=1)
    else:
        top_data = top_data.sample(n=(top_n - range_), random_state=1)
    return top_data, last_data


def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma


def get_tokens(sentence):
    tknzr = TweetTokenizer()
    tokens = tknzr.tokenize(sentence)
    stop_words = set(stopwords.words("english"))

    tokens = [token for token in tokens if (token not in stop_words and len(token) > 1)]
    tokens = [get_lemma(token) for token in tokens]
    return (tokens)


def calculate_n_similarity(base_model, new_model, represent_set):
    from numpy import dot, array
    from gensim import matutils

    ws=[]
    for word in represent_set:
        if word in new_model:
            ws.append(word)

    v1 = [base_model[word] for word in ws]
    v2 = [new_model[word] for word in ws]

    similarity = dot(matutils.unitvec(array(v1).mean(axis=0)), matutils.unitvec(array(v2).mean(axis=0)))
    return similarity


def train_w2v_model(filtered_word):
    from gensim.test.utils import datapath
    from gensim.test.utils import get_tmpfile, common_dictionary, common_corpus
    from gensim.models import KeyedVectors
    from gensim.scripts.glove2word2vec import glove2word2vec
    from gensim.models import Word2Vec

    base_model = Word2Vec(size=100, min_count=3)
    base_model.build_vocab(filtered_word)

    base_model.train(filtered_word, total_examples=len(filtered_word), epochs=5)
    base_model_wv = base_model.wv

    base_model.save("./w2v/word2vec.model")
    base_model_wv.save("./w2v/word2vec.wordvectors")
    return base_model, base_model_wv


def update_stat(model, random_batches, tokenizer, base_model_wv, config):
    from sklearn.neighbors import KernelDensity
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    from nltk.corpus import stopwords
    from copy import deepcopy

    X = []
    Y = []
    indicator_list = []
    origin_keyword = deepcopy(list(tokenizer.word_index.keys()))
    for batch_df in random_batches:
        stop = stopwords.words('english')
        batch_df['text'] = batch_df['text'].apply(lambda x:' '.join(x.lower() for x in x.split()))
        batch_df['text']= batch_df['text'].str.replace('[^\w\s]','')
        batch_df['text']= batch_df['text'].apply(lambda x: ' '.join(x for x in x.split() if  not x.isdigit()))
        batch_df['text'] = batch_df['text'].apply(lambda x:' '.join(x for x in x.split() if not x in stop))
        batch_df['text'] = batch_df['text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
        batch_df['text'] = batch_df['text'].map(lambda x: simple_preprocess(x.lower(),deacc=True, max_len=100))
        
        previous_word_counts = deepcopy(tokenizer.word_counts)
        previous_word_docs = deepcopy(tokenizer.word_docs)

        from custom_tokenizer import Tokenizer as CTokenizer
        custom_tokenizer = CTokenizer(num_words=1000, previous_word_counts=previous_word_counts, \
                                previous_word_docs=previous_word_docs, vocabulary=[]) 
        custom_tokenizer.fit_on_texts(evolving_event['text'][:config.event_size/2])

        _, new_model_wv = train_w2v_model(evolving_event['text'][:config.event_size/2])
        
        word_dict = sorted(tokenizer.word_counts.items(), key=lambda x: x[1], reverse=True)
        keywordset = [word_dict[i][0] for i in range(config.keyword_size)]

        cnt_fre = [sum(evolving_event['text'][:config.event_size/2].apply(lambda x: x.count(i))) for i in keywordset]
        FREQUENCY_INDICATOR = np.mean(cnt_fre)
        SEMANTIC_INDICATOR = calculate_n_similarity(base_model=base_model_wv, new_model=new_model_wv, represent_set=represent_set)
        vocab_cnt = 0
        for word, idx in custom_tokenizer.word_index.items():
            if idx == config.embedding_size:
                break
            if word not in origin_keyword:
                vocab_cnt += 1
        VOCABULARY_INDICATOR = (config.embedding_size-vocab_cnt)/config.embedding_size
        
        indicators = np.array([[SEMANTIC_INDICATOR, VOCABULARY_INDICATOR, FREQUENCY_INDICATOR]])
        X.append(indicators.flatten())
        indicator_list.append(indicators)

        accuracy_results = []
        for strategy in range(1, 6):
            _, _, _, _, score = update_model_by_strategy(
                model=deepcopy(initial_model),
                custom_tokenizer=deepcopy(custom_tokenizer),
                evolving_event=batch_df,
                base_model_wv=deepcopy(base_model_wv),
                new_model_wv=new_model_wv,
                chosen_strategy=strategy,
                config=config
            )
            accuracy_results.append(score[1])
        Y.append(accuracy_results)

        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error

        X = np.array(X)
        Y = np.array(Y)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        ml_model = RandomForestRegressor(n_estimators=100, random_state=42)
        ml_model.fit(X_train, Y_train)

    indicator_init_buffer = np.vstack(indicator_list)
    detector = OnlineOODDetector(alpha=0.05, tau=3.0, epsilon=1e-6)
    detector.initialize(indicator_init_buffer)
    return ml_model, detector