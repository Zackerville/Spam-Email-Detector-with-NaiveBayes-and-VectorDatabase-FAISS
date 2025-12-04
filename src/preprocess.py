import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import numpy as np

stopwords = set(stopwords.words('english'))
stemmer = PorterStemmer()
punct_table = str.maketrans("", "", string.punctuation)

def preprocess_text(text):
    if not isinstance(text, str): text = str(text)
    text = text.lower()
    text = text.translate(punct_table)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha()]
    tokens = [t for t in tokens if t not in stopwords]
    tokens = [stemmer.stem(t) for t in tokens]
    return tokens
    
def create_vocab(message):
    vocab = []
    for tokens in message:
        for token in tokens:
            if token not in vocab:
                vocab.append(token)
    
    return vocab


def compute_idf(message, vocab):
    n = len(message)
    v = len(vocab)
    token_to_idx = {token: i for i, token in enumerate(vocab)} 
    df = np.zeros(v, dtype=np.float32)
    for tokens in message:
        word_set = set()
        for token in tokens:
            idx = token_to_idx.get(token)
            if idx is not None and idx not in word_set:
                df[idx] += 1.0
                word_set.add(idx)
    idf = np.log((n + 1.0) / (df + 1.0)) + 1.0
    return idf


def compute_tfidf(message, vocab, idf):
    v = len(vocab)
    token_to_idx = {token: i for i, token in enumerate(vocab)}
    x = np.zeros((len(message), v), dtype=np.float32)
    for i, tokens in enumerate(message):
        tf = np.zeros(v, dtype=np.float32)
        for token in tokens:
            idx = token_to_idx.get(token)
            if idx is not None:
                tf[idx] += 1.0
        s = tf.sum()
        if s > 0: tf /= s
        x[i] = tf * idf
    
    return x
    