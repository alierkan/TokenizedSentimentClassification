from gensim.models.keyedvectors import KeyedVectors
import numpy as np
import csv
import os
from gensim.utils import tokenize as gensimTokenizer
import random


class MyReviews(object):
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        print(self.filename)
        for line in open(self.filename, encoding="utf8"):
            # yield tokenizer.tokenize(line.lower())
            # yield tokenizer.tokenize(text_preprocessing(line.lower()))
            yield list(gensimTokenizer(line.lower()))


def get_stopwords():
    wl = []
    with open('../../resources/stopwords.csv') as csvfile:
        stopwords = csv.reader(csvfile, delimiter=',')
        for words in stopwords:
            for w in words:
                wl.append(str.strip(w))
    return wl


def get_wordvector(model, token):
    try:
        return list(model.vectors[model.key_to_index[token]])
    except Exception:
        return np.zeros(shape=model.vector_size)


def get_word2vec_model(file, ncols, win):
    return KeyedVectors.load_word2vec_format(file + '_' + str(ncols) + '_' + str(win) + '.txt', binary=False)


def get_glove_data(glove_dir, file):
    glove_dict = {}
    f = open(os.path.join(glove_dir, file), encoding="utf8")
    for line in f:
        values = line.split()
        word = values[0]
        try:
            coefs = np.asarray(values[1:], dtype='float32')
        except Exception:
            print(values[1:])
        norm = np.sqrt((coefs ** 2).sum())
        glove_dict[word] = coefs /norm
    f.close()
    print('Found %s word vectors.' % len(glove_dict))
    return glove_dict


def normalize2(data):
    for i, d in enumerate(data):
        norm = np.sqrt((d ** 2).sum())
        data[i] = d / norm
    return data


def to_number(file):
    reviews = MyReviews(file)
    vocab = set()
    max_len = 0
    for review in reviews:
        for token in review:
            vocab.add(token)
        if len(review) > max_len:
            max_len = len(review)
    word_to_ix = {word: i for i, word in enumerate(vocab)}
    return word_to_ix, max_len


def get_shuffle_list_neutral(file_pos, file_neg, file_neu, shuffle, stop_words = None, max_len=100):
    clist = []
    list_pos = MyReviews(file_pos)
    list_neg = MyReviews(file_neg)
    list_neu = MyReviews(file_neu)
    for t in list_pos:
        # t = remove_stopwords(t, stop_words)
        clist.append((t, [1, 0, 0]))
    for t in list_neg:
        # t = remove_stopwords(t, stop_words)
        clist.append((t, [0, 0, 1]))
    for t in list_neu:
        # t = remove_stopwords(t, stop_words)
        clist.append((t, [0, 1, 0]))
    if shuffle:
        random.shuffle(clist)
    return clist


def unit(data):
    for i, d in enumerate(data):
        magnitude = np.sqrt(np.dot(d, d))
        data[i] = d / magnitude
    return data


def unitDic(dic):
    for k, v in dic.items():
        magnitude = np.sqrt(np.dot(v, v))
        dic[k] = v / magnitude
    return dic


def get_max_number_of_token_list(*reviews_list):
    max_token = 0
    for reviews in reviews_list:
        for review in reviews:
            tokens = []
            for token in review[0]:
                tokens.append(token)
            length = len(tokens)
            if max_token < length:
                max_token = length
    return max_token


def get_token_matrix(model, review, max_rlen, ncols, glove_dict, gm):
    token_list = np.zeros(shape=(max_rlen, gm*ncols))
    for t, token in enumerate(review):
        wv = np.asarray(get_wordvector(model, token))
        if np.all((wv == 0)):
            subToken = token[1:]
            wv = np.asarray(get_wordvector(model, subToken))
        if gm > 1:
            glove = np.asarray(glove_dict.get(token, np.zeros(ncols)))
            wv = np.append(wv, glove)
        token_list[t] = wv
    return token_list