from enum import Enum
from typing import Literal

PATH_2021 = '/home/alierkan/phd/2021'  # Path has to be defined.

# Datasets
SEMEVAL = "semeval"
TWITTER = "twitter"
IMDB = "imdb"
YELP = "yelp"
BEYAZPERDE = "beyazperde"

#Tokenizations
PRE = "pre"
BPE = "bpe"
PREBPE = "prebpe"
WORD = "word"
STEM = "stem"
MORPHEME = "morpheme"
LEMMA = "lemma"
WORDPIECE = "wordpiece"
WORDPIECENOSIGN = "wordpieceNoSign"
UNIGRAM = "unigram"
STOPWORDS = "stopwords"
BPESTOPWORDS = "bpestopwords"
SYLLABLE = "syllable"

# Class Enumerator
class Polarity(Enum):
    pos = "pos"
    neg = "neg"
    neu = "neu"

YELP_CORPUS = "yelp.txt"

# Path Dictionary for different datasets and tokenizations
path_dic = {SEMEVAL: {"bpe": PATH_2021 + '/ing/semeval_rest_2016/bpe/',
                      "bpestopwords": PATH_2021 + '/ing/semeval_rest_2016/bpestopwords/',
                      "word": PATH_2021 + '/ing/semeval_rest_2016/word/',
                      "stopwords": PATH_2021 + '/ing/semeval_rest_2016/stopwords/',
                      "stem": PATH_2021 + '/ing/semeval_rest_2016/stem/',
                      "morpheme": PATH_2021 + '/ing/semeval_rest_2016/morpheme/',
                      "lemma": PATH_2021 + '/ing/semeval_rest_2016/lemma/',
                      "pre": PATH_2021 + '/ing/semeval_rest_2016/pre/',
                      "prebpe": PATH_2021 + '/ing/semeval_rest_2016/prebpe/',
                      "wordpiece": PATH_2021 + '/ing/semeval_rest_2016/wordpiece/',
                      "unigram": PATH_2021 + '/ing/semeval_rest_2016/unigram/'},

            TWITTER: {"bpe": PATH_2021 + '/ing/semeval_twitter_2017/bpe/',
                      "word": PATH_2021 + '/ing/semeval_twitter_2017/word/',
                      "stem": PATH_2021 + '/ing/semeval_twitter_2017/stem/',
                      "lemma": PATH_2021 + '/ing/semeval_twitter_2017/lemma/',
                      "pre": PATH_2021 + '/ing/semeval_twitter_2017/pre/',
                      "prebpe": PATH_2021 + '/ing/semeval_twitter_2017/prebpe/',
                      "train": PATH_2021 + '/ing/semeval_twitter_2017/train/',
                      "stopwords": PATH_2021 + '/ing/semeval_twitter_2017/stopwords/',
                      "bpestopwords": PATH_2021 + '/ing/semeval_twitter_2017/bpestopwords/',
                      "morpheme": PATH_2021 + '/ing/semeval_twitter_2017/morpheme/',
                      "wordpiece": PATH_2021 + '/ing/semeval_twitter_2017/wordpiece/',
                      "unigram": PATH_2021 + '/ing/semeval_twitter_2017/unigram/'},

            IMDB: {"bpe": PATH_2021 + '/ing/imdb/bpe/',
                   "word": PATH_2021 + '/ing/imdb/word/',
                   "pre": PATH_2021 + '/ing/imdb/pre/',
                   "stopwords": PATH_2021 + '/ing/imdb/stopwords/',
                   "bpestopwords": PATH_2021 + '/ing/imdb/bpestopwords/',
                   "lemma": PATH_2021 + '/ing/imdb/lemma/',
                   "stem": PATH_2021 + '/ing/imdb/stem/',
                   "morpheme": PATH_2021 + '/ing/imdb/morpheme/',
                   "wordpiece": PATH_2021 + '/ing/imdb/wordpiece/',
                   "unigram": PATH_2021 + '/ing/imdb/unigram/'
                   },

            YELP: {"pre": PATH_2021 + '/ing/yelp/pre/'},

            BEYAZPERDE: {"word": PATH_2021 + '/turk/beyazperde/word/',
                         "pre": PATH_2021 + '/turk/beyazperde/pre/',
                         "stopwords": PATH_2021 + '/turk/beyazperde/stopwords/',
                         "bpe": PATH_2021 + '/turk/beyazperde/bpe/',
                         "bpestopwords": PATH_2021 + '/turk/beyazperde/bpestopwords/',
                         "morpheme": PATH_2021 + '/turk/beyazperde/morpheme/',
                         "stem": PATH_2021 + '/turk/beyazperde/stem/',
                         "wordpiece": PATH_2021 + '/turk/beyazperde/wordpiece/',
                         "unigram": PATH_2021 + '/turk/beyazperde/unigram/',
                         "wordpieceNoSign": PATH_2021 + '/turk/beyazperde/wordpiece-nosign/',
                         "lemma": PATH_2021 + '/turk/beyazperde/lemma/',
                          "syllable": PATH_2021 + '/turk/beyazperde/syllable/'
                         }
            }


def get_paths(data):
    return path_dic.get(data)


# Some other constants
STRING = "string/"
NUM = "num/"
TRAIN = "train"
AVERAGE = 'macro'
