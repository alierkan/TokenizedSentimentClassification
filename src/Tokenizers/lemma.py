"""
wget http://nlp.stanford.edu/software/stanford-corenlp-full-2018-02-27.zip
unzip stanford-corenlp-full-2018-02-27.zip
cd stanford-corenlp-full-2018-02-27

java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -preload tokenize,ssplit,pos,lemma,stem,ner,parse,depparse -status_port 9000 -port 9000 -timeout 15000 -threads 4&
"""
import glob
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.parse import CoreNLPParser
from nltk.corpus import wordnet

wordnet_lemmatizer = WordNetLemmatizer()
parser = CoreNLPParser(url='http://localhost:9000')
pos_tagger = CoreNLPParser(url='http://localhost:9000', tagtype='pos')

PATH = "/../"
ISYELP = False


def combine_files():
    filenames = glob.glob(PATH + '*.txt')
    new_file = "all.txt"
    if ISYELP:
        new_file = "yelp-all.txt"
    with open(PATH + new_file, 'w') as outfile:
        for fn in filenames:
            with open(fn) as infile:
                outfile.write(infile.read())
    return new_file


def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


class MyReviews(object):
    def __init__(self, file):
        self.file = file

    def __iter__(self):
        print(self.file)
        for line in open(self.file, encoding="utf8"):
            token_list = []
            word_list = line.lower().split()
            if word_list:
                pos = list(pos_tagger.tag(word_list))
                pos_dic = dict(pos)
                for w, p in pos_dic.items():
                    token_list.append(wordnet_lemmatizer.lemmatize(w, get_wordnet_pos(p)))
            yield token_list


def write_lemmas(input_file, output_file):
    rew_list = MyReviews(input_file)
    with open(output_file, "w") as output:
        for words in rew_list:
            sentence = " ".join(words) + "\n"
            output.write(sentence)


if __name__ == "__main__":
    combine_files()
    filenames = myFilesPaths = glob.glob(PATH + '*.txt')
    for filename in filenames:
        write_lemmas(filename, filename + '.lemma')
