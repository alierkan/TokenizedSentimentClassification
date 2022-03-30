import gensim
import logging
from gensim.utils import tokenize as gensimTokenizer

PATH = "/../"
YELP = False

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


class MyReviews(object):
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        print(self.filename)
        for line in open(self.filename, encoding="utf8"):
            yield list(gensimTokenizer(line.lower()))


def word2vec(input_file):
    reviews = MyReviews(input_file + '.txt')
    NSIZE = 300
    WIN = 5
    model = gensim.models.Word2Vec(reviews, vector_size=NSIZE, window=WIN, min_count=5)
    model.save(input_file + '_word2vector_' + str(NSIZE) + '_' + str(WIN) + '.bin')
    model.wv.save_word2vec_format(input_file + '_word2vector_' + str(NSIZE) + '_' + str(WIN) + '.txt', binary=False)


if __name__ == "__main__":
    word2vec(PATH + "all")
