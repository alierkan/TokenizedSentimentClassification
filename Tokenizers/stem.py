import glob
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')
stemmer = SnowballStemmer("english")

PATH = "/../"
print(stemmer.stem("running"))
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


class MyReviews(object):
    def __init__(self, file):
        self.file = file

    def __iter__(self):
        print(self.file)
        for line in open(self.file, encoding="utf8"):
            yield tokenizer.tokenize(line.lower())


def write_stem(input_file, output_file):
    rew_list = MyReviews(input_file)
    with open(output_file, "w") as output:
        for words in rew_list:
            stems = [stemmer.stem(word) for word in words]
            sentence = " ".join(stems)
            output.write(sentence+'\n')


if __name__ == "__main__":
    combine_files()
    filenames = myFilesPaths = glob.glob(PATH + '*.txt')
    for filename in filenames:
        write_stem(filename, filename + '.stem')
