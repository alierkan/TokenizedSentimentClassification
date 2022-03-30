import glob
from polyglot.text import Text, Word

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


class MyReviews(object):
    def __init__(self, file):
        self.file = file

    def __iter__(self):
        print(self.file)
        for line in open(self.file, encoding="utf8"):
            text = Text(line.lower())
            text.language = "en"
            yield text.morphemes


def write_morpheme(input_file, output_file):
    rew_list = MyReviews(input_file)
    with open(output_file, "w") as output:
        for words in rew_list:
            sentence = " ".join(words)
            # sentence = " ".join([Word(w, language="en") for w in words])
            output.write(sentence+'\n')


if __name__ == "__main__":
    combine_files()
    filenames = myFilesPaths = glob.glob(PATH + '*.txt')
    for filename in filenames:
        write_morpheme(filename, filename + '.morpheme')
