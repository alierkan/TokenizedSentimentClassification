"""
Turkish Syllable Rules (This code is valid only for Turkish):
1- Find the position of the rightmost vowel.
2- If the letter to the left of the rightmost vowel is a vowel,
   then separate the syllable from this rightmost vowel to the right.
3- No, if the letter to the left of the rightmost vowel is a consonant,
   add this consonant to the syllable and separate it that way.
4- After discarding the formed syllable,
   accept the part of the word up to there as a new word and return to the beginning.
5- If there are no vowels left, add the letter or letters to the last formed syllable.
"""
import glob

vowels = {'a', 'e', 'ı', 'i', 'o', 'ö', 'u', 'ü'}  # Turkish vowels
PATH = "/../"


def combine_files():
    filenames = glob.glob(PATH + '*.txt')
    new_file = "all.txt"
    with open(PATH + new_file, 'w') as outfile:
        for fn in filenames:
            with open(fn) as infile:
                outfile.write(infile.read())
    return new_file


def find_right_vowel(word):
    for i, w in reversed(list(enumerate(word))):
        if w in vowels:
            return i


def find_syllable(word, syllables):
    i = find_right_vowel(word)
    if i and i > 0:
        if word[i - 1] in vowels:
            syllables.append(word[i:])
            word = word[:i]
        else:
            syllables.append(word[i-1:])
            word = word[:i-1]
        find_syllable(word, syllables)
    elif i:
        if word[0] in vowels:
            syllables.append(word[0])
        else:
            syllables[-1] = word[0] + syllables[-1]
    else:
        syllables.append(word)


def get_syllables(word):
    syllables = []
    try:
        find_syllable(word, syllables)
    except Exception as e:
        print(word + "--" + str(e))
    return list(reversed(syllables))


class MyReviews(object):
    def __init__(self, file):
        self.file = file

    def __iter__(self):
        print(self.file)
        for line in open(self.file, encoding="utf8"):
            token_list = []
            word_list = line.lower().split()
            for w in word_list:
                sylables = get_syllables(w)
                token_list.extend(sylables)
            yield token_list


def write_syllables(input_file, output_file):
    rew_list = MyReviews(input_file)
    with open(output_file, "w") as output:
        for tokens in rew_list:
            sentence = " ".join(tokens) + "\n"
            output.write(sentence)


if __name__ == "__main__":
    combine_files()
    filenames = myFilesPaths = glob.glob(PATH + '*.txt')
    for filename in filenames:
        write_syllables(filename, filename + '.syllable')
