# Tokenized Sentiment Classification

## Tokenizers:

**bpe.py** : Byte pair encoding

**lemma.py** : Lemma encoding

**morpheme.py** : Morphemer

**preprocess.py** : Word Preprocssing

**sentencepiece.py** : Sentence Piece encoding

**stem.py** : Stemmer

**syllable.py** : Turkish syllable extractor

## Classifiers:

**cnn.py** : Static CNN Model with Word2vec and GloVe embedding vectors.

**cnn_dynamic.py** : Dynamic CNN Model with Word2vec and GloVe embedding vectors.

**bert_nn.py** : FFNN Model with BERT last layer CLS vector.

**bert_cnn.py** : Dynamic CNN Model with BERT last layer CLS vector.

**roberta_nn.py** : FNN Model with RoBERTa last layer CLS vector.

**roberta_cnn.py** : Dynamic CNN Model with RoBERTa last layer CLS vector.

## Embedding:

**word2vec_generate.py** : Word2vec generator by using gensim.
