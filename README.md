# Tokenized Sentiment Classification

## Tokenizers:

By using non-contextualized (Word2Vec and GloVe) and contextualized (BERT and RoBERTa) 
embeddings over some tokenization methods, classifiers learn to estimate 
sentiment polarities of reviews.  
Used tokenized methods are BPE, WordPiece, Lemma, Stem, Morpheme, Syllable.
Used classifiers are Feed-Forward Neural Network and Convolutional Neural Network.
Used embeddings are Word2Vec, GloVe, Bert, and RoBERTa.
Used datasets are:
1. IMDB Movie Reviews (https://ai.stanford.edu/~amaas/data/sentiment/).
2. Semeval 2016 Task 5 Restaurant Dataset (https://alt.qcri.org/semeval2016/task5/index.php?id=data-and-tools).
3. Semeval 2017 Task 6 Twitter Dataset.
4. Beyazperde Turkish Movie Dataset (http://humirapps.cs.hacettepe.edu.tr/tsad.aspx.)

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
