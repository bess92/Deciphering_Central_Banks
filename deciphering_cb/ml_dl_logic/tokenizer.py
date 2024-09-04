from gensim.models import Word2Vec
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

########################################################################################################################

#Word2Vec
def word2vec_tokenizer(sentences_train, sentences_test, vector_size=60, min_count=10, window=10):
    """
    This function returns padded and embedded sentences from a df of sentences
    Uses the Word2Vec tokenizer
    """

    word2vec = Word2Vec(sentences=sentences_train, vector_size=vector_size, min_count=min_count, window=window)
    def embed_sentence(word2vec, sentence):
        embedded_sentence = []
        for word in sentence:
            if word in word2vec.wv:
                embedded_sentence.append(word2vec.wv[word])

        return np.array(embedded_sentence)

    def embedding(word2vec, sentences):
        embed = []

        for sentence in sentences:
            embedded_sentence = embed_sentence(word2vec, sentence)
            embed.append(embedded_sentence)

        return embed

    # Embed the training and test sentences

    sentences_train_embed = embedding(word2vec=word2vec, sentences=sentences_train)
    sentences_test_embed = embedding(word2vec=word2vec, sentences=sentences_test)


    # Pad the training and test embedded sentences
    sentences_train_pad = pad_sequences(sentences_train_embed, dtype='float32', padding='post', maxlen=200)
    sentences_test_pad = pad_sequences(sentences_test_embed, dtype='float32', padding='post', maxlen=200)

    return sentences_train_pad, sentences_test_pad

########################################################################################################################

#Tokenizer
def Tokenizer_tokenizer(sentences_train, sentences_test):
    """
    Tokenizer_tokenizer takes df sentence_train and sntence_test, and applies
    Tokenizer() to the sentences
    """
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sentences_train)

    sentences_train_token = tokenizer.texts_to_sequence(sentences_train)
    sentences_test_token = tokenizer.texts_to_sequence(sentences_test)
    vocab_size = len(tokenizer.word_index)

    sentences_train_tok_pad = pad_sequences(sentences_train_token, dtype='float32', padding='post',maxlen=150)
    sentences_test_tok_pad = pad_sequences(sentences_test_token, dtype='float32', padding='post',maxlen=150)

    return sentences_train_tok_pad, sentences_test_tok_pad

########################################################################################################################

#BERT tokenizer
def BERT_tokenizer():
    pass # CODE HERE
