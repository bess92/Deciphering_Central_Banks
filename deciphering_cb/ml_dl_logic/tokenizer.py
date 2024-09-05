from gensim.models import Word2Vec
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from deciphering_cb.ml_dl_logic.registry import load_tokenizers
import numpy as np

########################################################################################################################

#Word2Vec
# def word2vec_tokenizer(sentences_train, sentences_test, vector_size=60, min_count=10, window=10):
#     """
#     This function returns padded and embedded sentences from a df of sentences
#     Uses the Word2Vec tokenizer
#     """

#     word2vec = Word2Vec(sentences=sentences_train, vector_size=vector_size, min_count=min_count, window=window)
#     def embed_sentence(word2vec, sentence):
#         embedded_sentence = []
#         for word in sentence:
#             if word in word2vec.wv:
#                 embedded_sentence.append(word2vec.wv[word])

#         return np.array(embedded_sentence)

#     def embedding(word2vec, sentences):
#         embed = []

#         for sentence in sentences:
#             embedded_sentence = embed_sentence(word2vec, sentence)
#             embed.append(embedded_sentence)

#         return embed

#     # Embed the training and test sentences

#     sentences_train_embed = embedding(word2vec=word2vec, sentences=sentences_train)
#     sentences_test_embed = embedding(word2vec=word2vec, sentences=sentences_test)


#     # Pad the training and test embedded sentences
#     sentences_train_pad = pad_sequences(sentences_train_embed, dtype='float32', padding='post', maxlen=200)
#     sentences_test_pad = pad_sequences(sentences_test_embed, dtype='float32', padding='post', maxlen=200)

#     return sentences_train_pad, sentences_test_pad

########################################################################################################################

#Tokenizer
def tokenizer_tokenizer(sentences_sent, sentences_ag):
    """
    Tokenizer_tokenizer takes df sentence_train and sntence_test, and applies
    Tokenizer() to the sentences
    """
    tokenizer_sent, tokenizer_ag = load_tokenizers('Tokenizer_sentiment.pkl', 'Tokenizer_agent.pkl')


    sentences_sent_token = tokenizer_sent.texts_to_sequences(sentences_sent)
    sentences_ag_token = tokenizer_ag.texts_to_sequences(sentences_ag)

    sentences_sent_tok_pad = pad_sequences(sentences_sent_token, dtype='float32', padding='post',maxlen=150)
    sentences_ag_tok_pad = pad_sequences(sentences_ag_token, dtype='float32', padding='post',maxlen=150)

    return sentences_sent_tok_pad, sentences_ag_tok_pad

########################################################################################################################

#BERT tokenizer
def bert_tokenizer():
    pass # CODE HERE
