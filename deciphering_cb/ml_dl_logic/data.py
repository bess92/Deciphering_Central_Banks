# Save, load, and clean data
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from deciphering_cb.ml_dl_logic.tokenizer import  tokenizer_tokenizer, bert_tokenizer
import os
########################################################################################################################

def train_data_sentiment():

    dirname = os.path.dirname(__file__)

    raw_data_path = os.path.abspath(os.path.join(dirname, f'../../raw_data'))

    """
    Loads the training data for sentiment classification
    """

    df_ecb_sent=pd.read_csv(os.path.join(raw_data_path,'ECB_prelabelled_sent.txt'))
    df_fed_sent=pd.read_csv(os.path.join(raw_data_path,'FED_prelabelled_sent.txt'))
    df_bis_sent=pd.read_csv(os.path.join(raw_data_path,'BIS_prelabelled_sent.txt'))

    df_fed_sent.drop(columns=['audience'], inplace=True)
    df_sent=pd.concat([df_ecb_sent,df_fed_sent,df_bis_sent], axis=0)
    return df_sent

########################################################################################################################

def train_data_agents():

    dirname = os.path.dirname(__file__)

    raw_data_path = os.path.abspath(os.path.join(dirname, f'../../raw_data'))
    """
    Loads the data for agent classification
    """
    df_ecb_ag=pd.read_csv(os.path.join(raw_data_path,'ECB_prelabelled.txt'))
    df_fed_ag=pd.read_csv(os.path.join(raw_data_path,'FED_prelabelled.txt'), lineterminator='\n')
    df_bis_ag=pd.read_csv(os.path.join(raw_data_path,'BIS_prelabelled.txt'))

    df_bis_ag.drop(columns=['Unnamed: 0'], inplace=True)
    df_ecb_ag.drop(columns=['Unnamed: 0'], inplace=True)

    df_ag=pd.concat([df_ecb_ag,df_fed_ag,df_bis_ag], axis=0)
    df_ag.columns=['text', 'agent']

    return df_ag

########################################################################################################################

def clean_data(text, remove_sw=True):
    """
    clean_text removes stopwords in the case remove_sw is set to false (sentiment)
    This function also lemmatizes the text and returns the data.
    Use df.apply(clean_data) to implement the function
    """

    text=text.split()
    if remove_sw:
        # remove words from stopword list
        stop_words = set(stopwords.words('english'))
        words_to_keep=['no','not','none', 'not', 'none',
            'neither',
            'never',
            'nobody',
            'nothing',
            'nowhere']
        filtered_stop_words = [word for word in stop_words if word not in words_to_keep]

        # remove stopwords from text
        words = [word for word in text if word not in filtered_stop_words]
    else:
        words=text

    lemma=WordNetLemmatizer() # Initiate Lemmatizer
    lemmatized = [lemma.lemmatize(word) for word in words] # Lemmatize
    cleaned = ' '.join(lemmatized) # Join back to a string
    return cleaned

########################################################################################################################

def new_text(new_speech):
    """
    takes new text as a string and puts it in the correct imput format
    """
    print(new_speech)
    # Remove any spaces after full stops, exclamation marks, or question marks.
    speech = new_speech.replace('. ', '.').replace('! ', '!').replace('? ', '?')

    # Split the speech into sentences
    sentences = [sentence.strip() for sentence in speech.split('.') if sentence]

    # Add a period back to the end of each sentence
    sentences = [sentence + '.' for sentence in sentences]

    # Create the DataFrame with each sentence as a separate column
    df = pd.DataFrame(sentences, columns=['Sentence'])
    print(df)
    return df

########################################################################################################################

def preprocess(text):

    speech_processed = new_text(text)
    speech_clean_sent = speech_processed['Sentence'].apply(lambda x: clean_data(x, remove_sw=False))
    speech_clean_ag = speech_processed['Sentence'].apply(lambda x: clean_data(x, remove_sw=True))
    #speech_tokenized_sent, speech_tokenized_ag = tokenizer_tokenizer(sentences_sent=speech_clean_sent, sentences_ag=speech_clean_ag)

    return speech_clean_sent, speech_clean_ag
