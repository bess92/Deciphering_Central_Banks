from fastapi import FastAPI
from deciphering_cb.ml_dl_logic.data import new_text, clean_data, train_data_agents, train_data_sentiment
from deciphering_cb.ml_dl_logic.tokenizer import  word2vec_tokenizer, Tokenizer_tokenizer, BERT_tokenizer
from deciphering_cb.ml_dl_logic.registry import load_model
# Create FastAPI app
app = FastAPI()

# Load model (e.g. tensorflow, transformer, sklearn)
# Load the model at startup and store it in app.state
app.state.model = load_model()

# if using preprocessor, load preprocessor
def preprocess(text):
    train_data_sent = train_data_sentiment() # Remember to select 1 column
    train_data_ag = train_data_agents() # Remember to select 1 column

    speech_processed = new_text(text)
    speech_clean_sent = clean_data(speech_processed, remove_sw=False)
    speech_clean_ag = clean_data(speech_clean_sent, remove_sw=True)
    speech_tokenized_sent_train, speech_tokenized_sent_test = Tokenizer_tokenizer(sentences_train=train_data_sent, sentences_test=speech_clean_sent) #word2vec_tokenizer used as a placeholder
    speech_tokenized_ag_train, speech_tokenized_ag_train_test = word2vec_tokenizer(sentences_train=train_data_ag, sentences_test=speech_clean_ag) #word2vec_tokenizer used as a placeholder
    return speech_tokenized_sent_test, speech_tokenized_ag_train_test

# predict
@app.get("/predict")
def predict(text):
    speech_preproc_sent, speech_preproc_ag = preprocess(text)
    y_pred_sent = app.state.model.predict(speech_preproc_sent)
    y_pred_ag = app.state.model.predict(speech_preproc_ag)

    return{
        'Sentiment Classification' : y_pred_sent,
        'Agent Classification' : y_pred_ag
    }
