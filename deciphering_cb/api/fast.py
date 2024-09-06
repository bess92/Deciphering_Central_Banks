from fastapi import FastAPI
from deciphering_cb.ml_dl_logic.data import preprocess
from deciphering_cb.ml_dl_logic.registry import load_models
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import nltk
# Create FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"], ) # Allows all headers

# Load model (e.g. tensorflow, transformer, sklearn)
# Load the model at startup and store it in app.states
app.state.model_sent, app.state.model_agent = load_models('svm_sentiment.pkl', 'svm_agent.pkl')

#nltk.download('wordnet')

# predict
@app.get("/predict")
def predict(text):
    speech_preproc_sent, speech_preproc_ag = preprocess(text)
    y_pred_sent = app.state.model_sent.predict(speech_preproc_sent)
    y_pred_ag = app.state.model_agent.predict(speech_preproc_ag)
    print(y_pred_ag, y_pred_sent)
    return{
        'Sentiment Classification' : int(np.argmax(y_pred_sent)),
        'Agent Classification' : int(np.argmax(y_pred_ag))
    }

#import transformers
#from transformers import AutoTokenizer,TFAutoModelForSequenceClassification
#bess = TFAutoModelForSequenceClassification.from_pretrained('sentiment_bert')
#bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
#X1_test_berttok = bert_tokenizer(X1_test.to_list(), max_length=150, padding = "max_length", truncation = True, return_tensors="tf")
#bess.predict(X1_test_berttok)
#prediction = tf.round(tf.nn.sigmoid(logit))
#np.argmax(prediction)
