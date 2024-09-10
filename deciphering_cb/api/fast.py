from fastapi import FastAPI
from deciphering_cb.ml_dl_logic.data import new_text, scrape_website_text
from deciphering_cb.ml_dl_logic.registry import load_models
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import tensorflow as tf
import pandas as pd
from transformers import AutoTokenizer

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
app.state.model_sent, app.state.model_agent = load_models()
app.state.bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
app.state.roberta_tokenizer = AutoTokenizer.from_pretrained("roberta-base")

# predict
@app.get("/predict")
def predict(text):

    ag_classes = {0: "Households", 1: "Firms", 2: 'Financial Sector', 3: 'Government', 4: 'Central bank'}
    sent_classes = {0: "Negative", 1: "Positive"}

    speech = new_text(text)

    speech_preproc_ag = app.state.bert_tokenizer(speech['Sentence'].to_list(), max_length=150, padding = "max_length", truncation = True, return_tensors="tf")
    speech_preproc_sent = app.state.roberta_tokenizer(speech['Sentence'].to_list(), max_length=150, padding = "max_length", truncation = True, return_tensors="tf")

    #sentiment predicition
    y1_preds=tf.nn.sigmoid(app.state.model_sent.predict(speech_preproc_sent)['logits'])
    sentiments=[np.argmax(i) if max(i)>0.65 else 2 for i in y1_preds]
    sent_p=[round(float(max(i)),2) for i in y1_preds]

    # agent prediction
    y2_preds = tf.nn.softmax(app.state.model_agent.predict(speech_preproc_ag)['logits'])
    agents=[np.argmax(i) if max(i)>0.85 else 5 for i in y2_preds]
    agents_p=[round(float(max(i)),2) for i in y2_preds]

    speech['agents']=agents
    speech['agents_prob']=agents_p
    speech['sentiment']=sentiments
    speech['sentiment_probs']=sent_p

    output=speech[(speech.agents<5) & (speech.sentiment<2)]
    output.replace({"agents": ag_classes},inplace=True)
    output.replace({"sentiment": sent_classes},inplace=True)

    output_dict = output.to_dict(orient='records')

    return output_dict

@app.get("/predict_by_url")
def predict_by_url(url):
    text = scrape_website_text(url)
    ag_classes = {0: "Households", 1: "Firms", 2: 'Financial Sector', 3: 'Government', 4: 'Central bank'}
    sent_classes = {0: "Negative", 1: "Positive"}

    speech = new_text(text)

    speech_preproc_ag = app.state.bert_tokenizer(speech['Sentence'].to_list(), max_length=150, padding = "max_length", truncation = True, return_tensors="tf")
    speech_preproc_sent = app.state.roberta_tokenizer(speech['Sentence'].to_list(), max_length=150, padding = "max_length", truncation = True, return_tensors="tf")

    #sentiment predicition
    y1_preds=tf.nn.sigmoid(app.state.model_sent.predict(speech_preproc_sent)['logits'])
    sentiments=[np.argmax(i) if max(i)>0.65 else 2 for i in y1_preds]
    sent_p=[round(float(max(i)),2) for i in y1_preds]

    #agent prediction
    y2_preds = tf.nn.softmax(app.state.model_agent.predict(speech_preproc_ag)['logits'])
    agents=[np.argmax(i) if max(i)>0.85 else 5 for i in y2_preds]
    agents_p=[round(float(max(i)),2) for i in y2_preds]

    speech['agents']=agents
    speech['agents_prob']=agents_p
    speech['sentiment']=sentiments
    speech['sentiment_probs']=sent_p

    output=speech[(speech.agents<5) & (speech.sentiment<2)]
    output.replace({"agents": ag_classes},inplace=True)
    output.replace({"sentiment": sent_classes},inplace=True)

    output_dict = output.to_dict(orient='records')
    return output_dict

# Make a model.py
# ag_pred and sent_pred taking tokenizer, model as arguments
# call these functions within prediction()
#agpreds = ag_pred(app.state.tokeniser, app.state.model)
# this is a test
