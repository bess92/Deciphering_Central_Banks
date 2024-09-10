from fastapi import FastAPI
from deciphering_cb.ml_dl_logic.data import new_text, scrape_website_text
from deciphering_cb.ml_dl_logic.registry import load_models
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import tensorflow as tf
import pandas as pd
from transformers import AutoTokenizer
from deciphering_cb.ml_dl_logic.model import ag_predict, sent_predict
import threading

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

    t1 = threading.Thread(target=ag_predict, args=(app.state.bert_tokenizer, app.state.model_agent, speech, app))
    t2 = threading.Thread(target=sent_predict, args=(app.state.roberta_tokenizer, app.state.model_sent, speech, app))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    agents, agents_p = app.state.agents, app.state.agents_p
    sentiments, sent_p = app.state.sentiments, app.state.sent_p

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

    t1 = threading.Thread(target=ag_predict, args=(app.state.bert_tokenizer, app.state.model_agent, speech, app))
    t2 = threading.Thread(target=sent_predict, args=(app.state.roberta_tokenizer, app.state.model_sent, speech, app))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    agents, agents_p = app.state.agents, app.state.agents_p
    sentiments, sent_p = app.state.sentiments, app.state.sent_p

    speech['agents']=agents
    speech['agents_prob']=agents_p
    speech['sentiment']=sentiments
    speech['sentiment_probs']=sent_p

    output=speech[(speech.agents<5) & (speech.sentiment<2)]
    output.replace({"agents": ag_classes},inplace=True)
    output.replace({"sentiment": sent_classes},inplace=True)

    output_dict = output.to_dict(orient='records')

    return output_dict
