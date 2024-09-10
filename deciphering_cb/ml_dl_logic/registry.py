import os
from transformers import TFAutoModelForSequenceClassification


def load_models():

    dirname = os.path.dirname(__file__)

    sent_model_path = os.path.abspath(os.path.join(dirname, f'../../sentiment_model'))
    agent_model_path = os.path.abspath(os.path.join(dirname, f'../../agent_model'))

    sent_model_tf = TFAutoModelForSequenceClassification.from_pretrained(sent_model_path)
    agent_model_tf = TFAutoModelForSequenceClassification.from_pretrained(agent_model_path)

    return sent_model_tf, agent_model_tf
