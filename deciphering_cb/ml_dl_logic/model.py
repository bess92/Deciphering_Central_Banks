import tensorflow as tf
import numpy as np

def ag_predict(tokenizer, model, speech, app):

    speech_preproc_ag = tokenizer(speech['Sentence'].to_list(), max_length=150, padding = "max_length", truncation = True, return_tensors="tf")
    y2_preds = tf.nn.softmax(model.predict(speech_preproc_ag)['logits'])
    app.state.agents=[np.argmax(i) if max(i)>0.85 else 5 for i in y2_preds]
    app.state.agents_p=[round(float(max(i)),2) for i in y2_preds]


def sent_predict(tokenizer, model, speech, app):

    speech_preproc_sent = tokenizer(speech['Sentence'].to_list(), max_length=150, padding = "max_length", truncation = True, return_tensors="tf")
    y1_preds=tf.nn.sigmoid(model.predict(speech_preproc_sent)['logits'])
    app.state.sentiments=[np.argmax(i) if max(i)>0.65 else 2 for i in y1_preds]
    app.state.sent_p=[round(float(max(i)),2) for i in y1_preds]
