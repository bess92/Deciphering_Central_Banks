import os
from transformers import TFAutoModelForSequenceClassification

def load_models():
    """
    Load pre-trained models for sentiment analysis and agent classification.

    Retrieves the models from the specified file paths and loads them using TensorFlow's
    `TFAutoModelForSequenceClassification`.

    Returns:
    -------
    tuple:
        sent_model_tf: The TensorFlow model for sentiment analysis.
        agent_model_tf: The TensorFlow model for agent classification.
    """
    dirname = os.path.dirname(__file__)

    sent_model_path = os.path.abspath(os.path.join(dirname, f'../../sentiment_model'))
    agent_model_path = os.path.abspath(os.path.join(dirname, f'../../agent_model'))

    sent_model_tf = TFAutoModelForSequenceClassification.from_pretrained(sent_model_path)
    agent_model_tf = TFAutoModelForSequenceClassification.from_pretrained(agent_model_path)

    return sent_model_tf, agent_model_tf
