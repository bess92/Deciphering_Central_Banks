import pickle
import os


def load_models(sent_model, agent_model):

    dirname = os.path.dirname(__file__)

    sent_model_path = os.path.abspath(os.path.join(dirname, f'../../notebooks/{sent_model}'))
    agent_model_path = os.path.abspath(os.path.join(dirname, f'../../notebooks/{agent_model}'))

    with open(sent_model_path, 'rb') as f:
        sent_model_pkl = pickle.load(f)
    with open(agent_model_path, 'rb') as f:
        agent_model_pkl = pickle.load(f)

    return sent_model_pkl, agent_model_pkl

def load_tokenizers(sent_tok, agent_tok):

    dirname = os.path.dirname(__file__)

    sent_model_path = os.path.abspath(os.path.join(dirname, f'../../notebooks/{sent_tok}'))
    agent_model_path = os.path.abspath(os.path.join(dirname, f'../../notebooks/{agent_tok}'))

    with open(sent_model_path, 'rb') as f:
        sent_tok_pkl = pickle.load(f)
    with open(agent_model_path, 'rb') as f:
        agent_tok_pkl = pickle.load(f)

    return sent_tok_pkl, agent_tok_pkl
