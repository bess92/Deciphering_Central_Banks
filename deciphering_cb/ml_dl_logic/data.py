# Save, load, and clean data
import pandas as pd
import re
from bs4 import BeautifulSoup
import requests
########################################################################################################################

# def train_data_sentiment():

#     dirname = os.path.dirname(__file__)

#     raw_data_path = os.path.abspath(os.path.join(dirname, f'../../raw_data'))

#     """
#     Loads the training data for sentiment classification
#     """

#     df_ecb_sent=pd.read_csv(os.path.join(raw_data_path,'ECB_prelabelled_sent.txt'))
#     df_fed_sent=pd.read_csv(os.path.join(raw_data_path,'FED_prelabelled_sent.txt'))
#     df_bis_sent=pd.read_csv(os.path.join(raw_data_path,'BIS_prelabelled_sent.txt'))

#     df_fed_sent.drop(columns=['audience'], inplace=True)
#     df_sent=pd.concat([df_ecb_sent,df_fed_sent,df_bis_sent], axis=0)
#     return df_sent

# ########################################################################################################################

# def train_data_agents():

#     dirname = os.path.dirname(__file__)

#     raw_data_path = os.path.abspath(os.path.join(dirname, f'../../raw_data'))
#     """
#     Loads the data for agent classification
#     """
#     df_ecb_ag=pd.read_csv(os.path.join(raw_data_path,'ECB_prelabelled.txt'))
#     df_fed_ag=pd.read_csv(os.path.join(raw_data_path,'FED_prelabelled.txt'), lineterminator='\n')
#     df_bis_ag=pd.read_csv(os.path.join(raw_data_path,'BIS_prelabelled.txt'))

#     df_bis_ag.drop(columns=['Unnamed: 0'], inplace=True)
#     df_ecb_ag.drop(columns=['Unnamed: 0'], inplace=True)

#     df_ag=pd.concat([df_ecb_ag,df_fed_ag,df_bis_ag], axis=0)
#     df_ag.columns=['text', 'agent']

#     return df_ag

########################################################################################################################

def replace_decimal_points(input_string):
    # Replace dots between digits with commas
    return re.sub(r'(?<=\d)\.(?=\d)', ',', input_string)

def new_text(new_speech):
    """
    takes new text as a string and puts it in the correct imput format
    """
    # Remove any spaces after full stops, exclamation marks, or question marks.
    speech = replace_decimal_points(new_speech.replace('. ', '.').replace('! ', '.').replace('? ', '?.').replace('*','').replace('"',''))


    # Split the speech into sentences
    sentences = [sentence.strip() for sentence in speech.split('.') if sentence]

    #Get rid of questions
    sentences=[sentence for sentence in sentences if not sentence.endswith("?")]

    # Add a period back to the end of each sentence
    sentences = [sentence + '.' for sentence in sentences]

    # Create the DataFrame with each sentence as a separate column
    df = pd.DataFrame(sentences, columns=['Sentence'])
    return df


########################################################################################################################

def scrape_website_text(url):
    try:
        response = requests.get(url)
        if response.status_code != 200:
            return f"Failed to retrieve the website. Status code: {response.status_code}"
        soup = BeautifulSoup(response.content, 'html.parser')
        target_div = soup.select_one('#main-wrapper > main > div.section')
        for h2 in target_div.find_all('h2'):
            h2.decompose()
        text = target_div.get_text(separator=' ', strip=True)
        return text
    except Exception as e:
        return f"An error occurred: {e}"
# How to use it
url = "https://www.ecb.europa.eu/press/press_conference/monetary-policy-statement/2024/html/ecb.is240718~6600b4add6.en.html"
scraped_text = scrape_website_text(url)

########################################################################################################################

# def preprocess(text):

#     speech_processed = new_text(text)
#     speech_tokenized_sent = roberta_tokenizer(sentences=speech_processed)
#     speech_tokenized_ag = bert_tokenizer(sentences=speech_processed)

#     return speech_tokenized_sent, speech_tokenized_ag, speech_processed
