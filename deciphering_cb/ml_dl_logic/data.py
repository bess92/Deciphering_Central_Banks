# Save, load, and clean data
import pandas as pd
import re
from bs4 import BeautifulSoup
import requests

def replace_decimal_points(input_string):
    """
    Replace decimal points between digits in a string with commas.

    Parameters:
    input_string: str - The input string where decimal points need to be replaced.

    Returns:
    str - The modified string with decimal points replaced by commas.
    """

    return re.sub(r'(?<=\d)\.(?=\d)', ',', input_string)

########################################################################################################################

def new_text(new_speech):
    """
    Process the input text to prepare it for model input.

    This function formats the input speech by removing unnecessary characters, splitting
    it into sentences, filtering out questions, and returning a DataFrame of sentences.

    Parameters:
    new_speech: str - The raw input text.

    Returns:
    pd.DataFrame - A DataFrame where each sentence is a row.
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
    """
    Scrape text content from a given URL.

    This function extracts the main text content from a web page, removing unwanted tags
    and formatting.

    Parameters:
    url: str - The URL of the web page to scrape.

    Returns:
    str - The cleaned and extracted text content, or an error message in case of failure.
    """

    try:
        response = requests.get(url)
        if response.status_code != 200:
            return f"Failed to retrieve the website. Status code: {response.status_code}"
        soup = BeautifulSoup(response.content, 'html.parser')
        target_div = soup.select_one('#main-wrapper > main > div.section')
        for h2 in target_div.find_all('h2'):
            h2.decompose()
        for tag in target_div.find_all(['p', 'a'], class_=['ecb-publicationDate', 'arrow']):
            tag.decompose()
        text = target_div.get_text(separator=' ', strip=True)
        return text
    except Exception as e:
        return f"An error occurred: {e}"
