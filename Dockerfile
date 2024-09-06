FROM python:3.10.6-buster

RUN pip install --upgrade pip

COPY requirements.txt /requirements.txt
RUN pip install -r requirements.txt

# RUN python -m nltk.downloader punkt -d /usr/local/nltk_data
# RUN python -m nltk.downloader averaged_perceptron_tagger -d /usr/local/nltk_data
# RUN python -m nltk.downloader wordnet -d /usr/local/nltk_data
# RUN python -m nltk.downloader stopwords -d /usr/local/nltk_data

COPY nltk_data /root/nltk_data
#RUN pip install .

COPY deciphering_cb deciphering_cb
COPY model model

CMD uvicorn deciphering_cb.api.fast:app --host 0.0.0.0 --port $PORT
