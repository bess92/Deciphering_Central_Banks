FROM python:3.10.6-buster

RUN pip install --upgrade pip

COPY requirements.txt /requirements.txt
RUN pip install -r requirements.txt

COPY deciphering_cb deciphering_cb
COPY sentiment_model sentiment_model
COPY agent_model agent_model

CMD uvicorn deciphering_cb.api.fast:app --host 0.0.0.0 --port $PORT
