FROM python:3.11

EXPOSE 8501
WORKDIR /app

ARG LANG=en_US.UTF-8

COPY requirements.txt ./requirements.txt
#Download and install dependencies
RUN pip3 install -r requirements.txt
RUN pip install -U pip setuptools wheel
RUN pip install -U spacy
RUN python -m spacy download en_core_web_sm

COPY . .

# he image what to do when it starts as a container
CMD streamlit run app.py

