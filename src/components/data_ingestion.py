import os
import sys
import re
import pickle
import nltk
import keras
import tensorflow as tf
from nltk.corpus import stopwords
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass

nltk.download('stopwords')
stopword = set(stopwords.words('english'))

load_model = keras.models.load_model('experiment/data/trained_data/trained_model.h5')
with open('experiment/data/trained_data/tokenizer.pickle', 'rb') as handle:
    load_tokenizer = pickle.load(handle)

@dataclass
class DataIngestionConfig:
    cleaned_text: str = os.path.join('artifacts', 'cleaned_text.txt')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self, text):
        logging.info("Starting data ingestion")
        try:
            def clean_data(text):
                text = str(text).lower()
                text = re.sub('https?://\S+|www.\S+', '', text)
                text = re.sub('<[^>]+>', '', text)
                text = re.sub(r'[^\w\s]', '', text)
                text = re.sub('\n', '', text)
                text = re.sub('\w\d\w', '', text)
                text = ' '.join([word for word in text.split(' ') if word not in stopword])
                return text

            cleaned_text = clean_data(text)
            return cleaned_text
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == '__main__':
    obj = DataIngestion()
    test_text = "Humans are idiots."
    clean_text = obj.initiate_data_ingestion(test_text)
    print(clean_text)
