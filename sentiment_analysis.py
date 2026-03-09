# Libraries
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords

# Preprocessing
data = pd.read_csv("IMDB Dataset.csv")
print(data.head())

print("\nDataset shape:")
print(data.shape)
print("\nMissing values count:")
print(data.isnull().sum())

# cleaning text data
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

data['clean_text'] = data['review'].apply(clean_text)