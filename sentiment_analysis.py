# Libraries
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

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

# perform vectorization
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(data['clean_text'])
y = data['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# model training
model = MultinomialNB()
model.fit(X_train, y_train)