# Libraries
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

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

predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))

def predict_sentiment(text):
    text = clean_text(text)
    vector = vectorizer.transform([text])
    prediction = model.predict(vector)
    return prediction[0]

print("Review Status:")
print(predict_sentiment("If you watch it like \"Not Sherlock\" its a good show If you watch it like \"Sherlock\" its not.Sherlocks intellectual and deduction capacity is not at its full potential and can be reflected much better. Mycroft is completely different which actually he s capabilities are also extra ordinary but this version is very ordinary.Dont compare with Sherlock and enjoy the show as its a different tv series."))