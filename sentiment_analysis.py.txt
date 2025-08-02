import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re

# Download NLTK data
nltk.download('stopwords')

# Sample data
data = {
    'text': [
        'I love this product!',
        'This is the worst thing I bought.',
        'Absolutely fantastic and useful.',
        'Horrible experience.',
        'It is okay, not great, not bad.',
        'I will never buy this again!',
        'Totally worth it!',
        'Terrible, very disappointing.'
    ],
    'sentiment': ['positive', 'negative', 'positive', 'negative', 'neutral', 'negative', 'positive', 'negative']
}

df = pd.DataFrame(data)

# Text preprocessing
def preprocess(text):
    ps = PorterStemmer()
    text = re.sub('[^a-zA-Z]', ' ', text).lower()
    text = text.split()
    text = [ps.stem(word) for word in text if word not in stopwords.words('english')]
    return ' '.join(text)

df['processed'] = df['text'].apply(preprocess)

# Feature extraction
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df['processed']).toarray()
y = df['sentiment']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Results
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Predict custom input
print("\nTest Your Own Text:")
while True:
    custom_text = input("Enter a sentence (or 'q' to quit): ")
    if custom_text.lower() == 'q':
        break
    cleaned = preprocess(custom_text)
    vector = tfidf.transform([cleaned]).toarray()
    print("Predicted Sentiment:", model.predict(vector)[0])
