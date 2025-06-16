import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import nltk
from nltk.corpus import stopwords
import joblib
import re

# Download NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load Kaggle dataset
df = pd.read_csv('training.csv')

# Map integer labels to priorities
label_to_priority = {
    0: 'medium',  # Sadness
    1: 'low',     # Joy
    2: 'low',     # Love
    3: 'high',    # Anger
    4: 'high',    # Fear
    5: 'high'     # Surprise
}
df['priority'] = df['label'].map(label_to_priority)

# Text preprocessing
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

df['clean_text'] = df['text'].apply(preprocess_text)

# Split data
X = df['clean_text']
y = df['priority']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('clf', LogisticRegression(multi_class='multinomial', solver='lbfgs'))
])

# Train model
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(pipeline, 'priority_model.joblib')
print("Model saved as priority_model.joblib")