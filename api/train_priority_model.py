from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import joblib

# Load data
data = pd.read_csv('training.csv')
X = data['text']
y = data['label']

# Vectorize text
vectorizer = TfidfVectorizer(max_features=500)  # Reduced features for smaller model
X_vec = vectorizer.fit_transform(X)

# Train model
model = LogisticRegression(max_iter=1000, multi_class='ovr')
model.fit(X_vec, y)

# Save model and vectorizer
joblib.dump(model, 'api/priority_model.joblib')
joblib.dump(vectorizer, 'api/vectorizer.joblib')