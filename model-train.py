import pandas as pd
import numpy as np
import re
import string
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Manually set file path (Change this based on your upload)
data_path = "/content/spam_or_not_spam.csv"

# Load dataset
df = pd.read_csv(data_path)

# Check for missing values
df.dropna(subset=['email'], inplace=True)

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)  # Remove punctuation
    text = re.sub(r"\d+", "", text)  # Remove numbers
    return text

# Apply text preprocessing
df['email'] = df['email'].apply(preprocess_text)

# Data Visualization: Spam vs. Not Spam Distribution
plt.figure(figsize=(6,4))
sns.countplot(x=df['label'], palette="coolwarm")
plt.title("Spam vs. Not Spam Distribution")
plt.xlabel("Label (0 = Not Spam, 1 = Spam)")
plt.ylabel("Count")
plt.show()

# Generate Word Cloud for Spam Emails
spam_words = " ".join(df[df['label'] == 1]['email'])
wordcloud = WordCloud(width=800, height=400, background_color="black").generate(spam_words)
plt.figure(figsize=(10,5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Most Common Words in Spam Emails")
plt.show()

# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(df['email'], df['label'], test_size=0.2, stratify=df['label'], random_state=42)

# Convert text to numerical features (Better TF-IDF settings)
vectorizer = TfidfVectorizer(stop_words='english', max_features=20000, ngram_range=(1,3))  # Includes unigrams, bigrams, trigrams
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Naïve Bayes model (Improved training)
model = MultinomialNB(alpha=0.1)  # Lower alpha for better discrimination
model.fit(X_train_tfidf, y_train)

# Predictions & Evaluation
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f'🔹 Model Accuracy: {accuracy:.4f}')
print(classification_report(y_test, y_pred))

# Confusion Matrix Visualization
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues", xticklabels=['Not Spam', 'Spam'], yticklabels=['Not Spam', 'Spam'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Save model and vectorizer
model_filename = "naive_bayes_spam.pkl"
vectorizer_filename = "tfidf_vectorizer.pkl"
joblib.dump(model, model_filename)
joblib.dump(vectorizer, vectorizer_filename)
print("✅ Model and vectorizer saved successfully!")