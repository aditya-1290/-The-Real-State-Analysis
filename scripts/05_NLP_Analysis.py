import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
from sklearn.decomposition import LatentDirichletAllocation
import nltk
nltk.download('punkt')

# Sample text data (replace with actual data)
texts = [
    "Beautiful modern home with spacious garden and updated kitchen.",
    "Cozy family house in quiet neighborhood with large backyard.",
    "Luxury condo with stunning city views and high-end finishes.",
    "Historic Victorian home with original features and charm.",
    "Contemporary apartment with open floor plan and modern amenities."
]

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
print(f'TF-IDF Matrix Shape: {tfidf_matrix.shape}')

# Sentiment Analysis
sentiments = []
for text in texts:
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    sentiments.append(sentiment)
    print(f'Text: {text[:50]}... Sentiment: {sentiment:.2f}')

# Topic Modeling with LDA
lda = LatentDirichletAllocation(n_components=2, random_state=42)
lda_topics = lda.fit_transform(tfidf_matrix)
print(f'LDA Topics Shape: {lda_topics.shape}')

# Display top words for each topic
feature_names = tfidf_vectorizer.get_feature_names_out()
for topic_idx, topic in enumerate(lda.components_):
    print(f'Topic {topic_idx}:')
    top_words = [feature_names[i] for i in topic.argsort()[:-11:-1]]
    print(' '.join(top_words))
