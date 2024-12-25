# This file provides the model interface - this calls the model
## User: Nitin Balaji Srinivasan AI& ML (Cohort 58)
# Importing Libraries
from nltk.corpus.reader import reviews
import pandas as pd
import re, nltk, spacy, string
import en_core_web_sm
import pickle as pk

from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import LancasterStemmer
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load the pickle files 
vectorizer = pk.load(open('pickle_file/count_vector.pkl', 'rb'))  # Count Vectorizer
transformer = pk.load(open('pickle_file/tfidf_transformer.pkl', 'rb'))  # TFIDF Transformer
classifier = pk.load(open('pickle_file/final_LR_model.pkl', 'rb'))  # Classification Model
user_recommendations = pk.load(open('pickle_file/final_user_reco_rating.pkl', 'rb'))  # User-User Recommendation System 

language_processor = spacy.load('en_core_web_sm', disable=['ner', 'parser'])

data_frame = pd.read_csv('sample30.csv', sep=",")

# Special characters removal
def clean_text(text, remove_numbers=True):
    """Remove special characters from text"""
    pattern = r'[^a-zA-Z0-9\s]' if not remove_numbers else r'[^a-zA-Z\s]'
    text = re.sub(pattern, '', text)
    return text

def convert_to_lowercase(tokens):
    """Convert all tokens to lowercase"""
    return [token.lower() for token in tokens]

def remove_punctuation_and_specials(tokens):
    """Remove punctuation and special characters from tokens"""
    cleaned_tokens = []
    for token in tokens:
        token = re.sub(r'[^\w\s]', '', token)
        if token:
            token = clean_text(token, True)
            cleaned_tokens.append(token)
    return cleaned_tokens

stop_words = stopwords.words('english')

def filter_stopwords(tokens):
    """Remove stopwords from tokens"""
    return [token for token in tokens if token not in stop_words]

def apply_stemming(tokens):
    """Stem the tokens"""
    stemmer = LancasterStemmer()
    return [stemmer.stem(token) for token in tokens]

def apply_lemmatization(tokens):
    """Lemmatize the tokens"""
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token, pos='v') for token in tokens]

def preprocess_text(tokens):
    """Preprocess tokens by normalizing and removing special characters"""
    tokens = convert_to_lowercase(tokens)
    tokens = remove_punctuation_and_specials(tokens)
    tokens = filter_stopwords(tokens)
    return tokens

def lemmatize_tokens(tokens):
    """Lemmatize tokens"""
    return apply_lemmatization(tokens)

# Predicting sentiment of product reviews
def predict_sentiment(review_text):
    vectorized_text = vectorizer.transform(review_text)
    transformed_text = transformer.transform(vectorized_text)
    return classifier.predict(transformed_text)

def process_and_lemmatize(text):
    """Clean, tokenize, and lemmatize text"""
    text = clean_text(text)
    tokens = nltk.word_tokenize(text)
    tokens = preprocess_text(tokens)
    lemmatized_tokens = lemmatize_tokens(tokens)
    return ' '.join(lemmatized_tokens)

# Recommend products based on sentiment analysis
def suggest_products(user_id):
    recommendations = pk.load(open('pickle_file/user_final_rating.pkl', 'rb'))
    user_top_products = pd.DataFrame(recommendations.loc[user_id].sort_values(ascending=False)[:20])
    filtered_products = data_frame[data_frame.name.isin(user_top_products.index.tolist())]
    filtered_products['cleaned_text'] = filtered_products['reviews_text'].map(lambda text: process_and_lemmatize(text))
    filtered_products['sentiment'] = predict_sentiment(filtered_products['cleaned_text'])
    return filtered_products

def top_rated_products(data):
    """Identify the top 5 products based on sentiment percentage"""
    product_totals = data.groupby(['name']).agg('count')
    sentiment_totals = data.groupby(['name', 'sentiment']).agg('count').reset_index()
    merged_data = pd.merge(sentiment_totals, product_totals['reviews_text'], on='name')
    merged_data['percentage'] = (merged_data['reviews_text_x'] / merged_data['reviews_text_y']) * 100
    merged_data = merged_data.sort_values(ascending=False, by='percentage')
    top_products = pd.DataFrame(merged_data['name'][merged_data['sentiment'] == 1][:5])
    return top_products