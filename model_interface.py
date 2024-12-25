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
count_vctr_obj = pk.load(open('pickle_file/count_vector.pkl', 'rb'))  # Count Vectorizer
tfidf_obj = pk.load(open('pickle_file/tfidf_transformer.pkl', 'rb'))  # TFIDF Transformer
LR_classifier_model = pk.load(open('pickle_file/final_LR_model.pkl', 'rb'))  # Classification Model
final_user_reco = pk.load(open('pickle_file/final_user_reco_rating.pkl', 'rb'))  # User-User Recommendation System 

language_processor = spacy.load('en_core_web_sm', disable=['ner', 'parser'])

# Read the source csv file into a df
src_df = pd.read_csv('sample30.csv', sep=",") 

#Functions to clean up text

def rem_spec_chars(text, remove_digits=True):
    """Remove the special Characters"""
    pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
    text = re.sub(pattern, '', text)
    return text

def clean_sent(sent):
  sent=sent.lower()  #make the text lowercase
  sent=re.sub('\[[\w\s]*\]',' ',sent) #Remove text in square brackets
  sent=re.sub('[^\w\s]',' ',sent) #Remove punctuation
  sent = rem_spec_chars (sent, True) #Remove special characters
  sent=re.sub('[\w]*[\d]+[\w]*',' ',sent) #Remove words containing numbers
  return sent


stop_words = stopwords.words('english')

#Removing stop words
def rem_stops(sent):
    """Remove stop words"""
    new_sent = []
    for word in sent:
        if word not in stop_words:
            new_sent.append(word)
    return new_sent

#Lemmatize verbs
def lemmat_vbs(sent):
    """Lemmatize verbs only"""
    lmt = WordNetLemmatizer()
    new_sent = []
    for word in sent:
        lemma = lmt.lemmatize(word, pos='v')
        new_sent.append(lemma)
    return new_sent

#Consolidate functions of removing stop words and lemmatization
def stop_and_lemmat(input_text):
    tokens = nltk.word_tokenize(input_text)
    word_list = rem_stops(tokens)
    lemma_list = lemmat_vbs(word_list)
    return ' '.join(lemma_list)


# Predicting sentiment of product reviews
def predict_sentiment(review_text):
    word_vector = count_vctr_obj.transform(review_text)
    tfidf_vector = tfidf_obj.transform(word_vector)
    return LR_classifier_model.predict(tfidf_vector)



# Recommend top 20 products based on user recommendation system
def recommend_top20_products(user_id):
    top20_products_user = pd.DataFrame(final_user_reco.loc[user_id].sort_values(ascending=False)[:20])
    products_subset_df = src_df[src_df.name.isin(top20_products_user.index.tolist())]
    products_subset_df['reviews_text_cln'] = products_subset_df['reviews_text'].apply(lambda x:clean_sent(x))
    products_subset_df['lemmatized_review'] = products_subset_df['reviews_text_cln'].apply(lambda text: stop_and_lemmat(text))
    products_subset_df['sentiment'] = predict_sentiment(products_subset_df['lemmatized_review'])
    return products_subset_df

# Recommend top 5 products based on sentiment analysis
def recommend_top5_products(df):
    """Identify the top 5 products based on sentiment percentage"""
    cnt_product_total_reviews = df.groupby(['name']).agg('count')
    cnt_sentiment_totals = df.groupby(['name', 'sentiment']).agg('count').reset_index()
    df_merged_data = pd.merge(cnt_sentiment_totals, cnt_product_total_reviews['reviews_text'], on='name')
    df_merged_data['percentage'] = (df_merged_data['reviews_text_x'] / df_merged_data['reviews_text_y']) * 100
    df_merged_data = df_merged_data.sort_values(ascending=False, by='percentage')
    top5_products = pd.DataFrame(df_merged_data['name'][df_merged_data['sentiment'] == 1][:5])
    return top5_products