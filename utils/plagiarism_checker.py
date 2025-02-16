import os
import numpy as np
import tensorflow_hub as hub
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load the Universal Sentence Encoder model
try:
    embedding_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
except Exception as e:
    print(f"Error loading embedding model: {e}")
    embedding_model = None  # Ensure the program does not crash

def preprocess_text(text):
    """
    Preprocesses the input text by lowercasing, removing punctuation, stopwords, and applying stemming.
    """
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize
    tokens = nltk.word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Stemming
    ps = PorterStemmer()
    stemmed = [ps.stem(word) for word in tokens]
    return ' '.join(stemmed)

def load_source_data(source_file='source.txt'):
    """
    Loads and preprocesses the source data from source.txt, splitting it into sentences.
    """
    try:
        with open(source_file, 'r', encoding='utf-8') as file:
            raw_text = file.read()
        # Split into sentences
        sentences = nltk.sent_tokenize(raw_text)
        preprocessed_sentences = [preprocess_text(sentence) for sentence in sentences]
        return preprocessed_sentences
    except Exception as e:
        print(f"Error loading source data: {e}")
        return []

def generate_embeddings(text_list):
    """
    Generates embeddings for a list of texts using the TensorFlow embedding model.
    """
    if embedding_model is None:
        print("Embedding model not loaded. Cannot generate embeddings.")
        return np.zeros((len(text_list), 512))  # Return empty embeddings
    return embedding_model(text_list).numpy()

def load_and_prepare_source_embeddings(source_file='source.txt'):
    """
    Loads the source data, preprocesses it, and generates embeddings for each sentence.
    """
    source_sentences = load_source_data(source_file)
    if not source_sentences:
        return [], np.array([])
    source_embeddings = generate_embeddings(source_sentences)
    return source_sentences, source_embeddings

def compute_similarity(upload_embedding, source_embeddings):
    """
    Computes cosine similarity between the uploaded text embedding and all source embeddings.
    Returns the highest similarity score as a percentage.
    """
    if source_embeddings.size == 0:
        return 0  # If there are no source embeddings, return zero similarity

    similarities = cosine_similarity(upload_embedding, source_embeddings)
    max_similarity = similarities.max() * 100  # Convert to percentage
    return max_similarity

def check_plagiarism(new_text, source_embeddings, threshold=50):
    """
    Checks for plagiarism by comparing the new_text with the source_embeddings.
    Returns the similarity score and a status message.
    """
    preprocessed_new = preprocess_text(new_text)
    upload_embedding = generate_embeddings([preprocessed_new])
    score = compute_similarity(upload_embedding, source_embeddings)
    status = 'Plagiarized' if score >= threshold else 'Not Plagiarized'
    return score, status

# Load and prepare source embeddings once at startup
source_sentences, source_embeddings = load_and_prepare_source_embeddings()
