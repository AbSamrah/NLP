import re
import spacy
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from nltk.stem import PorterStemmer, SnowballStemmer, LancasterStemmer

# Load SpaCy model once when module is imported
try:
    nlp = spacy.load("en_core_web_sm", disable=["parser"])
except OSError:
    print("SpaCy model not found. Please run: python -m spacy download en_core_web_sm")
    nlp = None

def preprocess_text(text):
    """Clean, tokenize, and remove artifacts."""
    if not isinstance(text, str):
        return [] 
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    return tokens

def replace_with_pos(text):
    """Replaces words with their POS tags (e.g., ADJ, NOUN)."""
    if not nlp: return text
    doc = nlp(text)
    return " ".join([token.pos_ for token in doc])

def replace_entities(text):
    """Replaces named entities with their labels (e.g., DATE, ORG)."""
    if not nlp: return text
    doc = nlp(text)
    new_text = list(text)
    for ent in reversed(doc.ents):
        new_text[ent.start_char:ent.end_char] = ent.label_
    return "".join(new_text)

def generate_ngrams(tokens, n):
    """Generates n-grams from a list of tokens."""
    n_grams = ngrams(tokens, n)
    return ["_".join(gram) for gram in n_grams]

# --- Stemming Wrappers ---

def get_porter_tokens(tokens):
    stemmer = PorterStemmer()
    return [stemmer.stem(t) for t in tokens]

def get_snowball_tokens(tokens):
    stemmer = SnowballStemmer('english')
    return [stemmer.stem(t) for t in tokens]

def get_lancaster_tokens(tokens):
    stemmer = LancasterStemmer()
    return [stemmer.stem(t) for t in tokens]