import numpy as np
from gensim.models import Word2Vec, FastText, KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import os

class EmbeddingsHandler:
    def __init__(self):
        self.model = None
        self.vector_size = 0

    def train_word2vec(self, sentences, vector_size=100, window=5, min_count=2):
        """Trains a Word2Vec model on the provided tokenized sentences."""
        print(f"Training Word2Vec (size={vector_size}) on {len(sentences)} documents...")
        self.model = Word2Vec(sentences=sentences, vector_size=vector_size, 
                              window=window, min_count=min_count, workers=4)
        self.vector_size = vector_size
        return self.model.wv

    def train_fasttext(self, sentences, vector_size=100, window=5, min_count=2):
        """Trains a FastText model (good for OOV words)."""
        print(f"Training FastText (size={vector_size}) on {len(sentences)} documents...")
        self.model = FastText(sentences=sentences, vector_size=vector_size, 
                              window=window, min_count=min_count, workers=4)
        self.vector_size = vector_size
        return self.model.wv

    def load_glove(self, glove_path):
        """
        Converts GloVe txt format to Word2Vec format and loads it.
        Ex: glove_path = 'glove.6B.50d.txt'
        """
        w2v_output_path = glove_path + ".w2v"
        
        if not os.path.exists(w2v_output_path):
            print("Converting GloVe format to Word2Vec format...")
            glove2word2vec(glove_path, w2v_output_path)
            
        print(f"Loading GloVe model from {w2v_output_path}...")
        self.model = KeyedVectors.load_word2vec_format(w2v_output_path, binary=False)
        self.vector_size = self.model.vector_size
        return self.model

    def load_pretrained_binary(self, path):
        """Loads GoogleNews or other binary .bin files."""
        print(f"Loading binary model from {path}...")
        self.model = KeyedVectors.load_word2vec_format(path, binary=True)
        self.vector_size = self.model.vector_size
        return self.model

    def get_document_vector(self, tokens):
        """
        Creates a single vector for a document by averaging the vectors 
        of its constituent words.
        """
        if self.model is None:
            raise ValueError("Model not loaded or trained yet.")
        
        # Depending on if it's a full model or KeyedVectors
        wv = self.model.wv if hasattr(self.model, 'wv') else self.model
        
        valid_vectors = []
        for word in tokens:
            if word in wv:
                valid_vectors.append(wv[word])
            # FastText can handle OOV, but standard W2V cannot.
            # If using KeyedVectors, we check `word in wv`.

        if not valid_vectors:
            return np.zeros(self.vector_size)
        
        return np.mean(valid_vectors, axis=0)

    # --- Analysis Functions from your notebook ---
    def check_similarity(self, word1, word2):
        wv = self.model.wv if hasattr(self.model, 'wv') else self.model
        try:
            return wv.similarity(word1, word2)
        except KeyError as e:
            return f"Error: {e}"

    def find_similar_words(self, word, topn=5):
        wv = self.model.wv if hasattr(self.model, 'wv') else self.model
        try:
            return wv.most_similar(word, topn=topn)
        except KeyError:
            return None

    def solve_analogy(self, positive, negative):
        """Example: positive=['king', 'woman'], negative=['man']"""
        wv = self.model.wv if hasattr(self.model, 'wv') else self.model
        try:
            return wv.most_similar(positive=positive, negative=negative, topn=1)
        except KeyError as e:
            return f"Error: {e}"