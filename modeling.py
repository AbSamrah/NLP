import time
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np

def vectorize_all_datasets(df, token_columns, ngram_ranges):
    """
    Vectorizes multiple token columns with specific ngram ranges.
    Returns a dictionary of (X_train, X_test, y_train, y_test).
    """
    print("--- Vectorizing all feature sets ---")
    vectorized_datasets = {} 

    for token_col in token_columns:
        for ngrams in ngram_ranges:
            # Join tokens back to string for TfidfVectorizer
            X = df[token_col].apply(lambda tokens: ' '.join(tokens) if isinstance(tokens, list) else '')
            y = df['class']
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
            
            vectorizer = TfidfVectorizer(max_features=5000, min_df=5, max_df=0.7, ngram_range=ngrams)
            X_train_tfidf = vectorizer.fit_transform(X_train)
            X_test_tfidf = vectorizer.transform(X_test)
            
            dataset_name = f"{token_col}__{ngrams}"
            vectorized_datasets[dataset_name] = (X_train_tfidf, X_test_tfidf, y_train, y_test)
            print(f"Stored: {dataset_name}")

    return vectorized_datasets

def run_model_experiments(vectorized_datasets, models_config):
    """
    Runs training and evaluation (with optional GridSearch) for defined models.
    """
    target_names = ['Negative (0)', 'Positive (1)']
    
    for dataset_name, (X_train, X_test, y_train, y_test) in vectorized_datasets.items():
        for model_name, (model, param_grid) in models_config.items():
            
            print("\n" + "="*80)
            print(f"MODEL: {model_name}   |   DATASET: {dataset_name}")
            print("="*80)
            
            start_time = time.time()
            final_model = model

            if param_grid:
                print("Using GridSearchCV...")
                grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
                grid_search.fit(X_train, y_train)
                final_model = grid_search.best_estimator_
                print(f"Best params: {grid_search.best_params_}")
            else:
                final_model.fit(X_train, y_train)
            
            # Predictions
            y_pred_train = final_model.predict(X_train)
            y_pred_test = final_model.predict(X_test)
            end_time = time.time()
            
            print("\n--- TEST SET Evaluation ---")
            print(classification_report(y_test, y_pred_test, target_names=target_names, digits=4, zero_division=0))
            print(f"Time: {end_time - start_time:.2f}s")

def prepare_embedding_dataset(df, token_col, embeddings_handler):
    """
    Converts a token column into a matrix of averaged word vectors.
    Returns X (features) and y (targets).
    """
    print(f"Generating document vectors using {token_col}...")
    
    # Apply the averaging function to every row
    # usage: df['tokens'].apply(handler.get_document_vector)
    X_series = df[token_col].apply(embeddings_handler.get_document_vector)
    
    # Convert list of arrays into a 2D numpy array (Matrix)
    X = np.vstack(X_series.values)
    y = df['class']
    
    return X, y

def get_default_models_config():
    """
    Returns default model configurations for TF-IDF experiments.
    """
    return {
        'DecisionTree': (DecisionTreeClassifier(random_state=42), {'max_depth': [10, 20]}),
        'LogisticRegression': (LogisticRegression(max_iter=1000), {'C': [0.1, 1.0]}),
        'MLP_NeuralNet': (MLPClassifier(max_iter=500), {'hidden_layer_sizes': [(50,)]})
    }

def get_embedding_models_config():
    """
    Returns model configurations optimized for embedding experiments.
    """
    return {
        # 1. Logistic Regression (Baseline - Fast & Good)
        'LogReg': (
            LogisticRegression(max_iter=1000),
            {'C': [1.0, 10.0]}
        ),

        # 2. Linear SVM (Often the best for text/embeddings)
        'LinearSVC': (
            LinearSVC(dual="auto", max_iter=2000, random_state=42),
            {'C': [0.1, 1.0]}
        ),

        # 3. Neural Network (Captures complex relationships in vectors)
        'MLP_Net': (
            MLPClassifier(max_iter=500, random_state=42),
            {'hidden_layer_sizes': [(100,), (100, 50)]}  # Try 1 layer vs 2 layers
        ),

        # 4. Random Forest (Tree-based, good for non-linear data)
        'RandomForest': (
            RandomForestClassifier(n_estimators=100, random_state=42),
            {'max_depth': [10, 20]}
        )
    }
