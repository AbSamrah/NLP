import pandas as pd
import nltk
import data_loader
import preprocessor
import modeling
import embedding_setup
from embeddings_handler import EmbeddingsHandler
from sklearn.model_selection import train_test_split

def main():
    # Load data
    df = data_loader.load_mini_dataset("mini_hotel_reviews.json")

    # Download NLTK data
    nltk.download('punkt_tab')

    # Preprocessing
    df = preprocessor.create_target_class(df)
    df['clean_tokens'] = df['text'].apply(preprocessor.preprocess_text)

    # Setup embeddings
    embedding_datasets = {}
    handlers = {}

    print("\n" + "="*60)
    print("PHASE 1: Local Training (Domain Specific)")
    print("="*60)

    # Local Word2Vec
    h_w2v = EmbeddingsHandler()
    h_w2v.train_word2vec(df['clean_tokens'], vector_size=100, min_count=2)
    handlers['Local_W2V'] = h_w2v
    X, y = modeling.prepare_embedding_dataset(df, 'clean_tokens', h_w2v)
    embedding_datasets['Local_W2V'] = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    # Local FastText
    h_ft = EmbeddingsHandler()
    h_ft.train_fasttext(df['clean_tokens'], vector_size=100, min_count=2)
    handlers['Local_FT'] = h_ft
    X, y = modeling.prepare_embedding_dataset(df, 'clean_tokens', h_ft)
    embedding_datasets['Local_FastText'] = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    print("\n" + "="*60)
    print("PHASE 2: Pre-trained GloVe")
    print("="*60)

    glove_path = embedding_setup.setup_glove()
    h_glove = EmbeddingsHandler()
    h_glove.load_glove(glove_path)
    handlers['Pre_GloVe'] = h_glove
    X, y = modeling.prepare_embedding_dataset(df, 'clean_tokens', h_glove)
    embedding_datasets['Pre_GloVe_100d'] = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    print("\n" + "="*60)
    print("PHASE 3: Pre-trained Word2Vec (Google News)")
    print("="*60)

    gn_path = embedding_setup.setup_google_news_word2vec()
    h_gn = EmbeddingsHandler()
    h_gn.model = h_gn.load_pretrained_binary(gn_path)
    h_gn.vector_size = 300
    handlers['Pre_GoogleNews'] = h_gn
    X, y = modeling.prepare_embedding_dataset(df, 'clean_tokens', h_gn)
    embedding_datasets['Pre_W2V_GoogleNews'] = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    print("\n" + "="*60)
    print("PHASE 4: Pre-trained FastText (Wiki News)")
    print("="*60)

    ft_path = embedding_setup.setup_wiki_fasttext()
    h_pre_ft = EmbeddingsHandler()
    h_pre_ft.load_pretrained_text(ft_path)
    handlers['Pre_FastText'] = h_pre_ft
    X, y = modeling.prepare_embedding_dataset(df, 'clean_tokens', h_pre_ft)
    embedding_datasets['Pre_FastText_Wiki'] = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    # Semantic comparison
    print("\n" + "="*60)
    print("SEMANTIC COMPARISON: Local vs Global")
    print("="*60)

    test_word = "staff"
    print(f"Neighbors for '{test_word}':")
    for name, h in handlers.items():
        neighbors = h.get_semantic_neighbors(test_word, topn=3)
        print(f"{name:>15}: {neighbors}")

    # Final results
    print("\n" + "="*60)
    print("FINAL RESULTS: Training Multiple Classifiers")
    print("="*60)

    models_emb_config = modeling.get_embedding_models_config()
    modeling.run_model_experiments(embedding_datasets, models_emb_config)

if __name__ == "__main__":
    main()
