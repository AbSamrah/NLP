import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter

def plot_top_ngrams(token_series, n_grams_label="Unigrams", top_k=10, palette='viridis'):
    """
    Plots the most frequent n-grams (or tokens) from a pandas Series of lists.
    """
    sns.set_style('whitegrid')
    
    # Flatten list
    all_tokens = [token for sublist in token_series for token in sublist]
    
    # Count
    counts = Counter(all_tokens)
    top_items = counts.most_common(top_k)
    
    # Create DF
    df_plot = pd.DataFrame(top_items, columns=['term', 'frequency'])
    
    # Plot
    plt.figure(figsize=(12, 8))
    sns.barplot(x='frequency', y='term', data=df_plot, palette=palette)
    plt.title(f'Top {top_k} Most Frequent {n_grams_label}')
    plt.xlabel('Frequency')
    plt.ylabel(n_grams_label)
    plt.show()