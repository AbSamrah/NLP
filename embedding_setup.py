import os
import zipfile
import urllib.request
import gzip
import shutil

def download_file(url, filename):
    """
    Downloads a file from the given URL if it doesn't already exist.
    """
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)
        print("Download complete!")
    else:
        print(f"Found {filename}, skipping download.")

def setup_glove():
    """
    Downloads and extracts GloVe embeddings.
    """
    glove_zip = "glove.6B.zip"
    glove_txt = "glove.6B.100d.txt"

    download_file("http://nlp.stanford.edu/data/glove.6B.zip", glove_zip)

    if not os.path.exists(glove_txt):
        print("Extracting GloVe...")
        with zipfile.ZipFile(glove_zip, 'r') as z:
            z.extract(glove_txt)
        print("GloVe extracted!")

    return glove_txt

def setup_google_news_word2vec():
    """
    Downloads Google News Word2Vec embeddings.
    """
    gn_url = "https://figshare.com/ndownloader/files/10798046"
    gn_path = "GoogleNews-vectors-negative300.bin"

    download_file(gn_url, gn_path)

    return gn_path

def setup_wiki_fasttext():
    """
    Downloads and extracts Wiki News FastText embeddings.
    """
    ft_url = "https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip"
    ft_zip = "wiki-news-300d-1M.vec.zip"
    ft_vec = "wiki-news-300d-1M.vec"

    download_file(ft_url, ft_zip)

    if not os.path.exists(ft_vec):
        print("Extracting FastText...")
        with zipfile.ZipFile(ft_zip, 'r') as z:
            z.extract(ft_vec)
        print("FastText extracted!")

    return ft_vec
