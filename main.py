import requests
from bs4 import BeautifulSoup
import pandas as pd
from bertopic import BERTopic
from bertopic.representation import TextGeneration
import numpy as np
from transformers import pipeline
from nltk.tokenize import RegexpTokenizer
import re
from umap.umap_ import UMAP
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
from top2vec import Top2Vec
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


def rescale(x, inplace = False):
    if not inplace:
        x = np.array(x, copy = True)
    x /= np.std(x[:,0]) * 10000
    return x
def LDA(abstract_df):

    #Converts all words to lowercase
    abstracts_df = abstract_df.map (lambda x: x.lower ())

    # Punctuation removal
    abstracts_df = abstract_df.map (lambda x: re.sub ('[,\.!?]', '', x))

    # Tokenization
    tokenizer = RegexpTokenizer (r'\w+')
    tfidf = TfidfVectorizer (lowercase=True,
                             stop_words='english',
                             tokenizer=tokenizer.tokenize)
    data = tfidf.fit_transform (abstract_df)

    model = LatentDirichletAllocation(n_components=9, random_state=0)

    lda_matrix = model.fit_transform (data)
    lda_components = model.components_
    terms = tfidf.get_feature_names_out ()

    i=[]
    keyword_lists= []
    for index, component in enumerate (lda_components):
        zipped = zip (terms, component)
        top_terms_key = sorted (zipped, key=lambda t: t[1], reverse=True)[:10]
        top_terms_list = list (dict (top_terms_key).keys ())
        print ("Topic " + str (index) + ": ", top_terms_list)
        i.append(index)
        keyword_lists.append(top_terms_list)
    df = pd.DataFrame({'0' :i ,
                        'Keywords' : keyword_lists})
    df.to_csv("LDA_topic.csv")

def top2vec(abstract_df):
    #abstracts = abstract_df.to_numpy()
    umap_args = {'n_neighbors': 20,
                 'n_components': 5,
                 'metric': 'cosine',
                 "random_state": 25}
    hdbscan_args = {'min_cluster_size': 20,
                    'min_samples': 5,
                    'metric': 'euclidean',
                    'cluster_selection_method': 'eom'}

    model = Top2Vec (documents=abstract_df.values,speed = 'deep-learn',workers=16, min_count=40,umap_args=umap_args,hdbscan_args=hdbscan_args,
                         embedding_model='doc2vec')
    df = pd.DataFrame(model.topic_words)

    df.to_csv("top2vec_topics.csv")

def bertopic(abstracts_df):
    abstracts = abstracts_df.to_numpy()
    sentence_ = SentenceTransformer("all-MiniLM-L6-v2")
    embedding = sentence_.encode(abstracts)
    pca_embedding = rescale (PCA (n_components=5).fit_transform (embedding))
    vectorizer_model = CountVectorizer(stop_words = "english")
    umap_ = UMAP(n_neighbors = 20, n_components = 5, min_dist= 0.2, metric = 'cosine', init=pca_embedding, random_state=2207)
    bertopic_model = BERTopic(umap_model=umap_, language = 'english', calculate_probabilities=True, vectorizer_model=vectorizer_model )
    topics, probs = bertopic_model.fit_transform(abstracts_df)
    df = pd.DataFrame([topics, probs])
    docs = bertopic_model.get_topic_info()
    docs.to_csv("bertopic_topics.csv")

def main():
    '''''Load Abstracts from CSV files'''''
    BMCbio_2024 = pd.read_csv ("BMC-Bioinformatics_abstracts_2024.csv")
    BMCbio_2023 = pd.read_csv ("BMC-Bioinformatics_abstracts_2023.csv")
    nature_2024 = pd.read_csv ("nature_abstracts_2024.csv")
    nature_2023 = pd.read_csv ("nature_abstracts_2023.csv")
    all_abs = [nature_2024, BMCbio_2024, nature_2023, BMCbio_2023]
    abstracts = pd.concat (all_abs)
    abstracts = abstracts['Abstract']

    LDA(abstracts)
    top2vec (abstracts)
    bertopic(abstracts)

main()
