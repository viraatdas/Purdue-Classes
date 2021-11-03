# define files

query_file = "allqueries.json"
relevance_file = "project1.rel"
relevance_fullpath_file = "project1_fullpath.rel"
index_dir = "project1-index"


# load output from dump-corpus
from subprocess import check_output, STDOUT
import os
os.environ.copy()
out = check_output(['galago', 'dump-corpus', f'--path={index_dir}/corpus'], universal_newlines=True)


import re
doc_names = re.findall('#NAME: (.*)', out)
text = re.findall(r"(?:<pre>)\s*(.*?)\s*(?!\1)(?:</pre>)", out, flags=re.DOTALL)


import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

stemmer = PorterStemmer()
analyzer = TfidfVectorizer().build_analyzer()


def stemmed_words(doc):
    return (stemmer.stem(w) for w in analyzer(doc))


vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'), analyzer=stemmed_words)
docs_tfidf = vectorizer.fit_transform(text)

def get_tfidf_query(vectorizer, docs_tfidf, query):
    query_tfidf = vectorizer.transform([query])
    return cosine_similarity(query_tfidf, docs_tfidf).flatten()

def preprocess_query(query):
    query = query.replace('#combine(','')
    query = query.replace(' )','')
    return query

def print_cosine_scores(query_id, score_array):
    import numpy as np
    sorted_indices = np.argsort(score_array)[::-1]
    sorted_score_array = np.sort(score_array)[::-1]
    for i, (sorted_index, sorted_score) in enumerate(zip(sorted_indices, sorted_score_array)):
        if sorted_score == 0:
            continue
        print(f"{query_id} Q0 {doc_names[sorted_index]} {i+1} {sorted_score:10.10f} galago")

import json
queries = json.load(open(query_file))

for el in queries['queries']:
    query = preprocess_query(el['text'])
    query_id = el['number']
    cosine_scores = get_tfidf_query(vectorizer, docs_tfidf, query)

    print_cosine_scores(query_id, cosine_scores)

