import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from pycode.LoadWord2Vec import wv

# Load phrases
phrases_df = pd.read_csv('phrases.csv')
phrases = phrases_df['phrase'].tolist()


def get_phrase_vector(phrase, wv):
    words = phrase.split()
    word_vectors = [wv[word] for word in words if word in wv]
    if not word_vectors:  # If no valid words, return a zero vector
        return np.zeros(wv.vector_size)
    return np.mean(word_vectors, axis=0)  # Use mean to normalize the sum


# Generate phrase vectors for all phrases
phrase_vectors = np.array([get_phrase_vector(phrase, wv) for phrase in phrases])

# Compute pairwise cosine similarity
similarity_matrix = cosine_similarity(phrase_vectors)

# Store results in a CSV file
similarity_df = pd.DataFrame(similarity_matrix, index=phrases, columns=phrases)
similarity_df.to_csv('phrase_similarity.csv')

