import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from pycode.GetPhrases import phrase_vectors, phrases, get_phrase_vector
from pycode.LoadWord2Vec import wv


def find_closest_phrase(user_phrase, wv, phrase_vectors, phrases):
    user_vector = get_phrase_vector(user_phrase, wv)
    similarities = cosine_similarity([user_vector], phrase_vectors)[0]
    best_match_index = np.argmax(similarities)
    return phrases[best_match_index], similarities[best_match_index]


# Example usage
user_input = "Insurance premiums market in Country"
closest_phrase, distance = find_closest_phrase(user_input, wv, phrase_vectors, phrases)
print(f"Closest match: {closest_phrase}, Distance: {distance}")
