import argparse

import pandas as pd
from gensim.models import KeyedVectors
import numpy as np
import pandas as pd
from pycode.FindClosestPhrases import find_closest_phrase
from pycode.GetPhrases import get_phrase_vector


class PhraseSimilarityApp:
    def __init__(self, word_vector_file, phrase_file):
        self.wv = KeyedVectors.load_word2vec_format(word_vector_file, binary=False)
        self.phrases_df = pd.read_csv(phrase_file)
        self.phrases = self.phrases_df['phrase'].tolist()
        self.phrase_vectors = np.array([get_phrase_vector(phrase, self.wv) for phrase in self.phrases])

    def find_similarity(self, user_phrase):
        closest_phrase, distance = find_closest_phrase(user_phrase, self.wv, self.phrase_vectors, self.phrases)
        return closest_phrase, distance


def main():
    parser = argparse.ArgumentParser(description='Phrase Similarity CLI')
    parser.add_argument('--phrase', type=str, help='Input phrase to find similarity')
    args = parser.parse_args()

    app = PhraseSimilarityApp('vectors.csv', 'phrases.csv')
    closest_phrase, distance = app.find_similarity(args.phrase)
    print(f"Closest match: {closest_phrase}, Distance: {distance}")


if __name__ == "__main__":
    main()
